//! Axum routes, authentication, protocol dispatch, and graceful shutdown.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, HeaderMap, HeaderValue, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Value};

use crate::anthropic::{self, ProtocolError};
use crate::cache;
use crate::config::{defaults, Config};
use crate::optimize;
use crate::pool::KeyPool;
use crate::redaction::{sanitize_json, sanitize_text};
use crate::state::{self, StateLoadOutcome};
use crate::upstream::{
    transport_error_class, BufferedUpstreamError, ClineUpstream, UpstreamError, UpstreamResponse,
    UpstreamResult,
};

const MAX_UPSTREAM_BODY_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub upstream: ClineUpstream,
    pub state_file: Option<Arc<PathBuf>>,
    pub persistence_healthy: Arc<AtomicBool>,
    pub started: Instant,
    /// Bounding permits for the CPU-bound exact tokenizer background jobs
    /// (`glm53.telemetry.max_concurrent_token_counts`). `None` disables the
    /// accounting entirely, which is handled before acquisition.
    pub token_count_permits: Option<Arc<tokio::sync::Semaphore>>,
}

impl AppState {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let pool = KeyPool::new(&config.cline_api_keys);
        if pool.is_empty() {
            anyhow::bail!("at least one enabled Cline API key is required");
        }
        let state_file = config.state_file_path().map(|path| Arc::new(path.clone()));
        if let Some(path) = &state_file {
            match state::load(path) {
                StateLoadOutcome::Loaded(persisted) => {
                    let summary = pool.restore(&persisted);
                    tracing::info!(
                        state_file = %path.display(),
                        restored_cooldowns = summary.restored_cooldowns,
                        expired_entries = summary.expired_entries,
                        unknown_entries = summary.unknown_entries,
                        restored_active_key = summary.restored_active_key.as_deref().unwrap_or(""),
                        "runtime state loaded"
                    );
                }
                StateLoadOutcome::Missing => {
                    tracing::info!(
                        state_file = %path.display(),
                        "runtime state file not present; starting with empty state"
                    );
                }
                StateLoadOutcome::Corrupt(reason) => {
                    // Corrupt state is advisory only; never block startup and
                    // never echo file contents (defense against unexpected
                    // secret-shaped data in a damaged file).
                    tracing::warn!(
                        state_file = %path.display(),
                        reason,
                        "runtime state file was unreadable; starting with empty state"
                    );
                }
            }
        }
        let upstream = ClineUpstream::new(&config, pool)?;
        let token_count_permits = if config.glm53.telemetry.exact_input_tokens {
            Some(Arc::new(tokio::sync::Semaphore::new(
                config.glm53.telemetry.max_concurrent_token_counts.max(1) as usize,
            )))
        } else {
            None
        };
        Ok(Self {
            config: Arc::new(config),
            upstream,
            state_file,
            persistence_healthy: Arc::new(AtomicBool::new(true)),
            started: Instant::now(),
            token_count_permits,
        })
    }

    /// One synchronous debounced-state flush. Called by the writer task and
    /// once more during graceful shutdown. Never called on the request path.
    pub fn flush_runtime_state(&self) {
        let Some(path) = self.state_file.as_ref() else {
            return;
        };
        let persisted = self.upstream.pool().persisted_state();
        match state::store(path, &persisted) {
            Ok(()) => {
                self.persistence_healthy.store(true, Ordering::Relaxed);
            }
            Err(error) => {
                self.persistence_healthy.store(false, Ordering::Relaxed);
                tracing::error!(
                    state_file = %path.display(),
                    error = %error,
                    "failed to persist runtime state; cooldowns will be lost on restart"
                );
            }
        }
    }
}

pub fn router(state: AppState) -> Router {
    let max_request_bytes = state.config.runtime.max_request_bytes;
    let protected = Router::new()
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .route("/v1/chat/completions", post(openai_chat))
        .route("/v1/models", get(models))
        .route("/admin/status", get(admin_status))
        .layer(DefaultBodyLimit::max(max_request_bytes))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));
    Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .merge(protected)
        .layer(middleware::from_fn(response_log_middleware))
        .layer(middleware::from_fn(request_id_middleware))
        .with_state(state)
}

async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok\n")
}

async fn readyz(State(state): State<AppState>) -> Response {
    if state.upstream.pool().is_empty() {
        return (StatusCode::SERVICE_UNAVAILABLE, "not ready\n").into_response();
    }
    (StatusCode::OK, "ready\n").into_response()
}

/// Authenticated, read-only operational snapshot. Never returns key material,
/// authorization values, or raw upstream error text.
async fn admin_status(State(state): State<AppState>) -> Response {
    let pool = state.upstream.pool();
    let keys = pool
        .snapshots()
        .into_iter()
        .map(|snapshot| {
            json!({
                "name": snapshot.name.as_ref(),
                "state": snapshot.phase.as_str(),
                "cooldown_remaining_seconds": snapshot.cooldown_remaining.map(|d| d.as_secs()),
                "cooldown_until_unix": snapshot.cooldown_until
                    .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                    .map(|duration| duration.as_secs()),
                "rate_limit_kind": snapshot.rate_limit_kind.map(|kind| kind.as_str()),
                "rate_limited_model": snapshot.last_429_model.as_deref().unwrap_or(""),
                "last_429_at_unix": snapshot.last_429_at
                    .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                    .map(|duration| duration.as_secs()),
                "last_success_at_unix": snapshot.last_success_at
                    .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                    .map(|duration| duration.as_secs()),
                "requests": snapshot.requests,
                "successes": snapshot.successes,
                "rate_limits": snapshot.rate_limits,
            })
        })
        .collect::<Vec<_>>();
    let body = json!({
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": state.started.elapsed().as_secs(),
        "default_model": state.config.models.default,
        "active_key": pool.active_key_name().map(|name| name.to_string()),
        "active_key_age_seconds": pool.active_age().as_secs(),
        "state_persistence": {
            "enabled": state.state_file.is_some(),
            "healthy": state.persistence_healthy.load(Ordering::Relaxed),
        },
        "keys": keys,
    });
    let request_id = "req_admin_status";
    json_response(StatusCode::OK, body, request_id)
}

async fn models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let request_id = request_id(&headers);
    let ids = state.config.model_ids();
    let data = ids
        .iter()
        .map(|id| {
            json!({
                "id":id,
                "object":"model",
                "created":0,
                "owned_by":"cline-proxy"
            })
        })
        .collect::<Vec<_>>();
    if headers.contains_key("anthropic-version") {
        let anthropic_models = ids
            .iter()
            .map(|id| {
                json!({
                    "type":"model", "id":id, "display_name":id,
                    "created_at":"1970-01-01T00:00:00Z"
                })
            })
            .collect::<Vec<_>>();
        return json_response(
            StatusCode::OK,
            json!({"data":anthropic_models,"has_more":false,
                "first_id":ids.first(),"last_id":ids.last()}),
            &request_id,
        );
    }
    json_response(
        StatusCode::OK,
        json!({"object":"list", "data":data}),
        &request_id,
    )
}

async fn openai_chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let request_id = request_id(&headers);
    let body = match request_body(body, false, &request_id) {
        Ok(body) => body,
        Err(response) => return *response,
    };
    let mut value: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(error) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("invalid JSON: {error}"),
                &request_id,
            )
        }
    };
    let Some(object) = value.as_object_mut() else {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "request body must be a JSON object",
            &request_id,
        );
    };
    let Some(requested_model) = object
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.is_empty())
        .map(ToOwned::to_owned)
    else {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "model must be a non-empty string",
            &request_id,
        );
    };
    let stream = match object.get("stream") {
        Some(Value::Bool(stream)) => *stream,
        Some(_) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                "stream must be a boolean",
                &request_id,
            )
        }
        None => false,
    };
    let upstream_model = state.config.resolve_model(&requested_model);
    object.insert("model".into(), Value::String(upstream_model.clone()));
    // GLM policy for OpenAI-protocol clients: explicit reasoning effort
    // (default high, never unset), bounded output, historical-reasoning
    // strip, safe compaction.
    let optimization =
        match optimize::optimize_request(&mut value, &state.config.glm53, optimize::Origin::OpenAi)
        {
            Ok(optimization) => optimization,
            Err(message) => return openai_error(StatusCode::BAD_REQUEST, message, &request_id),
        };
    // Canonicalize historical tool-call argument JSON for byte-stable
    // prefixes (config-gated; plain-text content is never touched).
    let canonicalized = if state.config.glm53.context.canonical_tool_json {
        cache::canonicalize_tool_arguments(
            value.as_object_mut().unwrap_or(&mut serde_json::Map::new()),
        )
    } else {
        0
    };
    let prefix_telemetry = cache::log_prefix_telemetry(&value, None, &request_id);
    log_request_optimization(
        &request_id,
        "openai",
        &optimization,
        0,
        canonicalized,
        &prefix_telemetry,
    );
    let upstream_body = match serde_json::to_vec(&value) {
        Ok(body) => Bytes::from(body),
        Err(_) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                "request could not be serialized",
                &request_id,
            )
        }
    };
    tracing::info!(
        request_id,
        protocol = "openai",
        requested_model,
        upstream_model,
        stream,
        request_bytes = upstream_body.len(),
        "client request accepted"
    );
    let started = Instant::now();
    match state
        .upstream
        .send_chat(upstream_body, stream, &request_id, &upstream_model)
        .await
    {
        Ok(result) => openai_upstream_response(&state, result, &request_id, stream, started).await,
        Err(error) => upstream_failure(&state, error, false, &request_id),
    }
}

async fn anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let request_id = request_id(&headers);
    let body = match request_body(body, true, &request_id) {
        Ok(body) => body,
        Err(response) => return *response,
    };
    let mut converted = match anthropic::convert_request(&body) {
        Ok(converted) => converted,
        Err(error) => return protocol_error(error, StatusCode::BAD_REQUEST, &request_id),
    };
    let requested_model = converted.model.clone();
    let upstream_model = state.config.resolve_model(&requested_model);
    anthropic::apply_model(&mut converted, upstream_model.clone());
    // Prefix stability (issue #8): remove the volatile leading
    // x-anthropic-billing-header line before anything else observes the
    // system text, so the wire body, count_tokens, and the prefix hash all
    // see the same normalized system.
    let billing_header_bytes_removed = if state.config.glm53.context.strip_volatile_billing_header {
        anthropic::normalize_system_messages(&mut converted.body)
    } else {
        0
    };
    // Session fingerprint: extracted from metadata (if present) before the
    // GLM policy may drop it; only the HMAC fingerprint is ever logged.
    let session_fp = state
        .config
        .glm53
        .telemetry
        .prefix_hash
        .then(|| cache::extract_session_fingerprint(&body, &state.config.server.api_key))
        .flatten();
    // GLM policy: explicit reasoning effort (default high, never unset),
    // bounded output, historical-thinking strip, safe compaction, and the
    // per-request size breakdown telemetry.
    let optimization = match optimize::optimize_request(
        &mut converted.body,
        &state.config.glm53,
        optimize::Origin::Anthropic {
            thinking: converted.thinking.as_ref(),
            output_effort: converted.output_effort.as_deref(),
        },
    ) {
        Ok(optimization) => optimization,
        Err(message) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
                &request_id,
            )
        }
    };
    converted.expose_thinking = optimization.expose_thinking;
    // Canonicalize historical tool-call argument JSON for byte-stable
    // prefixes (config-gated; plain-text content is never touched). The
    // converted body is always an object at this point.
    let canonicalized = if state.config.glm53.context.canonical_tool_json {
        converted
            .body
            .as_object_mut()
            .map(cache::canonicalize_tool_arguments)
            .unwrap_or(0)
    } else {
        0
    };
    let prefix_telemetry =
        cache::log_prefix_telemetry(&converted.body, session_fp.as_deref(), &request_id);
    log_request_optimization(
        &request_id,
        "anthropic",
        &optimization,
        billing_header_bytes_removed,
        canonicalized,
        &prefix_telemetry,
    );
    // Exact tokenizer telemetry is GLM-specific; generic-model requests have
    // no embedded tokenizer and skip it entirely.
    if optimization.model_family == optimize::ModelFamily::Glm53 {
        spawn_exact_token_telemetry(
            state.clone(),
            body.clone(),
            optimization.reasoning_effort,
            request_id.clone(),
        );
    }
    let upstream_body = match serde_json::to_vec(&converted.body) {
        Ok(body) => Bytes::from(body),
        Err(_) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "request could not be serialized",
                &request_id,
            )
        }
    };
    tracing::info!(
        request_id,
        protocol = "anthropic",
        requested_model,
        upstream_model,
        stream = converted.stream,
        request_bytes = body.len(),
        upstream_request_bytes = upstream_body.len(),
        "client request accepted"
    );
    let started = Instant::now();
    let result = match state
        .upstream
        .send_chat(
            upstream_body,
            converted.stream,
            &request_id,
            &upstream_model,
        )
        .await
    {
        Ok(result) => result,
        Err(error) => return upstream_failure(&state, error, true, &request_id),
    };
    let status = result.response.status();
    tracing::info!(
        request_id,
        selected_key_name = %result.selected.name,
        selected_key_index = result.selected.configured_index,
        attempt = result.attempt,
        failover_count = result.failover_count,
        upstream_status = status.as_u16(),
        "Anthropic upstream attempt selected"
    );
    let response = match result.response {
        UpstreamResponse::Success(response) => response,
        UpstreamResponse::HttpError(error) => {
            return sanitized_upstream_error(&state, error, true, &request_id)
        }
    };
    if converted.stream {
        let mut response = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header("x-accel-buffering", "no")
            .body(anthropic::stream_body(
                response,
                request_id.clone(),
                upstream_model,
                result.selected.name.to_string(),
                started,
                state.config.runtime.stream_progress_secs,
                converted.expose_thinking,
            ))
            .unwrap_or_else(|_| Response::new(Body::empty()));
        insert_request_id(response.headers_mut(), &request_id);
        return response;
    }
    let value = match anthropic::parse_json_response(response).await {
        Ok(value) => value,
        Err(error) => return protocol_error(error, StatusCode::BAD_GATEWAY, &request_id),
    };
    match anthropic::convert_response(
        &value,
        &request_id,
        &upstream_model,
        converted.expose_thinking,
    ) {
        Ok(value) => json_response(StatusCode::OK, value, &request_id),
        Err(error) => protocol_error(error, StatusCode::BAD_GATEWAY, &request_id),
    }
}

async fn anthropic_count_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let request_id = request_id(&headers);
    let body = match request_body(body, true, &request_id) {
        Ok(body) => body,
        Err(response) => return *response,
    };
    let value: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(error) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("invalid JSON: {error}"),
                &request_id,
            )
        }
    };
    let Some(requested_model) = value
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.is_empty())
    else {
        return anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "model must be a non-empty string",
            &request_id,
        );
    };
    let upstream_model = state.config.resolve_model(requested_model);
    let started = Instant::now();
    // The exact counter embodies the GLM official template; non-GLM models
    // have no exact counter and must not be silently counted as GLM prompts.
    if optimize::ModelFamily::from_upstream_model(&upstream_model) != optimize::ModelFamily::Glm53 {
        return anthropic_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request_error",
            format!(
                "exact token counting is only available for GLM models (requested upstream model: {upstream_model})"
            ),
            &request_id,
        );
    }
    // Count what the gateway would actually send: the volatile billing
    // header stripped (same normalization as the wire path — issue #8),
    // historical thinking stripped (when the policy strips), and the
    // resolved reasoning effort applied, so Claude Code's context budgeting
    // matches real upstream usage. Falls back to the official-oracle count
    // when the policies are disabled.
    let mut counted = value.clone();
    if state.config.glm53.context.strip_volatile_billing_header {
        // The count pipeline consumes the *Anthropic* shape, so the
        // system-level strip is applied to the `system` field directly.
        cache::strip_billing_header_in_anthropic_system(&mut counted);
    }
    if state.config.glm53.reasoning.strip_historical_thinking {
        optimize::strip_anthropic_thinking(&mut counted);
    }
    let effort = match crate::glm53::reasoning::resolve_reasoning_policy(
        value.get("thinking"),
        value
            .get("output_config")
            .and_then(|config| config.get("effort"))
            .and_then(Value::as_str),
        value
            .get("max_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(u64::MAX),
        state.config.glm53.reasoning.default_effort,
        state.config.glm53.reasoning.adaptive_effort,
        state.config.glm53.reasoning.expose_thinking,
    ) {
        Ok(policy) => policy,
        Err(message) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
                &request_id,
            )
        }
    };
    let count_method = if state.config.glm53.reasoning.strip_historical_thinking {
        counted["output_config"] = json!({"effort": effort.effort.as_str()});
        "exact_glm53_optimized"
    } else {
        "exact_glm53"
    };
    let (count, count_method) = match crate::glm53::count::count_input_tokens(&counted) {
        Ok(count) => (count, count_method),
        Err(error) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                error.error_type,
                error.message,
                &request_id,
            )
        }
    };
    tracing::info!(
        request_id,
        protocol = "anthropic",
        requested_model,
        upstream_model,
        input_tokens = count,
        token_count = count_method,
        count_micros = started.elapsed().as_micros(),
        "token count completed"
    );
    let mut response = json_response(StatusCode::OK, json!({"input_tokens":count}), &request_id);
    if let Ok(value) = HeaderValue::from_str(count_method) {
        response
            .headers_mut()
            .insert("x-cline-proxy-token-count", value);
    }
    response
}

/// One log line attributing where request bytes went and what the GLM
/// policy decided. Sizes and counts only — never request content.
fn log_request_optimization(
    request_id: &str,
    protocol: &str,
    optimization: &optimize::RequestOptimization,
    billing_header_bytes_removed: u64,
    canonicalized_arguments: usize,
    prefix_telemetry: &(String, usize),
) {
    tracing::info!(
        request_id,
        protocol,
        model_family = match optimization.model_family {
            optimize::ModelFamily::Glm53 => "glm53",
            optimize::ModelFamily::GenericOpenAi => "generic_openai",
        },
        reasoning_effort = optimization.reasoning_effort,
        thinking_exposure = if optimization.expose_thinking {
            "exposed"
        } else {
            "suppressed"
        },
        client_max_tokens = optimization.client_max_tokens.unwrap_or(0),
        effective_max_tokens = optimization.effective_max_tokens.unwrap_or(0),
        openai_bytes_before = optimization.before_bytes,
        openai_bytes_after = optimization.after_bytes,
        system_bytes = optimization.system_bytes,
        messages_bytes = optimization.messages_bytes,
        tools_bytes = optimization.tools_bytes,
        other_bytes = optimization.other_bytes(),
        historical_reasoning_bytes_removed = optimization.historical_reasoning_bytes_removed,
        billing_header_bytes_removed,
        canonicalized_arguments,
        empty_blocks_removed = optimization.empty_blocks_removed,
        normalized_text_blocks = optimization.normalized_text_blocks,
        prefix_hash = %prefix_telemetry.0,
        prefix_bytes = prefix_telemetry.1,
        "request optimization"
    );
}

/// Exact GLM-5.3-Flash token accounting for the optimized request, run in a
/// **bounded** `spawn_blocking` task so the embedded-official tokenizer (a
/// CPU-bound, hundreds-of-ms job on megabyte-scale prompts) never occupies a
/// Tokio worker thread or stacks up unbounded background work. Never on the
/// TTFT path. Reports:
/// - `input_tokens`: exact count of what is actually sent (historical
///   thinking stripped, resolved effort applied)
/// - `estimated_tokens_removed_historical_reasoning`: tokenization of the
///   stripped reasoning chunks alone — an *estimate* of what the full request
///   would have cost (exact would require a second full pass over the
///   pre-strip request, which production does not pay for)
/// - derived before/after ratio (before = after + removed estimate)
fn spawn_exact_token_telemetry(
    state: AppState,
    request_bytes: Bytes,
    reasoning_effort: &'static str,
    request_id: String,
) {
    if !state.config.glm53.telemetry.exact_input_tokens {
        return;
    }
    let Some(permits) = state.token_count_permits.clone() else {
        return;
    };
    // The tokenizer is embedded and CPU-bound; one permit at a time by
    // default. When the permit is busy, telemetry is skipped rather than
    // queued: losing a count beats piling up megabyte-scale jobs. The owned
    // permit keeps the semaphore alive for the count's duration.
    let Ok(permit) = permits.clone().try_acquire_owned() else {
        tracing::info!(
            request_id,
            token_telemetry_skipped_busy = true,
            "exact token telemetry skipped: another count is in flight"
        );
        return;
    };
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let started = Instant::now();
        let Ok(mut request) = serde_json::from_slice::<Value>(&request_bytes) else {
            return;
        };
        let removed_tokens = match optimize::removed_reasoning_tokens(&request) {
            Ok(tokens) => tokens,
            Err(_) => return, // uncountable content; bytes telemetry still applies
        };
        optimize::strip_anthropic_thinking(&mut request);
        // Align the count with the effort actually placed on the wire.
        request["output_config"] = json!({"effort": reasoning_effort});
        let Ok(input_tokens) = crate::glm53::count::count_input_tokens(&request) else {
            return;
        };
        let before_estimate = u64::from(input_tokens).saturating_add(removed_tokens);
        let saved_percent = if before_estimate > 0 {
            (removed_tokens as f64 / before_estimate as f64 * 100.0 * 10.0).round() / 10.0
        } else {
            0.0
        };
        tracing::info!(
            request_id,
            input_tokens,
            estimated_input_tokens_before = before_estimate,
            estimated_tokens_removed_historical_reasoning = removed_tokens,
            saved_percent,
            count_method = "exact_glm53_optimized",
            count_duration_ms = started.elapsed().as_millis(),
            "exact GLM token accounting for optimized request"
        );
    });
}

async fn openai_upstream_response(
    state: &AppState,
    result: UpstreamResult,
    request_id: &str,
    requested_stream: bool,
    started: Instant,
) -> Response {
    let status = result.response.status();
    tracing::info!(
        request_id,
        selected_key_name = %result.selected.name,
        selected_key_index = result.selected.configured_index,
        attempt = result.attempt,
        failover_count = result.failover_count,
        upstream_status = status.as_u16(),
        "OpenAI upstream attempt selected"
    );
    let response = match result.response {
        UpstreamResponse::Success(response) => response,
        UpstreamResponse::HttpError(error) => {
            return sanitized_upstream_error(state, error, false, request_id)
        }
    };
    if requested_stream {
        let mut response = Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header("x-accel-buffering", "no")
            .body(openai_stream_body(
                response,
                request_id.to_owned(),
                result.selected.name.to_string(),
                started,
            ))
            .unwrap_or_else(|_| Response::new(Body::empty()));
        insert_request_id(response.headers_mut(), request_id);
        return response;
    }
    let bytes = match read_response_limited(response, MAX_UPSTREAM_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(message) => return openai_error(StatusCode::BAD_GATEWAY, message, request_id),
    };
    if serde_json::from_slice::<Value>(&bytes).is_err() {
        return openai_error(
            StatusCode::BAD_GATEWAY,
            "upstream returned invalid JSON",
            request_id,
        );
    }
    let mut response =
        (status, [(header::CONTENT_TYPE, "application/json")], bytes).into_response();
    insert_request_id(response.headers_mut(), request_id);
    response
}

fn openai_stream_body(
    response: reqwest::Response,
    request_id: String,
    key_name: String,
    started: Instant,
) -> Body {
    let output = async_stream::stream! {
        let mut upstream = response.bytes_stream();
        let mut committed_to_client = false;
        let mut chunks = 0u64;
        while let Some(chunk) = upstream.next().await {
            match chunk {
                Ok(chunk) => {
                    if !chunk.is_empty() {
                        committed_to_client = true;
                    }
                    chunks = chunks.saturating_add(1);
                    yield Ok::<Bytes, std::io::Error>(chunk);
                }
                Err(error) => {
                    tracing::warn!(
                        request_id,
                        selected_key_name = key_name,
                        committed_to_client,
                        chunks,
                        error_class = transport_error_class(&error),
                        "OpenAI stream interrupted; request will not be replayed"
                    );
                    let event = json!({"error":{"type":"upstream_error",
                        "message":"upstream stream was interrupted"}});
                    yield Ok(Bytes::from(format!("data: {event}\n\n")));
                    return;
                }
            }
        }
        tracing::info!(
            request_id,
            selected_key_name = key_name,
            committed_to_client,
            chunks,
            duration_ms = started.elapsed().as_millis(),
            "OpenAI stream closed"
        );
    };
    Body::from_stream(output)
}

fn upstream_failure(
    state: &AppState,
    error: UpstreamError,
    anthropic: bool,
    request_id: &str,
) -> Response {
    match error {
        UpstreamError::Transport(error) => {
            let message = match transport_error_class(&error) {
                "timeout" => "upstream request timed out",
                "connect" => "could not connect to upstream",
                _ => "upstream request failed",
            };
            if anthropic {
                anthropic_error(StatusCode::BAD_GATEWAY, "api_error", message, request_id)
            } else {
                openai_error(StatusCode::BAD_GATEWAY, message, request_id)
            }
        }
        UpstreamError::AllRateLimited { retry_after } => {
            for snapshot in state.upstream.pool().snapshots() {
                tracing::debug!(
                    request_id,
                    runtime_key_index = snapshot.index,
                    configured_key_index = snapshot.configured_index,
                    selected_key_name = %snapshot.name,
                    cooldown_remaining_ms = snapshot
                        .cooldown_remaining
                        .map(|duration| duration.as_millis())
                        .unwrap_or(0),
                    cooldown_until_unix = snapshot
                        .cooldown_until
                        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|duration| duration.as_secs())
                        .unwrap_or(0),
                    last_429_at_unix = snapshot
                        .last_429_at
                        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|duration| duration.as_secs())
                        .unwrap_or(0),
                    last_429_message = snapshot.last_429_message.as_deref().unwrap_or(""),
                    last_429_model = snapshot.last_429_model.as_deref().unwrap_or(""),
                    "Cline key cooldown state"
                );
            }
            let mut response = if anthropic {
                anthropic_error(
                    StatusCode::TOO_MANY_REQUESTS,
                    "rate_limit_error",
                    "all Cline API keys are currently rate-limited",
                    request_id,
                )
            } else {
                openai_error(
                    StatusCode::TOO_MANY_REQUESTS,
                    "all Cline API keys are currently rate-limited",
                    request_id,
                )
            };
            if let Some(duration) = retry_after {
                let seconds = duration
                    .as_secs()
                    .max(u64::from(duration.subsec_nanos() > 0));
                if let Ok(value) = HeaderValue::try_from(seconds.to_string()) {
                    response.headers_mut().insert(header::RETRY_AFTER, value);
                }
            }
            tracing::warn!(
                request_id,
                protocol = if anthropic { "anthropic" } else { "openai" },
                configured_keys = state.upstream.pool().len(),
                "all Cline keys are cooling"
            );
            response
        }
    }
}

fn sanitized_upstream_error(
    state: &AppState,
    error: BufferedUpstreamError,
    anthropic: bool,
    request_id: &str,
) -> Response {
    let secrets = state.upstream.exact_secrets();
    raw_upstream_error(error, anthropic, request_id, &secrets)
}

fn raw_upstream_error(
    error: BufferedUpstreamError,
    anthropic: bool,
    request_id: &str,
    secrets: &[&str],
) -> Response {
    let status = error.classification.outer_status;
    let upstream_headers = error.headers;
    let bytes = error.body;
    let text = String::from_utf8_lossy(&bytes);
    let sanitized = serde_json::from_slice::<Value>(&bytes)
        .map(|value| sanitize_json(value, secrets))
        .unwrap_or_else(|_| {
            json!({"error":{"type":"upstream_error",
                "message":sanitize_text(&text, secrets)}})
        });
    let mut result = if anthropic {
        let message = sanitized
            .get("error")
            .and_then(|error| {
                error
                    .get("message")
                    .or_else(|| error.as_str().map(|_| error))
            })
            .and_then(Value::as_str)
            .or_else(|| sanitized.get("message").and_then(Value::as_str))
            .unwrap_or("upstream request failed");
        let error_type = match status.as_u16() {
            400 | 422 => "invalid_request_error",
            401 => "authentication_error",
            403 => "permission_error",
            404 => "not_found_error",
            413 => "request_too_large",
            429 => "rate_limit_error",
            _ => "api_error",
        };
        anthropic_error(status, error_type, message, request_id)
    } else {
        json_response(status, sanitized, request_id)
    };
    copy_safe_upstream_headers(result.headers_mut(), &upstream_headers);
    result
}

async fn read_response_limited(
    mut response: reqwest::Response,
    limit: usize,
) -> std::result::Result<Bytes, &'static str> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err("upstream response was too large");
    }
    let mut output = Vec::new();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                if output.len().saturating_add(chunk.len()) > limit {
                    return Err("upstream response was too large");
                }
                output.extend_from_slice(&chunk);
            }
            Ok(None) => return Ok(Bytes::from(output)),
            Err(_) => return Err("failed to read upstream response"),
        }
    }
}

fn copy_safe_upstream_headers(target: &mut HeaderMap, source: &HeaderMap) {
    for (name, value) in source {
        let lower = name.as_str();
        if lower == "retry-after"
            || lower == "cache-control"
            || lower.starts_with("x-ratelimit-")
            || lower.starts_with("anthropic-ratelimit-")
        {
            target.insert(name.clone(), value.clone());
        }
    }
}

async fn auth_middleware(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let request_id = request_id(request.headers());
    let anthropic = request.uri().path().starts_with("/v1/messages")
        || request.headers().contains_key("anthropic-version");
    let bearer = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    let x_api_key = request
        .headers()
        .get("x-api-key")
        .and_then(|value| value.to_str().ok());
    if bearer
        .or(x_api_key)
        .is_some_and(|candidate| constant_time_eq(candidate, &state.config.server.api_key))
    {
        next.run(request).await
    } else if anthropic {
        anthropic_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid gateway API key",
            &request_id,
        )
    } else {
        openai_error(
            StatusCode::UNAUTHORIZED,
            "invalid gateway API key",
            &request_id,
        )
    }
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    let mut difference = left.len() ^ right.len();
    let length = left.len().max(right.len());
    for index in 0..length {
        difference |= usize::from(
            left.get(index).copied().unwrap_or_default()
                ^ right.get(index).copied().unwrap_or_default(),
        );
    }
    difference == 0
}

async fn request_id_middleware(mut request: Request<Body>, next: Next) -> Response {
    let id = request
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| {
            !value.is_empty()
                && value.len() <= 128
                && value.chars().all(|character| character.is_ascii_graphic())
        })
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("req_{}", uuid::Uuid::new_v4().simple()));
    if let Ok(value) = HeaderValue::try_from(id.as_str()) {
        request.headers_mut().insert("x-request-id", value);
    }
    let mut response = next.run(request).await;
    insert_request_id(response.headers_mut(), &id);
    response
}

async fn response_log_middleware(request: Request<Body>, next: Next) -> Response {
    let started = Instant::now();
    let request_id = request_id(request.headers());
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let protocol = if path.starts_with("/v1/messages") {
        "anthropic"
    } else {
        "openai"
    };
    let response = next.run(request).await;
    let streaming = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains("text/event-stream"));
    tracing::info!(
        request_id,
        protocol,
        method = %method,
        path,
        status = response.status().as_u16(),
        duration_ms = started.elapsed().as_millis(),
        streaming,
        "client response ready"
    );
    response
}

fn request_body(
    body: std::result::Result<Bytes, BytesRejection>,
    anthropic: bool,
    request_id: &str,
) -> std::result::Result<Bytes, Box<Response>> {
    body.map_err(|_| {
        if anthropic {
            Box::new(anthropic_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request_too_large",
                "request body exceeds the configured limit",
                request_id,
            ))
        } else {
            Box::new(openai_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request body exceeds the configured limit",
                request_id,
            ))
        }
    })
}

fn request_id(headers: &HeaderMap) -> String {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("req_unknown")
        .to_owned()
}

fn insert_request_id(headers: &mut HeaderMap, request_id: &str) {
    if let Ok(value) = HeaderValue::try_from(request_id) {
        headers.insert("x-request-id", value);
    }
}

fn protocol_error(error: ProtocolError, status: StatusCode, request_id: &str) -> Response {
    anthropic_error(status, error.error_type, error.message, request_id)
}

fn anthropic_error(
    status: StatusCode,
    error_type: &str,
    message: impl Into<String>,
    request_id: &str,
) -> Response {
    json_response(
        status,
        anthropic::error_envelope(error_type, message, request_id),
        request_id,
    )
}

fn openai_error(status: StatusCode, message: impl Into<String>, request_id: &str) -> Response {
    json_response(
        status,
        json!({"error":{"type":"proxy_error", "message":message.into()}}),
        request_id,
    )
}

fn json_response(status: StatusCode, value: Value, request_id: &str) -> Response {
    let mut response = (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        value.to_string(),
    )
        .into_response();
    insert_request_id(response.headers_mut(), request_id);
    response
}

pub async fn serve(state: AppState) -> Result<()> {
    let bind = state.config.server.bind.clone();
    let shutdown_timeout = Duration::from_secs(state.config.runtime.shutdown_timeout_secs);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("binding gateway to {bind}"))?;
    tracing::info!(bind, "cline-proxy listening");
    // Debounced runtime-state writer: mutations signal the pool's Notify,
    // this task coalesces them and performs the atomic file replacement off
    // the request path. Healthy requests never touch the disk.
    let writer = state.state_file.as_ref().map(|path| {
        let state = state.clone();
        let debounce = Duration::from_millis(defaults::STATE_DEBOUNCE_MS);
        let path = path.clone();
        tokio::spawn(async move {
            loop {
                state.upstream.pool().dirty().notified().await;
                // Coalesce any notifications that arrived during the wait.
                tokio::time::sleep(debounce).await;
                state.flush_runtime_state();
                tracing::debug!(state_file = %path.display(), "runtime state persisted");
            }
        })
    });
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let app = router(state.clone());
    let mut task = tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(async move {
            let _ = shutdown_rx.await;
        })
        .await
    });
    let shutdown_result = tokio::select! {
        result = &mut task => {
            result.context("gateway task failed")?.context("serving HTTP")?;
            Ok(())
        }
        signal = shutdown_signal() => {
            match &signal {
                Ok(()) => tracing::info!(
                    timeout_secs = shutdown_timeout.as_secs(),
                    "shutdown signal received; draining requests"
                ),
                Err(error) => tracing::warn!(
                    error = %error,
                    timeout_secs = shutdown_timeout.as_secs(),
                    "shutdown signal handler failed; stopping the gateway"
                ),
            }
            let _ = shutdown_tx.send(());
            match tokio::time::timeout(shutdown_timeout, &mut task).await {
                Ok(result) => {
                    result.context("gateway task failed during shutdown")?
                        .context("serving HTTP during shutdown")?;
                }
                Err(_) => {
                    tracing::warn!("graceful shutdown timed out; aborting remaining connections");
                    task.abort();
                    let _ = task.await;
                }
            }
            signal
        }
    };
    // Final flush happens inside the shutdown deadline; it is a small
    // serialized snapshot and cannot meaningfully block exit.
    if let Some(writer) = writer {
        writer.abort();
        let _ = writer.await;
    }
    state.flush_runtime_state();
    shutdown_result
}

async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .context("installing SIGTERM handler")?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result.context("installing SIGINT handler")?,
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c()
        .await
        .context("installing interrupt handler")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use axum::body::to_bytes;
    use axum::extract::State;
    use axum::routing::post;
    use tokio::io::AsyncReadExt;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    use super::*;
    use crate::config::ClineKeyConfig;

    #[derive(Clone)]
    enum MockBody {
        Plain,
        StreamError,
        Stall(Arc<AtomicBool>),
        Delay(Duration),
    }

    #[derive(Clone)]
    struct Spec {
        status: StatusCode,
        body: String,
        content_type: &'static str,
        headers: Vec<(&'static str, &'static str)>,
        kind: MockBody,
    }

    impl Spec {
        fn json(status: u16, body: impl Into<String>) -> Self {
            Self {
                status: StatusCode::from_u16(status).unwrap(),
                body: body.into(),
                content_type: "application/json",
                headers: Vec::new(),
                kind: MockBody::Plain,
            }
        }

        fn sse(body: impl Into<String>) -> Self {
            Self {
                status: StatusCode::OK,
                body: body.into(),
                content_type: "text/event-stream",
                headers: Vec::new(),
                kind: MockBody::Plain,
            }
        }
    }

    #[derive(Clone, Debug)]
    struct SeenRequest {
        authorization: String,
        headers: HeaderMap,
        body: Value,
    }

    #[derive(Clone, Default)]
    struct MockUpstream {
        sequences: Arc<Mutex<HashMap<String, VecDeque<Spec>>>>,
        calls: Arc<Mutex<Vec<SeenRequest>>>,
    }

    impl MockUpstream {
        async fn set(&self, key: &str, specs: Vec<Spec>) {
            self.sequences
                .lock()
                .await
                .insert(key.to_owned(), VecDeque::from(specs));
        }

        async fn seen(&self) -> Vec<SeenRequest> {
            self.calls.lock().await.clone()
        }
    }

    async fn mock_handler(
        State(mock): State<MockUpstream>,
        headers: HeaderMap,
        body: Bytes,
    ) -> Response {
        let authorization = headers
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        let key = authorization
            .strip_prefix("Bearer ")
            .unwrap_or_default()
            .to_owned();
        let parsed = serde_json::from_slice(&body).unwrap_or(Value::Null);
        mock.calls.lock().await.push(SeenRequest {
            authorization,
            headers,
            body: parsed,
        });
        let spec = mock
            .sequences
            .lock()
            .await
            .get_mut(&key)
            .and_then(VecDeque::pop_front)
            .unwrap_or_else(|| Spec::json(500, r#"{"error":{"message":"unscripted"}}"#));
        if let MockBody::Delay(duration) = spec.kind {
            tokio::time::sleep(duration).await;
        }
        let mut builder = Response::builder()
            .status(spec.status)
            .header(header::CONTENT_TYPE, spec.content_type);
        for (name, value) in spec.headers {
            builder = builder.header(name, value);
        }
        let body = match spec.kind {
            MockBody::StreamError => {
                let initial = spec.body;
                let stream = async_stream::stream! {
                    yield Ok::<Bytes, std::io::Error>(Bytes::from(initial));
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    yield Err(std::io::Error::new(std::io::ErrorKind::ConnectionReset, "mock reset"));
                };
                Body::from_stream(stream)
            }
            MockBody::Stall(cancelled) => {
                struct CancellationGuard(Arc<AtomicBool>);
                impl Drop for CancellationGuard {
                    fn drop(&mut self) {
                        self.0.store(true, Ordering::Release);
                    }
                }
                let initial = spec.body;
                let stream = async_stream::stream! {
                    let _guard = CancellationGuard(cancelled);
                    yield Ok::<Bytes, Infallible>(Bytes::from(initial));
                    std::future::pending::<()>().await;
                };
                Body::from_stream(stream)
            }
            MockBody::Plain | MockBody::Delay(_) => Body::from(spec.body),
        };
        builder.body(body).unwrap()
    }

    async fn start_mock() -> (String, MockUpstream, tokio::task::JoinHandle<()>) {
        let mock = MockUpstream::default();
        let app = Router::new()
            .route("/api/v1/chat/completions", post(mock_handler))
            .with_state(mock.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}/api/v1"), mock, task)
    }

    fn test_config(base_url: String, keys: usize) -> Config {
        let mut config = Config::default();
        config.server.api_key = "gateway-secret".into();
        config.upstream.base_url = base_url;
        config.upstream.timeout_secs = 5;
        config.upstream.connect_timeout_secs = 2;
        config.upstream.fallback_429_cooldown_secs = 60;
        config
            .models
            .aliases
            .insert("claude-sonnet-4-6".into(), "z-ai/glm-5.3-flash".into());
        config.cline_api_keys = (0..keys)
            .map(|index| ClineKeyConfig {
                name: format!("cline-{}", index + 1),
                api_key: format!("cline-key-{}", index + 1),
                enabled: true,
            })
            .collect();
        // Tests opt in to persistence explicitly so the default working
        // directory is never polluted by runtime-state.json.
        config.runtime.state_file = None;
        config
    }

    fn gateway_request(path: &str, body: impl Into<Body>) -> Request<Body> {
        Request::builder()
            .method(if path == "/v1/models" { "GET" } else { "POST" })
            .uri(path)
            .header(header::AUTHORIZATION, "Bearer gateway-secret")
            .header(header::CONTENT_TYPE, "application/json")
            .body(body.into())
            .unwrap()
    }

    fn chat_body(stream: bool) -> String {
        json!({
            "model":"claude-sonnet-4-6",
            "messages":[{"role":"user","content":"hello"}],
            "stream":stream,
            "future_field":{"preserve":true}
        })
        .to_string()
    }

    fn anthropic_body(stream: bool) -> String {
        json!({
            "model":"claude-sonnet-4-6",
            "max_tokens":128,
            "stream":stream,
            "messages":[{"role":"user","content":"hello"}]
        })
        .to_string()
    }

    async fn response_text(response: Response) -> String {
        String::from_utf8(
            to_bytes(response.into_body(), 2 * 1024 * 1024)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap()
    }

    fn successful_json(text: &str) -> String {
        json!({
            "id":"chat_1","model":"z-ai/glm-5.3-flash",
            "choices":[{"message":{"role":"assistant","content":text},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":5,"completion_tokens":2}
        })
        .to_string()
    }

    fn successful_sse(text: &str) -> String {
        format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            json!({"id":"chat_1","model":"z-ai/glm-5.3-flash",
                "choices":[{"delta":{"content":text},"finish_reason":null}]}),
            json!({"choices":[{"delta":{},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":5,"completion_tokens":2}})
        )
    }

    #[tokio::test]
    async fn openai_normal_requests_are_sticky_alias_and_headers_are_safe() {
        let (base, mock, task) = start_mock().await;
        mock.set(
            "cline-key-1",
            vec![
                Spec::json(200, successful_json("one")),
                Spec::json(200, successful_json("two")),
            ],
        )
        .await;
        let app = router(AppState::new(test_config(base, 2)).unwrap());
        for _ in 0..2 {
            let mut request = gateway_request("/v1/chat/completions", chat_body(false));
            request.headers_mut().insert(
                "x-client-type",
                HeaderValue::from_static("malicious-caller"),
            );
            request
                .headers_mut()
                .insert("x-api-key", HeaderValue::from_static("gateway-secret"));
            let response = app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        let seen = mock.seen().await;
        assert_eq!(seen.len(), 2);
        assert!(seen
            .iter()
            .all(|request| request.authorization == "Bearer cline-key-1"));
        assert!(seen
            .iter()
            .all(|request| request.body["model"] == "z-ai/glm-5.3-flash"));
        assert!(seen
            .iter()
            .all(|request| request.body["future_field"]["preserve"] == true));
        assert_eq!(seen[0].headers["x-client-type"], "cline-vscode");
        assert_ne!(
            seen[0].headers[header::AUTHORIZATION],
            "Bearer gateway-secret"
        );
        assert!(!seen[0].headers.contains_key("x-api-key"));
        task.abort();
    }

    #[tokio::test]
    async fn http_429_walks_keys_once_and_successful_switch_stays_sticky() {
        let (base, mock, task) = start_mock().await;
        let mut limited = Spec::json(
            429,
            r#"{"model":"cline-key-1","error":{"message":"gateway-secret cline-key-1: Try again in 2h 30m"}}"#,
        );
        limited.headers.push(("retry-after", "7"));
        mock.set("cline-key-1", vec![limited]).await;
        mock.set(
            "cline-key-2",
            vec![
                Spec::json(200, successful_json("after failover")),
                Spec::json(200, successful_json("sticky")),
            ],
        )
        .await;
        let state = AppState::new(test_config(base, 3)).unwrap();
        let app = router(state.clone());
        for _ in 0..2 {
            let response = app
                .clone()
                .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        let authorizations = mock
            .seen()
            .await
            .into_iter()
            .map(|request| request.authorization)
            .collect::<Vec<_>>();
        assert_eq!(
            authorizations,
            [
                "Bearer cline-key-1",
                "Bearer cline-key-2",
                "Bearer cline-key-2"
            ]
        );
        let snapshot = &state.upstream.pool().snapshots()[0];
        assert!(snapshot.cooldown_remaining.is_some());
        let message = snapshot.last_429_message.as_deref().unwrap_or_default();
        assert!(!message.contains("gateway-secret"));
        assert!(!message.contains("cline-key-1"));
        assert_eq!(snapshot.last_429_model.as_deref(), Some("[REDACTED]"));
        task.abort();
    }

    #[tokio::test]
    async fn proxy_wrapped_429_parses_cooldown_switches_once_and_stays_sticky() {
        let (base, mock, task) = start_mock().await;
        mock.set(
            "cline-key-1",
            vec![Spec::json(
                502,
                "upstream returned 429: Error 429: Daily free limit reached on model \
                 z-ai/glm-5.3-flash. Try again in 23h 17m",
            )],
        )
        .await;
        mock.set(
            "cline-key-2",
            vec![Spec::json(200, successful_json("after wrapped failover"))],
        )
        .await;
        let state = AppState::new(test_config(base, 3)).unwrap();
        let app = router(state.clone());
        let response = app
            .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let seen = mock.seen().await;
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].authorization, "Bearer cline-key-1");
        assert_eq!(seen[1].authorization, "Bearer cline-key-2");

        let snapshots = state.upstream.pool().snapshots();
        let remaining = snapshots[0].cooldown_remaining.unwrap();
        assert!(remaining <= Duration::from_secs(83_820));
        assert!(remaining > Duration::from_secs(83_700));
        assert_eq!(
            snapshots[0].last_429_model.as_deref(),
            Some("z-ai/glm-5.3-flash")
        );
        assert_eq!(
            state.upstream.pool().select(&HashSet::new()).unwrap().index,
            1
        );
        task.abort();
    }

    #[tokio::test]
    async fn structured_proxy_upstream_status_429_switches_keys() {
        let (base, mock, task) = start_mock().await;
        mock.set(
            "cline-key-1",
            vec![Spec::json(
                502,
                json!({"error":{
                    "upstream_status":429,
                    "message":"Daily free limit reached on model z-ai/glm-5.3-flash. Try again in 23h 17m"
                }})
                .to_string(),
            )],
        )
        .await;
        mock.set(
            "cline-key-2",
            vec![Spec::json(200, successful_json("structured failover"))],
        )
        .await;
        let app = router(AppState::new(test_config(base, 2)).unwrap());
        let response = app
            .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let seen = mock.seen().await;
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].authorization, "Bearer cline-key-1");
        assert_eq!(seen[1].authorization, "Bearer cline-key-2");
        task.abort();
    }

    #[tokio::test]
    async fn ambiguous_429_text_and_model_output_never_switch_keys() {
        let (base, mock, task) = start_mock().await;
        mock.set(
            "cline-key-1",
            vec![
                Spec::json(502, "bad gateway"),
                Spec::json(502, "proxy request id 429123 failed"),
                Spec::json(500, "error 429"),
                Spec::json(200, successful_json("HTTP status 429 means rate limited")),
            ],
        )
        .await;
        let state = AppState::new(test_config(base, 2)).unwrap();
        let app = router(state.clone());
        for (expected_status, expected_text) in [
            (502, "bad gateway"),
            (502, "request id 429123"),
            (500, "error 429"),
            (200, "status 429"),
        ] {
            let response = app
                .clone()
                .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), expected_status);
            assert!(response_text(response).await.contains(expected_text));
        }
        let seen = mock.seen().await;
        assert_eq!(seen.len(), 4);
        assert!(seen
            .iter()
            .all(|request| request.authorization == "Bearer cline-key-1"));
        assert_eq!(
            state.upstream.pool().select(&HashSet::new()).unwrap().index,
            0
        );
        task.abort();
    }

    #[tokio::test]
    async fn proxy_wrapped_429_walks_each_key_once_then_stays_on_third() {
        let (base, mock, task) = start_mock().await;
        mock.set(
            "cline-key-1",
            vec![Spec::json(
                502,
                "upstream returned 429: Daily free limit reached. Try again in 23h 17m",
            )],
        )
        .await;
        mock.set(
            "cline-key-2",
            vec![Spec::json(
                503,
                "upstream returned HTTP 429: rate limited. Try again in 4h",
            )],
        )
        .await;
        mock.set(
            "cline-key-3",
            vec![Spec::json(200, successful_json("third key"))],
        )
        .await;
        let state = AppState::new(test_config(base, 3)).unwrap();
        let app = router(state.clone());
        let response = app
            .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let authorizations = mock
            .seen()
            .await
            .into_iter()
            .map(|request| request.authorization)
            .collect::<Vec<_>>();
        assert_eq!(
            authorizations,
            [
                "Bearer cline-key-1",
                "Bearer cline-key-2",
                "Bearer cline-key-3"
            ]
        );
        let snapshots = state.upstream.pool().snapshots();
        assert!(snapshots[0].cooldown_remaining.is_some());
        assert!(snapshots[1].cooldown_remaining.is_some());
        assert_eq!(
            state.upstream.pool().select(&HashSet::new()).unwrap().index,
            2
        );
        task.abort();
    }

    #[tokio::test]
    async fn all_proxy_wrapped_429_is_bounded_and_returns_semantic_429() {
        let (base, mock, task) = start_mock().await;
        for index in 1..=3 {
            mock.set(
                &format!("cline-key-{index}"),
                vec![Spec::json(
                    502,
                    format!(
                        "upstream response status: 429; quota exceeded; Try again in {}h",
                        index + 1
                    ),
                )],
            )
            .await;
        }
        let app = router(AppState::new(test_config(base, 3)).unwrap());
        let response = app
            .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(response.headers().contains_key(header::RETRY_AFTER));
        let body = response_text(response).await;
        assert!(body.contains("all Cline API keys are currently rate-limited"));
        assert_eq!(mock.seen().await.len(), 3);
        task.abort();
    }

    #[tokio::test]
    async fn repeated_429_is_bounded_by_configured_key_count() {
        let (base, mock, task) = start_mock().await;
        mock.set(
            "cline-key-1",
            vec![Spec::json(429, r#"{"retry_after_seconds":10}"#)],
        )
        .await;
        mock.set(
            "cline-key-2",
            vec![Spec::json(429, r#"{"message":"Retry in 3h"}"#)],
        )
        .await;
        mock.set(
            "cline-key-3",
            vec![Spec::json(200, successful_json("third"))],
        )
        .await;
        let app = router(AppState::new(test_config(base, 3)).unwrap());
        let response = app
            .clone()
            .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(mock.seen().await.len(), 3);
        task.abort();
    }

    #[tokio::test]
    async fn all_429_returns_useful_bounded_error() {
        let (base, mock, task) = start_mock().await;
        for index in 1..=3 {
            mock.set(
                &format!("cline-key-{index}"),
                vec![Spec::json(429, "not useful")],
            )
            .await;
        }
        let app = router(AppState::new(test_config(base, 3)).unwrap());
        let response = app
            .clone()
            .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(response.headers().contains_key(header::RETRY_AFTER));
        let body = response_text(response).await;
        assert!(body.contains("all Cline API keys are currently rate-limited"));
        assert_eq!(mock.seen().await.len(), 3);
        task.abort();
    }

    #[tokio::test]
    async fn every_non_429_status_returns_without_calling_key_two() {
        let (base, mock, task) = start_mock().await;
        let statuses = [
            400u16, 401, 402, 403, 404, 408, 409, 422, 500, 502, 503, 504,
        ];
        mock.set(
            "cline-key-1",
            statuses
                .iter()
                .map(|status| {
                    Spec::json(
                        *status,
                        format!(r#"{{"error":{{"message":"status {status}"}}}}"#),
                    )
                })
                .collect(),
        )
        .await;
        let app = router(AppState::new(test_config(base, 2)).unwrap());
        for status in statuses {
            let response = app
                .clone()
                .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), status);
        }
        let seen = mock.seen().await;
        assert_eq!(seen.len(), statuses.len());
        assert!(seen
            .iter()
            .all(|request| request.authorization == "Bearer cline-key-1"));
        task.abort();
    }

    #[tokio::test]
    async fn malformed_success_body_returns_502_without_failover() {
        let (base, mock, task) = start_mock().await;
        mock.set("cline-key-1", vec![Spec::json(200, "not-json")])
            .await;
        let app = router(AppState::new(test_config(base, 2)).unwrap());
        let response = app
            .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(mock.seen().await.len(), 1);
        task.abort();
    }

    #[tokio::test]
    async fn stream_429_before_commit_can_fail_over() {
        let (base, mock, task) = start_mock().await;
        mock.set("cline-key-1", vec![Spec::json(429, "Retry after 30m")])
            .await;
        mock.set("cline-key-2", vec![Spec::sse(successful_sse("streamed"))])
            .await;
        let app = router(AppState::new(test_config(base, 2)).unwrap());
        let response = app
            .oneshot(gateway_request("/v1/chat/completions", chat_body(true)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("streamed"));
        assert_eq!(mock.seen().await.len(), 2);
        task.abort();
    }

    #[tokio::test]
    async fn openai_midstream_error_never_replays_request() {
        let (base, mock, task) = start_mock().await;
        let mut spec = Spec::sse(format!(
            "data: {}\n\n",
            json!({"choices":[{"delta":{"content":"partial"}}]})
        ));
        spec.kind = MockBody::StreamError;
        mock.set("cline-key-1", vec![spec]).await;
        mock.set("cline-key-2", vec![Spec::sse(successful_sse("duplicate"))])
            .await;
        let app = router(AppState::new(test_config(base, 2)).unwrap());
        let response = app
            .oneshot(gateway_request("/v1/chat/completions", chat_body(true)))
            .await
            .unwrap();
        let body = response_text(response).await;
        assert!(body.contains("partial"));
        assert!(body.contains("upstream stream was interrupted"));
        assert!(!body.contains("duplicate"));
        assert_eq!(mock.seen().await.len(), 1);
        task.abort();
    }

    #[tokio::test]
    async fn anthropic_truncated_sse_never_replays_request() {
        let (base, mock, task) = start_mock().await;
        mock.set(
            "cline-key-1",
            vec![Spec::sse(format!(
                "data: {}\n\n",
                json!({"id":"chat","choices":[{"delta":{"content":"partial"}}]})
            ))],
        )
        .await;
        mock.set("cline-key-2", vec![Spec::sse(successful_sse("duplicate"))])
            .await;
        let app = router(AppState::new(test_config(base, 2)).unwrap());
        let response = app
            .oneshot(gateway_request("/v1/messages", anthropic_body(true)))
            .await
            .unwrap();
        let body = response_text(response).await;
        assert!(body.contains("text_delta"));
        assert!(body.contains("upstream stream ended unexpectedly"));
        assert!(!body.contains("duplicate"));
        assert_eq!(mock.seen().await.len(), 1);
        task.abort();
    }

    #[tokio::test]
    async fn anthropic_malformed_sse_never_replays_request() {
        let (base, mock, task) = start_mock().await;
        mock.set("cline-key-1", vec![Spec::sse("data: this-is-not-json\n\n")])
            .await;
        mock.set("cline-key-2", vec![Spec::sse(successful_sse("duplicate"))])
            .await;
        let app = router(AppState::new(test_config(base, 2)).unwrap());
        let response = app
            .oneshot(gateway_request("/v1/messages", anthropic_body(true)))
            .await
            .unwrap();
        let body = response_text(response).await;
        assert!(body.contains("upstream sent invalid SSE JSON"));
        assert!(!body.contains("duplicate"));
        assert_eq!(mock.seen().await.len(), 1);
        task.abort();
    }

    #[tokio::test]
    async fn client_cancellation_drops_upstream_and_never_replays() {
        let (base, mock, task) = start_mock().await;
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut spec = Spec::sse(format!(
            "data: {}\n\n",
            json!({"choices":[{"delta":{"content":"first"}}]})
        ));
        spec.kind = MockBody::Stall(cancelled.clone());
        mock.set("cline-key-1", vec![spec]).await;
        mock.set("cline-key-2", vec![Spec::sse(successful_sse("duplicate"))])
            .await;
        let app = router(AppState::new(test_config(base, 2)).unwrap());
        let response = app
            .oneshot(gateway_request("/v1/chat/completions", chat_body(true)))
            .await
            .unwrap();
        let mut stream = response.into_body().into_data_stream();
        assert!(stream.next().await.is_some());
        drop(stream);
        for _ in 0..40 {
            if cancelled.load(Ordering::Acquire) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(cancelled.load(Ordering::Acquire));
        assert_eq!(mock.seen().await.len(), 1);
        task.abort();
    }

    #[tokio::test]
    async fn anthropic_nonstream_converts_tool_loop_reasoning_and_usage() {
        let (base, mock, task) = start_mock().await;
        mock.set(
            "cline-key-1",
            vec![Spec::json(
                200,
                json!({
                    "id":"chat","model":"z-ai/glm-5.3-flash",
                    "choices":[{"message":{
                        "reasoning_content":"consider", "content":"calling",
                        "tool_calls":[{"id":"toolu_next","function":{
                            "name":"Read","arguments":"{\"file_path\":\"/tmp/a\"}"}}]
                    },"finish_reason":"tool_calls"}],
                    "usage":{"prompt_tokens":12,"completion_tokens":4}
                })
                .to_string(),
            )],
        )
        .await;
        let app = router(AppState::new(test_config(base, 1)).unwrap());
        let body = json!({
            "model":"claude-sonnet-4-6","max_tokens":128,
            "messages":[
                {"role":"assistant","content":[{"type":"tool_use","id":"toolu_old","name":"Read","input":{"file_path":"/tmp/old"}}]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_old","content":"old contents"}]}
            ],
            "tools":[{"name":"Read","input_schema":{"type":"object"}}]
        })
        .to_string();
        let response = app
            .oneshot(gateway_request("/v1/messages", body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let value: Value = serde_json::from_str(&response_text(response).await).unwrap();
        // The request carries no `thinking`, so upstream reasoning is not
        // exposed: the response starts with the text block, not thinking.
        assert_eq!(value["content"][0]["type"], "text");
        assert_eq!(value["content"][0]["text"], "calling");
        assert_eq!(value["content"][1]["type"], "tool_use");
        assert_eq!(value["content"][1]["input"]["file_path"], "/tmp/a");
        assert_eq!(value["stop_reason"], "tool_use");
        assert_eq!(value["usage"]["output_tokens"], 4);
        let seen = mock.seen().await;
        assert_eq!(seen[0].body["messages"][1]["role"], "tool");
        assert_eq!(seen[0].body["model"], "z-ai/glm-5.3-flash");
        // GLM policy on the wire: explicit effort + capped output.
        assert_eq!(seen[0].body["reasoning_effort"], "high");
        assert_eq!(seen[0].body["max_tokens"], 128);
        task.abort();
    }

    #[tokio::test]
    async fn anthropic_stream_has_valid_fragmented_parallel_tool_lifecycle() {
        let (base, mock, task) = start_mock().await;
        let events = [
            json!({"id":"chat","model":"z-ai/glm-5.3-flash","choices":[{"delta":{"reasoning":"think"}}]}),
            json!({"choices":[{"delta":{"content":"hi","tool_calls":[
                {"index":0,"id":"call_a","function":{"name":"Read","arguments":"{\"path\":"}},
                {"index":1,"id":"call_b","function":{"name":"Write","arguments":"{\"text\":\"é"}}
            ]}}]}),
            json!({"choices":[{"delta":{"tool_calls":[
                {"index":1,"function":{"arguments":"\"}" }},
                {"index":0,"function":{"arguments":"\"/tmp/a\"}"}}
            ]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":8,"completion_tokens":5}}),
        ];
        let sse = format!(
            "data: {}\n\ndata: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            events[0], events[1], events[2]
        );
        mock.set("cline-key-1", vec![Spec::sse(sse)]).await;
        let app = router(AppState::new(test_config(base, 1)).unwrap());
        let response = app
            .oneshot(gateway_request("/v1/messages", anthropic_body(true)))
            .await
            .unwrap();
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let body = response_text(response).await;
        assert!(body.starts_with("event: message_start"));
        assert_eq!(body.matches("\"type\":\"tool_use\"").count(), 2);
        assert!(body.contains("input_json_delta"));
        // No `thinking` in the request: reasoning is suppressed, and no
        // thinking block (hence no signature) is ever emitted.
        assert!(!body.contains("thinking_delta"));
        assert!(!body.contains("signature_delta"));
        assert!(body.contains("\"stop_reason\":\"tool_use\""));
        assert!(body.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
        task.abort();
    }

    #[tokio::test]
    async fn anthropic_stream_exposes_thinking_only_when_requested() {
        let (base, mock, task) = start_mock().await;
        let sse = format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            json!({"id":"chat","model":"z-ai/glm-5.3-flash","choices":[{"delta":{"reasoning":"visible think"}}]}),
            json!({"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":3,"completion_tokens":2,
                    "completion_tokens_details":{"reasoning_tokens":1}}})
        );
        mock.set("cline-key-1", vec![Spec::sse(sse.clone())]).await;
        let app = router(AppState::new(test_config(base, 1)).unwrap());
        let mut request = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(header::AUTHORIZATION, "Bearer gateway-secret")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model":"claude-sonnet-4-6","max_tokens":128,
                    "stream":true,
                    "thinking":{"type":"adaptive"},
                    "messages":[{"role":"user","content":"hello"}]
                })
                .to_string(),
            ))
            .unwrap();
        *request.uri_mut() = "/v1/messages".parse().unwrap();
        let response = app.oneshot(request).await.unwrap();
        let body = response_text(response).await;
        assert!(body.contains("thinking_delta"));
        assert!(body.contains("visible think"));
        assert!(body.contains("signature_delta"));
        // And when thinking is NOT requested, the same reasoning is suppressed.
        let (base2, mock2, task2) = start_mock().await;
        mock2.set("cline-key-1", vec![Spec::sse(sse.clone())]).await;
        let app2 = router(AppState::new(test_config(base2, 1)).unwrap());
        let response2 = app2
            .oneshot(gateway_request("/v1/messages", anthropic_body(true)))
            .await
            .unwrap();
        let body2 = response_text(response2).await;
        assert!(!body2.contains("visible think"));
        assert!(!body2.contains("thinking_delta"));
        task.abort();
        task2.abort();
    }

    #[tokio::test]
    async fn local_models_count_tokens_health_and_auth_work_without_upstream() {
        let (base, mock, task) = start_mock().await;
        let app = router(AppState::new(test_config(base, 1)).unwrap());
        for path in ["/healthz", "/readyz"] {
            let response = app
                .clone()
                .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        let unauthorized = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
        let models = app
            .clone()
            .oneshot(gateway_request("/v1/models", Body::empty()))
            .await
            .unwrap();
        let model_text = response_text(models).await;
        assert!(model_text.contains("claude-sonnet-4-6"));
        assert!(model_text.contains("z-ai/glm-5.3-flash"));
        let count = app
            .clone()
            .oneshot(gateway_request(
                "/v1/messages/count_tokens",
                json!({"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"hello"}]})
                    .to_string(),
            ))
            .await
            .unwrap();
        assert_eq!(
            count.headers()["x-cline-proxy-token-count"],
            "exact_glm53_optimized"
        );
        let count_value: Value = serde_json::from_str(&response_text(count).await).unwrap();
        assert!(count_value["input_tokens"].as_u64().unwrap() > 0);
        assert!(mock.seen().await.is_empty());
        task.abort();
    }

    #[tokio::test]
    async fn upstream_error_echoes_are_recursively_redacted() {
        let (base, mock, task) = start_mock().await;
        mock.set(
            "cline-key-1",
            vec![Spec::json(
                400,
                json!({"error":{"message":"gateway-secret cline-key-1 cline-key-2",
                    "authorization":"Bearer cline-key-1"}})
                .to_string(),
            )],
        )
        .await;
        let app = router(AppState::new(test_config(base, 2)).unwrap());
        let response = app
            .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
            .await
            .unwrap();
        let body = response_text(response).await;
        assert!(!body.contains("gateway-secret"));
        assert!(!body.contains("cline-key-1"));
        assert!(!body.contains("cline-key-2"));
        assert!(body.contains("REDACTED"));
        assert_eq!(mock.seen().await.len(), 1);
        task.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_normal_requests_remain_sticky() {
        let (base, mock, task) = start_mock().await;
        mock.set(
            "cline-key-1",
            (0..24)
                .map(|_| Spec::json(200, successful_json("ok")))
                .collect(),
        )
        .await;
        let app = router(AppState::new(test_config(base, 3)).unwrap());
        let responses = futures_util::future::join_all((0..24).map(|_| {
            app.clone()
                .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
        }))
        .await;
        assert!(responses
            .into_iter()
            .all(|response| response.unwrap().status() == StatusCode::OK));
        let seen = mock.seen().await;
        assert_eq!(seen.len(), 24);
        assert!(seen
            .iter()
            .all(|request| request.authorization == "Bearer cline-key-1"));
        task.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_429_transitions_do_not_corrupt_active_index() {
        let (base, mock, task) = start_mock().await;
        mock.set(
            "cline-key-1",
            (0..16)
                .map(|_| Spec::json(429, "Retry after 30s"))
                .collect(),
        )
        .await;
        mock.set(
            "cline-key-2",
            (0..32)
                .map(|_| Spec::json(200, successful_json("ok")))
                .collect(),
        )
        .await;
        let state = AppState::new(test_config(base, 3)).unwrap();
        let app = router(state.clone());
        let responses = futures_util::future::join_all((0..16).map(|_| {
            app.clone()
                .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
        }))
        .await;
        assert!(responses
            .into_iter()
            .all(|response| response.unwrap().status() == StatusCode::OK));
        let seen = mock.seen().await;
        assert!(seen
            .iter()
            .all(|request| request.authorization != "Bearer cline-key-3"));
        assert_eq!(
            state.upstream.pool().select(&HashSet::new()).unwrap().index,
            1
        );
        task.abort();
    }

    #[tokio::test]
    async fn timeout_does_not_switch_key() {
        let (base, mock, task) = start_mock().await;
        let mut spec = Spec::json(200, successful_json("late"));
        spec.kind = MockBody::Delay(Duration::from_secs(2));
        mock.set("cline-key-1", vec![spec]).await;
        let mut config = test_config(base, 2);
        config.upstream.timeout_secs = 1;
        let state = AppState::new(config).unwrap();
        let app = router(state.clone());
        let response = app
            .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(mock.seen().await.len(), 1);
        assert_eq!(
            state.upstream.pool().select(&HashSet::new()).unwrap().index,
            0
        );
        task.abort();
    }

    #[tokio::test]
    async fn connection_refused_does_not_switch_key() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let state = AppState::new(test_config(format!("http://{address}/api/v1"), 2)).unwrap();
        let app = router(state.clone());
        let response = app
            .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            state.upstream.pool().select(&HashSet::new()).unwrap().index,
            0
        );
    }

    #[tokio::test]
    async fn connection_reset_does_not_switch_key() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            if let Ok((mut stream, _)) = listener.accept().await {
                let mut buffer = [0u8; 128];
                let _ = stream.read(&mut buffer).await;
                // Dropping the socket before an HTTP response simulates a reset/EOF.
            }
        });
        let state = AppState::new(test_config(format!("http://{address}/api/v1"), 2)).unwrap();
        let app = router(state.clone());
        let response = app
            .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            state.upstream.pool().select(&HashSet::new()).unwrap().index,
            0
        );
        task.await.unwrap();
    }

    fn unique_state_path(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "cline-proxy-server-test-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("runtime-state.json")
    }

    /// Production regression: key1–key5 confirm daily-quota 429s, key6
    /// succeeds, state persists, and after a full process restart the next
    /// logical request must reach key6 on the FIRST attempt with zero failed
    /// probes.
    #[tokio::test]
    async fn restart_with_persisted_state_skips_known_cooling_keys() {
        let (base, mock, task) = start_mock().await;
        let cooldowns = ["8h 37m", "9h 42m", "10h 13m", "10h 55m", "11h 53m"];
        for (index, cooldown) in cooldowns.iter().enumerate() {
            mock.set(
                &format!("cline-key-{}", index + 1),
                vec![Spec::json(
                    429,
                    json!({"error":{"message":format!(
                        "Daily free limit reached on model z-ai/glm-5.3-flash. Try again in {cooldown}"
                    )}})
                    .to_string(),
                )],
            )
            .await;
        }
        mock.set(
            "cline-key-6",
            vec![
                Spec::json(200, successful_json("healthy key")),
                Spec::json(200, successful_json("healthy key after restart")),
            ],
        )
        .await;

        let state_path = unique_state_path("restart");
        let mut config = test_config(base, 6);
        config.runtime.state_file = Some(state_path.to_string_lossy().into_owned());

        // First logical request: serial discovery of five exhausted keys.
        let state = AppState::new(config.clone()).unwrap();
        let app = router(state.clone());
        let response = app
            .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(mock.seen().await.len(), 6);
        let snapshots = state.upstream.pool().snapshots();
        assert_eq!(snapshots[5].name.as_ref(), "cline-6");
        for snapshot in &snapshots[..5] {
            assert_eq!(snapshot.phase, crate::pool::KeyPhase::Cooling);
            assert_eq!(
                snapshot.rate_limit_kind,
                Some(crate::rate_limit::RateLimitKind::DailyQuota)
            );
        }

        // Persist (the writer task debounces; tests flush synchronously) and
        // simulate a full process restart with a fresh AppState.
        state.flush_runtime_state();
        let state_file_contents = std::fs::read_to_string(&state_path).unwrap();
        assert!(!state_file_contents.contains("cline-key-"));
        assert!(!state_file_contents.contains("gateway-secret"));
        let restarted = AppState::new(config).unwrap();
        assert_eq!(
            restarted
                .upstream
                .pool()
                .active_key_name()
                .map(|name| name.to_string())
                .as_deref(),
            Some("cline-6")
        );
        let app = router(restarted);
        let response = app
            .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let seen = mock.seen().await;
        assert_eq!(seen.len(), 7);
        assert_eq!(seen[6].authorization, "Bearer cline-key-6");
        std::fs::remove_dir_all(state_path.parent().unwrap()).ok();
        task.abort();
    }

    /// Prefix stability end-to-end (issue #8): two Claude Code-shaped
    /// requests that differ ONLY in the dynamic billing-header metadata and
    /// in tool-argument key insertion order must produce byte-identical
    /// upstream system prefixes. The billing header must be gone from the
    /// wire, and the historical tool arguments canonicalized.
    #[tokio::test]
    async fn dynamic_billing_header_and_argument_order_produce_identical_prefix() {
        let (base, mock, task) = start_mock().await;
        mock.set(
            "cline-key-1",
            vec![
                Spec::json(200, successful_json("one")),
                Spec::json(200, successful_json("two")),
            ],
        )
        .await;
        let app = router(AppState::new(test_config(base, 1)).unwrap());
        let history = json!([
            {"role":"user","content":[{"type":"text","text":"fix the bug"}]},
            {"role":"assistant","content":[
                {"type":"thinking","thinking":"old reasoning","signature":"s"},
                {"type":"tool_use","id":"toolu_1","name":"Edit",
                 "input":{"path":"src/a.rs","line":12}}
            ]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"toolu_1","content":"done"}
            ]}
        ]);
        let make_request = |header_value: &str, argument_order: bool| {
            let tool_use_input = if argument_order {
                json!({"path":"src/a.rs","line":12})
            } else {
                json!({"line":12,"path":"src/a.rs"})
            };
            let mut history = history.clone();
            history[1]["content"][1]["input"] = tool_use_input;
            json!({
                "model":"claude-sonnet-4-6",
                "max_tokens":1_000,
                "system":[
                    {"type":"text","text":format!(
                        "x-anthropic-billing-header: {{\"cch\":\"{header_value}\"}}\nYou are Claude Code.")},
                    {"type":"text","text":"Be careful."}
                ],
                "messages":history,
                "tools":[{"name":"Edit","input_schema":{"type":"object"}}]
            })
            .to_string()
        };
        let request_a = make_request("AAA", true);
        let request_b = make_request("BBB", false);
        for body in [request_a, request_b] {
            let response = app
                .clone()
                .oneshot(gateway_request("/v1/messages", body))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        let seen = mock.seen().await;
        assert_eq!(seen.len(), 2);
        for request in &seen {
            let system = &request.body["messages"][0];
            assert_eq!(system["role"], "system");
            let text = system["content"].as_array().unwrap();
            // Billing header stripped from the leading text block only.
            assert!(!text[0]["text"]
                .as_str()
                .unwrap()
                .starts_with("x-anthropic-billing-header"));
            assert!(text[0]["text"]
                .as_str()
                .unwrap()
                .starts_with("You are Claude Code."));
            assert_eq!(text[1]["text"], "Be careful.");
            // Historical tool arguments canonicalized.
            assert_eq!(
                request.body["messages"][2]["tool_calls"][0]["function"]["arguments"],
                "{\"line\":12,\"path\":\"src/a.rs\"}"
            );
        }
        task.abort();
    }

    /// The exact tokenizer is CPU-bound and runs on the blocking pool under
    /// a semaphore. Saturation test: many concurrent large requests must all
    /// complete, the semaphore must hold (never more permits in flight than
    /// configured), and the async runtime must stay responsive throughout
    /// (SSE mock polling keeps progressing while counts run).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn token_telemetry_saturation_keeps_runtime_responsive_and_bounded() {
        let (base, mock, task) = start_mock().await;
        mock.set(
            "cline-key-1",
            (0..10)
                .map(|_| Spec::json(200, successful_json("ok")))
                .collect(),
        )
        .await;
        let mut config = test_config(base, 1);
        config.glm53.telemetry.exact_input_tokens = true;
        config.glm53.telemetry.max_concurrent_token_counts = 1;
        let state = AppState::new(config).unwrap();
        let app = router(state.clone());
        // ~70 KB bodies with thinking history so each count does real
        // tokenization work.
        let history_text = "analysis step: inspect the tokenizer pipeline and the ".repeat(300);
        let bodies = (0..10)
            .map(|index| {
                json!({
                    "model":"claude-sonnet-4-6",
                    "max_tokens":128,
                    "messages":[
                        {"role":"user","content":[{"type":"text","text":format!("task {index}")}]},
                        {"role":"assistant","content":[
                            {"type":"thinking","thinking":history_text,"signature":"s"},
                            {"type":"text","text":"reading"}
                        ]},
                        {"role":"user","content":[{"type":"text","text":"go on"}]}
                    ]
                })
                .to_string()
            })
            .collect::<Vec<_>>();
        // Drive requests and a concurrent "SSE polling" ticker together. The
        // assertion is not tick count (CPU contention with the tokenizer is
        // expected and legitimate) but liveness: no single tick gap may
        // stall, which is what an occupied Tokio worker would cause.
        let ticker = tokio::spawn(async move {
            let mut max_gap_ms: u128 = 0;
            let started = Instant::now();
            let mut last = started;
            while started.elapsed() < Duration::from_secs(15) {
                tokio::time::sleep(Duration::from_millis(5)).await;
                let now = Instant::now();
                max_gap_ms = max_gap_ms.max(now.duration_since(last).as_millis());
                last = now;
            }
            max_gap_ms
        });
        let responses = futures_util::future::join_all(bodies.into_iter().map(|body| {
            let app = app.clone();
            async move {
                app.oneshot(gateway_request("/v1/messages", body))
                    .await
                    .unwrap()
                    .status()
            }
        }))
        .await;
        assert!(responses.iter().all(|status| *status == StatusCode::OK));
        let max_gap_ms = ticker.await.unwrap();
        // A 5 ms sleep waking with a multi-hundred-ms gap means an async
        // worker was blocked by CPU-bound work. spawn_blocking keeps the
        // workers free; allow generous CI variance.
        assert!(
            max_gap_ms < 500,
            "async runtime stalled under tokenizer load: max tick gap {max_gap_ms}ms"
        );
        task.abort();
    }

    #[tokio::test]
    async fn corrupt_runtime_state_starts_safely_and_serves() {
        let (base, mock, task) = start_mock().await;
        mock.set("cline-key-1", vec![Spec::json(200, successful_json("ok"))])
            .await;
        let state_path = unique_state_path("corrupt");
        std::fs::write(&state_path, b"{ definitively not json").unwrap();
        let mut config = test_config(base, 2);
        config.runtime.state_file = Some(state_path.to_string_lossy().into_owned());
        let state = AppState::new(config).unwrap();
        let app = router(state);
        let response = app
            .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(mock.seen().await.len(), 1);
        std::fs::remove_dir_all(state_path.parent().unwrap()).ok();
        task.abort();
    }

    #[tokio::test]
    async fn admin_status_requires_auth_and_leaks_no_secrets() {
        let (base, mock, task) = start_mock().await;
        mock.set(
            "cline-key-1",
            vec![Spec::json(
                429,
                r#"{"error":{"message":"Daily free limit reached. Try again in 9h"}}"#,
            )],
        )
        .await;
        mock.set("cline-key-2", vec![Spec::json(200, successful_json("ok"))])
            .await;
        let state_path = unique_state_path("admin");
        let mut config = test_config(base, 2);
        config.runtime.state_file = Some(state_path.to_string_lossy().into_owned());
        let state = AppState::new(config).unwrap();
        let app = router(state.clone());
        let unauthenticated = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/admin/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthenticated.status(), StatusCode::UNAUTHORIZED);
        // Trigger a 429 so rate-limit metadata exists, then flush state.
        let response = app
            .clone()
            .oneshot(gateway_request("/v1/chat/completions", chat_body(false)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let status = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/admin/status")
                    .header(header::AUTHORIZATION, "Bearer gateway-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(status.status(), StatusCode::OK);
        let body = response_text(status).await;
        let value: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["keys"][0]["state"], "cooling");
        assert_eq!(value["keys"][0]["rate_limit_kind"], "daily_quota");
        assert_eq!(value["active_key"], "cline-2");
        assert!(!body.contains("cline-key-1"));
        assert!(!body.contains("cline-key-2"));
        assert!(!body.contains("gateway-secret"));
        assert!(!body.contains("Daily free limit"));
        state.flush_runtime_state();
        std::fs::remove_dir_all(state_path.parent().unwrap()).ok();
        task.abort();
    }
}
