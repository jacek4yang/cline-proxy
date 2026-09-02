//! Axum routes, authentication, protocol dispatch, and graceful shutdown.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
use crate::config::Config;
use crate::pool::KeyPool;
use crate::redaction::{sanitize_json, sanitize_text};
use crate::upstream::{
    transport_error_class, BufferedUpstreamError, ClineUpstream, UpstreamError, UpstreamResponse,
    UpstreamResult,
};

const MAX_UPSTREAM_BODY_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub upstream: ClineUpstream,
}

impl AppState {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let pool = KeyPool::new(&config.cline_api_keys);
        if pool.is_empty() {
            anyhow::bail!("at least one enabled Cline API key is required");
        }
        let upstream = ClineUpstream::new(&config, pool)?;
        Ok(Self {
            config: Arc::new(config),
            upstream,
        })
    }
}

pub fn router(state: AppState) -> Router {
    let max_request_bytes = state.config.runtime.max_request_bytes;
    let protected = Router::new()
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .route("/v1/chat/completions", post(openai_chat))
        .route("/v1/models", get(models))
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
        request_bytes = upstream_body.len(),
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
            ))
            .unwrap_or_else(|_| Response::new(Body::empty()));
        insert_request_id(response.headers_mut(), &request_id);
        return response;
    }
    let value = match anthropic::parse_json_response(response).await {
        Ok(value) => value,
        Err(error) => return protocol_error(error, StatusCode::BAD_GATEWAY, &request_id),
    };
    match anthropic::convert_response(&value, &request_id, &upstream_model) {
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
    let count = match anthropic::approximate_input_tokens(&body) {
        Ok(count) => count,
        Err(error) => return protocol_error(error, StatusCode::BAD_REQUEST, &request_id),
    };
    tracing::info!(
        request_id,
        protocol = "anthropic",
        requested_model,
        upstream_model,
        input_tokens = count,
        token_count = "local_approximation",
        "token count completed"
    );
    let mut response = json_response(StatusCode::OK, json!({"input_tokens":count}), &request_id);
    response.headers_mut().insert(
        "x-cline-proxy-token-count",
        HeaderValue::from_static("approximate"),
    );
    response
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
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let app = router(state);
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
    tokio::select! {
        result = &mut task => {
            result.context("gateway task failed")?.context("serving HTTP")?;
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
            signal?;
        }
    }
    Ok(())
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
        assert_eq!(value["content"][0]["type"], "thinking");
        assert_eq!(value["content"][2]["type"], "tool_use");
        assert_eq!(value["content"][2]["input"]["file_path"], "/tmp/a");
        assert_eq!(value["stop_reason"], "tool_use");
        assert_eq!(value["usage"]["output_tokens"], 4);
        let seen = mock.seen().await;
        assert_eq!(seen[0].body["messages"][1]["role"], "tool");
        assert_eq!(seen[0].body["model"], "z-ai/glm-5.3-flash");
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
        assert!(body.contains("thinking_delta"));
        assert!(body.contains("signature_delta"));
        assert!(body.contains("\"stop_reason\":\"tool_use\""));
        assert!(body.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
        task.abort();
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
        assert_eq!(count.headers()["x-cline-proxy-token-count"], "approximate");
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
}
