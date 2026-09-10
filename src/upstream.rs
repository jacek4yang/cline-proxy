//! Shared Cline HTTP client and effective-HTTP-429-only failover loop.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use bytes::Bytes;
use thiserror::Error;

use crate::config::Config;
use crate::pool::{KeyPool, ProbeLease, SelectedKey};
use crate::proxy_route::ProxyRoute;
use crate::rate_limit::{
    classify_rate_limit_kind, classify_upstream_response, ClassifiedUpstreamResponse,
    RetryHintSource,
};
use crate::redaction::sanitize_text;

const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

/// Per-logical-request data shared by every failover attempt: the body,
/// protocol flags, and session-affinity header must be byte-identical for
/// key A and key B; only Authorization rotates between attempts.
struct LogicalRequest<'a> {
    body: &'a Bytes,
    stream: bool,
    request_id: &'a str,
    model: &'a str,
    task_id: Option<&'a str>,
}

#[derive(Clone)]
pub struct ClineUpstream {
    inner: Arc<Inner>,
}

struct Inner {
    http: reqwest::Client,
    chat_url: reqwest::Url,
    headers: HeaderMap,
    pool: KeyPool,
    fallback_cooldown: Duration,
    exact_secrets: Vec<String>,
    route: ProxyRoute,
}

pub struct UpstreamResult {
    pub response: UpstreamResponse,
    pub selected: SelectedKey,
    pub attempt: usize,
    pub failover_count: usize,
}

pub enum UpstreamResponse {
    Success(reqwest::Response),
    HttpError(BufferedUpstreamError),
}

impl UpstreamResponse {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::Success(response) => response.status(),
            Self::HttpError(error) => error.classification.outer_status,
        }
    }
}

pub struct BufferedUpstreamError {
    pub headers: HeaderMap,
    pub body: Bytes,
    pub classification: ClassifiedUpstreamResponse,
}

#[derive(Debug, Error)]
pub enum UpstreamError {
    #[error("upstream request failed")]
    Transport(#[source] reqwest::Error),
    #[error("all Cline API keys are currently rate-limited")]
    AllRateLimited { retry_after: Option<Duration> },
}

impl ClineUpstream {
    pub fn new(config: &Config, pool: KeyPool) -> Result<Self> {
        let route = config.upstream.proxy_route()?;
        let builder = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(config.upstream.connect_timeout_secs))
            // Like the reference gateway, this is an inactivity timeout. A
            // healthy SSE response may live longer while chunks keep arriving.
            .read_timeout(Duration::from_secs(config.upstream.timeout_secs))
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_nodelay(true)
            .tcp_keepalive(Duration::from_secs(60))
            .http2_keep_alive_interval(Duration::from_secs(30))
            .http2_keep_alive_timeout(Duration::from_secs(20))
            .http2_keep_alive_while_idle(true)
            .gzip(true);
        // Direct: ignore env/system proxies. SOCKS: explicit Cline-only hop.
        let http = route
            .apply(builder)?
            .build()
            .context("building shared Cline HTTP client")?;
        let base = config.upstream.base_url.trim_end_matches('/');
        let chat_url =
            reqwest::Url::parse(&format!("{}{}", base, config.upstream.chat_path.as_str()))
                .context("building Cline chat-completions URL")?;
        let mut headers = HeaderMap::new();
        for (name, value) in &config.upstream.headers {
            headers.insert(
                HeaderName::try_from(name.as_str()).context("invalid configured header name")?,
                HeaderValue::try_from(value.as_str()).context("invalid configured header value")?,
            );
        }
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        headers.insert(header::ACCEPT_ENCODING, HeaderValue::from_static("gzip"));
        let mut exact_secrets = config
            .cline_api_keys
            .iter()
            .map(|key| key.api_key.clone())
            .collect::<Vec<_>>();
        exact_secrets.push(config.server.api_key.clone());
        if let Some(secret) = route.credential_secret() {
            exact_secrets.push(secret.to_owned());
        }
        Ok(Self {
            inner: Arc::new(Inner {
                http,
                chat_url,
                headers,
                pool,
                fallback_cooldown: Duration::from_secs(config.upstream.fallback_429_cooldown_secs),
                exact_secrets,
                route,
            }),
        })
    }

    pub fn pool(&self) -> &KeyPool {
        &self.inner.pool
    }

    pub fn route(&self) -> &ProxyRoute {
        &self.inner.route
    }

    pub fn exact_secrets(&self) -> Vec<&str> {
        self.inner
            .exact_secrets
            .iter()
            .map(String::as_str)
            .collect()
    }

    /// Send one logical chat request. Only a classified effective HTTP 429 can
    /// enter the retry branch. Transport errors return before response-body
    /// classification, and successful responses retain streaming ownership.
    ///
    /// `task_id` is the server-derived session fingerprint (X-Task-ID). It is
    /// computed ONCE per logical request and passed unchanged through every
    /// failover attempt, so key A and key B observe the same body, the same
    /// X-Task-ID, and differ only in Authorization. `None` sends no header —
    /// without a reliable session identity affinity is never faked.
    pub async fn send_chat(
        &self,
        body: Bytes,
        stream: bool,
        request_id: &str,
        model: &str,
        task_id: Option<&str>,
    ) -> std::result::Result<UpstreamResult, UpstreamError> {
        // Everything a failover attempt needs that must stay byte-identical
        // across attempts lives here; only Authorization rotates.
        let logical = LogicalRequest {
            body: &body,
            stream,
            request_id,
            model,
            task_id,
        };
        let mut attempted = HashSet::with_capacity(self.inner.pool.len());
        let mut failed_probes_ms = 0u128;
        loop {
            let Some(selected) = self.inner.pool.select(&attempted) else {
                return Err(UpstreamError::AllRateLimited {
                    retry_after: self.inner.pool.earliest_retry_after(),
                });
            };
            attempted.insert(selected.index);
            let attempt = attempted.len();
            // Guards the HalfOpen single-flight probe. Dropping it (early
            // return, transport error, or client cancellation) releases the
            // lease so the key stays probe-eligible.
            let _lease: Option<ProbeLease<'_>> = selected
                .is_probe
                .then(|| self.inner.pool.probe_lease(selected.index));
            let (response, attempt_elapsed) = self
                .send_once(&selected, &logical, attempt)
                .await
                .map_err(UpstreamError::Transport)?;

            let outer_status = response.status();
            if outer_status.is_success() {
                // A usable 2xx response proves the key's quota is available:
                // clear cooldowns, close HalfOpen probing, count success.
                self.inner.pool.mark_success(selected.index);
                if attempt > 1 || failed_probes_ms > 0 {
                    tracing::info!(
                        request_id,
                        successful_key = %selected.name,
                        attempts = attempt,
                        failovers = attempt.saturating_sub(1),
                        failed_key_probe_ms = failed_probes_ms,
                        successful_upstream_headers_ms = attempt_elapsed.as_millis(),
                        // Transport success only: the body may still fail
                        // protocol validation downstream (issue #14). The
                        // logical completion is logged by the protocol layer.
                        "upstream HTTP response selected after failover"
                    );
                }
                return Ok(UpstreamResult {
                    response: UpstreamResponse::Success(response),
                    selected,
                    attempt,
                    failover_count: attempt.saturating_sub(1),
                });
            }

            let headers = response.headers().clone();
            let error_body = read_limited(response, MAX_ERROR_BODY_BYTES).await;
            let classification = classify_upstream_response(
                outer_status,
                &headers,
                &error_body,
                self.inner.fallback_cooldown,
            );
            if !classification.disposition.is_rate_limited() {
                return Ok(UpstreamResult {
                    response: UpstreamResponse::HttpError(BufferedUpstreamError {
                        headers,
                        body: error_body,
                        classification,
                    }),
                    selected,
                    attempt,
                    failover_count: attempt.saturating_sub(1),
                });
            }
            failed_probes_ms = failed_probes_ms.saturating_add(attempt_elapsed.as_millis());
            let Some(hint) = classification.retry_hint.as_ref() else {
                // A rate-limited classification always carries a retry hint.
                // If that internal invariant changes, fail closed without
                // rotating rather than risking an unbounded retry policy.
                return Ok(UpstreamResult {
                    response: UpstreamResponse::HttpError(BufferedUpstreamError {
                        headers,
                        body: error_body,
                        classification,
                    }),
                    selected,
                    attempt,
                    failover_count: attempt.saturating_sub(1),
                });
            };
            let secret_refs = self.exact_secrets();
            let safe_message = hint
                .message
                .as_deref()
                .map(|message| sanitize_text(message, &secret_refs));
            let safe_model = hint
                .model
                .as_deref()
                .map(|model| sanitize_text(model, &secret_refs));
            // Kind classification happens only here, after the effective 429
            // is already confirmed. It never widens failover conditions.
            let kind = classify_rate_limit_kind(safe_message.as_deref(), hint.duration);
            let update = self.inner.pool.mark_http_429(
                selected.index,
                hint.duration,
                kind,
                safe_message.clone(),
                safe_model.clone(),
            );
            let will_failover = attempted.len() < self.inner.pool.len();
            tracing::warn!(
                request_id,
                selected_key_name = %selected.name,
                selected_key_index = selected.configured_index,
                outer_status = classification.outer_status.as_u16(),
                effective_status = classification.effective_status.as_u16(),
                error_class = classification.disposition.error_class(),
                rate_limit_kind = kind.as_str(),
                attempt,
                failover_count = attempt,
                failover = will_failover,
                cooldown_ms = update.cooldown.as_millis(),
                retry_after_secs = duration_ceil_secs(update.cooldown),
                cooldown_until_unix = update.cooldown_until
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                cooldown_source = retry_source_name(hint.source),
                rate_limited_model = safe_model.as_deref().unwrap_or(""),
                rate_limit_message = safe_message.as_deref().unwrap_or(""),
                "effective upstream HTTP 429 placed Cline key in cooldown"
            );
            if attempted.len() >= self.inner.pool.len() {
                return Err(UpstreamError::AllRateLimited {
                    retry_after: self.inner.pool.earliest_retry_after(),
                });
            }
        }
    }

    /// One failover attempt. `request` carries everything that must stay
    /// byte-identical across attempts; only Authorization rotates here.
    async fn send_once(
        &self,
        selected: &SelectedKey,
        request: &LogicalRequest<'_>,
        attempt: usize,
    ) -> std::result::Result<(reqwest::Response, Duration), reqwest::Error> {
        let LogicalRequest {
            body,
            stream,
            request_id,
            model,
            task_id,
        } = request;
        let started = Instant::now();
        let mut headers = self.inner.headers.clone();
        headers.insert(
            header::ACCEPT,
            HeaderValue::from_static(if *stream {
                "text/event-stream"
            } else {
                "application/json"
            }),
        );
        // Dynamic session affinity (X-Task-ID): inserted after the configured
        // static headers so the per-session fingerprint always wins over any
        // user-configured value (config validation also rejects a configured
        // `x-task-id`). Identical for every attempt of one logical request.
        if let Some(task_id) = task_id {
            if let Ok(value) = HeaderValue::try_from(*task_id) {
                headers.insert("x-task-id", value);
            }
        }
        // Construct authorization last and never copy caller headers. Config
        // validation also forbids every authentication/framing override.
        let authorization = format!("Bearer {}", selected.api_key());
        match HeaderValue::try_from(authorization) {
            Ok(value) => {
                headers.insert(header::AUTHORIZATION, value);
            }
            Err(_) => {
                // Startup validation rejects this case. Keep request building
                // defensive so future call-sequence changes still return a
                // reqwest error without leaking key material.
                return self
                    .inner
                    .http
                    .post(self.inner.chat_url.clone())
                    .header(header::AUTHORIZATION, "\n")
                    .send()
                    .await
                    .map(|response| (response, started.elapsed()));
            }
        }
        if let Ok(value) = HeaderValue::try_from(*request_id) {
            headers.insert("x-request-id", value);
        }
        let result = self
            .inner
            .http
            .post(self.inner.chat_url.clone())
            .headers(headers)
            .body((*body).clone())
            .send()
            .await;
        let elapsed = started.elapsed();
        match &result {
            Ok(response) => tracing::debug!(
                request_id,
                requested_model = *model,
                selected_key_name = %selected.name,
                selected_key_index = selected.configured_index,
                attempt,
                route = self.inner.route.kind(),
                outer_status = response.status().as_u16(),
                duration_ms = elapsed.as_millis(),
                stream,
                "Cline upstream response"
            ),
            Err(error) => tracing::warn!(
                request_id,
                requested_model = *model,
                selected_key_name = %selected.name,
                selected_key_index = selected.configured_index,
                attempt,
                route = self.inner.route.kind(),
                duration_ms = elapsed.as_millis(),
                stream,
                error_class = transport_error_class(error),
                "Cline upstream request failed without failover"
            ),
        }
        result.map(|response| (response, elapsed))
    }
}

async fn read_limited(mut response: reqwest::Response, limit: usize) -> Bytes {
    let mut output = Vec::new();
    while let Ok(Some(chunk)) = response.chunk().await {
        let remaining = limit.saturating_sub(output.len());
        if remaining == 0 {
            break;
        }
        output.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
        if output.len() >= limit {
            break;
        }
    }
    Bytes::from(output)
}

fn retry_source_name(source: RetryHintSource) -> &'static str {
    match source {
        RetryHintSource::RetryAfter => "retry_after_header",
        RetryHintSource::StructuredJson => "structured_json",
        RetryHintSource::HumanText => "human_text",
        RetryHintSource::Fallback => "fallback",
    }
}

fn duration_ceil_secs(duration: Duration) -> u64 {
    duration
        .as_secs()
        .saturating_add(u64::from(duration.subsec_nanos() > 0))
}

pub fn transport_error_class(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "timeout"
    } else if error.is_connect() {
        "connect"
    } else if error.is_builder() {
        "builder"
    } else if error.is_request() {
        "request"
    } else if error.is_decode() {
        "decode"
    } else {
        "transport"
    }
}
