//! Shared Cline HTTP client and the exact HTTP-429-only failover loop.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use bytes::Bytes;
use thiserror::Error;

use crate::config::Config;
use crate::pool::{KeyPool, SelectedKey};
use crate::rate_limit::{retry_hint, RetryHintSource};
use crate::redaction::sanitize_text;

const MAX_RATE_LIMIT_BODY_BYTES: usize = 64 * 1024;

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
}

pub struct UpstreamResult {
    pub response: reqwest::Response,
    pub selected: SelectedKey,
    pub attempt: usize,
    pub failover_count: usize,
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
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(config.upstream.connect_timeout_secs))
            // Like the reference gateway, this is an inactivity timeout. A
            // healthy SSE response may live longer while chunks keep arriving.
            .read_timeout(Duration::from_secs(config.upstream.timeout_secs))
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_keepalive(Duration::from_secs(60))
            .http2_keep_alive_interval(Duration::from_secs(30))
            .http2_keep_alive_timeout(Duration::from_secs(20))
            .http2_keep_alive_while_idle(true)
            .gzip(true)
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
        Ok(Self {
            inner: Arc::new(Inner {
                http,
                chat_url,
                headers,
                pool,
                fallback_cooldown: Duration::from_secs(config.upstream.fallback_429_cooldown_secs),
                exact_secrets,
            }),
        })
    }

    pub fn pool(&self) -> &KeyPool {
        &self.inner.pool
    }

    pub fn exact_secrets(&self) -> Vec<&str> {
        self.inner
            .exact_secrets
            .iter()
            .map(String::as_str)
            .collect()
    }

    /// Send one logical chat request. The equality check below is the sole
    /// failover gate in the gateway: only an actual HTTP 429 enters the retry
    /// branch. Every other HTTP status and every transport error returns now.
    pub async fn send_chat(
        &self,
        body: Bytes,
        stream: bool,
        request_id: &str,
        model: &str,
    ) -> std::result::Result<UpstreamResult, UpstreamError> {
        let mut attempted = HashSet::with_capacity(self.inner.pool.len());
        loop {
            let Some(selected) = self.inner.pool.select(&attempted) else {
                return Err(UpstreamError::AllRateLimited {
                    retry_after: self.inner.pool.earliest_retry_after(),
                });
            };
            attempted.insert(selected.index);
            let attempt = attempted.len();
            let response = self
                .send_once(&selected, body.clone(), stream, request_id, model, attempt)
                .await
                .map_err(UpstreamError::Transport)?;

            // CORE INVARIANT: no status other than exactly 429 can switch keys.
            if response.status() != StatusCode::TOO_MANY_REQUESTS {
                return Ok(UpstreamResult {
                    response,
                    selected,
                    attempt,
                    failover_count: attempt.saturating_sub(1),
                });
            }

            let headers = response.headers().clone();
            let body = read_limited(response, MAX_RATE_LIMIT_BODY_BYTES).await;
            let hint = retry_hint(&headers, &body, self.inner.fallback_cooldown);
            let secret_refs = self.exact_secrets();
            let safe_message = hint
                .message
                .as_deref()
                .map(|message| sanitize_text(message, &secret_refs));
            let safe_model = hint
                .model
                .as_deref()
                .map(|model| sanitize_text(model, &secret_refs));
            let update = self.inner.pool.mark_http_429(
                selected.index,
                hint.duration,
                safe_message.clone(),
                safe_model.clone(),
            );
            tracing::warn!(
                request_id,
                selected_key_name = %selected.name,
                selected_key_index = selected.configured_index,
                upstream_status = 429,
                attempt,
                failover_count = attempt,
                cooldown_ms = update.cooldown.as_millis(),
                cooldown_until_unix = update.cooldown_until
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                cooldown_source = retry_source_name(hint.source),
                retry_model = safe_model.as_deref().unwrap_or(""),
                rate_limit_message = safe_message.as_deref().unwrap_or(""),
                "Cline key entered cooldown; trying the next eligible key"
            );
            if attempted.len() >= self.inner.pool.len() {
                return Err(UpstreamError::AllRateLimited {
                    retry_after: self.inner.pool.earliest_retry_after(),
                });
            }
        }
    }

    async fn send_once(
        &self,
        selected: &SelectedKey,
        body: Bytes,
        stream: bool,
        request_id: &str,
        model: &str,
        attempt: usize,
    ) -> std::result::Result<reqwest::Response, reqwest::Error> {
        let started = Instant::now();
        let mut headers = self.inner.headers.clone();
        headers.insert(
            header::ACCEPT,
            HeaderValue::from_static(if stream {
                "text/event-stream"
            } else {
                "application/json"
            }),
        );
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
                    .await;
            }
        }
        if let Ok(value) = HeaderValue::try_from(request_id) {
            headers.insert("x-request-id", value);
        }
        let result = self
            .inner
            .http
            .post(self.inner.chat_url.clone())
            .headers(headers)
            .body(body)
            .send()
            .await;
        match &result {
            Ok(response) => tracing::info!(
                request_id,
                requested_model = model,
                selected_key_name = %selected.name,
                selected_key_index = selected.configured_index,
                attempt,
                upstream_status = response.status().as_u16(),
                duration_ms = started.elapsed().as_millis(),
                stream,
                "Cline upstream response"
            ),
            Err(error) => tracing::warn!(
                request_id,
                requested_model = model,
                selected_key_name = %selected.name,
                selected_key_index = selected.configured_index,
                attempt,
                duration_ms = started.elapsed().as_millis(),
                stream,
                error_class = transport_error_class(error),
                "Cline upstream request failed without failover"
            ),
        }
        result
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
