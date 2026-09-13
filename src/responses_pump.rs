//! Responses streaming pump and non-stream aggregation.
//!
//! Mirrors the Anthropic stream discipline (`anthropic::stream_body`): the
//! same `StreamWatch` timers, SSE comment keepalive (never a synthetic event
//! frame -- that would fail Grok Build's typed deserialization), bounded
//! buffers, and the no-replay rule after downstream commitment. Only the
//! downstream event rendering differs (chat SSE to Responses SSE via
//! `ResponsesConverter`).

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::response::IntoResponse;
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::Value;

use crate::responses::stream::ResponsesConverter;
use crate::stream_watch::{StreamTimeouts, StreamWatch};

const MAX_UPSTREAM_SSE_LINE_BYTES: usize = 8 * 1024 * 1024;

#[cfg(not(test))]
const PING_INTERVAL: Duration = Duration::from_secs(10);
#[cfg(test)]
const PING_INTERVAL: Duration = Duration::from_millis(25);

/// One frame batch forwarded to the downstream body.
enum PumpItem {
    Bytes(Vec<u8>),
    Finished,
}

/// Terminal stats handed from the pump task to the summary task.
#[derive(Default)]
struct PumpStats {
    ttft_ms: Option<u128>,
    first_reasoning_ms: Option<u128>,
    first_text_ms: Option<u128>,
    first_tool_ms: Option<u128>,
    first_event_ms: Option<u128>,
    last_byte_ms: Option<u128>,
    last_semantic_ms: Option<u128>,
    got_first_byte: bool,
    duration_ms: u128,
    finish_reason: Option<String>,
    /// Upstream usage rendered in the OpenAI shape the summary builder reads.
    usage: Option<Value>,
    text_bytes: u64,
    reasoning_bytes: u64,
    tool_bytes: u64,
    error: bool,
}

fn openai_usage_shape(usage: (u64, u64, u64, u64)) -> Value {
    let (input, output, cached, reasoning) = usage;
    serde_json::json!({
        "prompt_tokens": input,
        "completion_tokens": output,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    })
}

/// Commit shadow state for one finished generation round (same rule as the
/// other frontends: tool-loop rounds keep the reasoning, final answers clear).
fn commit_shadow(
    converter: &ResponsesConverter,
    shadow: Option<&Arc<crate::reasoning_shadow::ReasoningShadowStore>>,
    session_fp: Option<&str>,
) {
    let (Some(shadow), Some(fp)) = (shadow, session_fp) else {
        return;
    };
    let ids = converter.tool_call_ids();
    if converter.finish_reason() == Some("tool_calls") {
        shadow.store(fp, &ids, converter.reasoning_text());
    } else if converter.finish_reason().is_some() {
        shadow.clear_session(fp);
    }
}

/// Response-side stream pump: consumes upstream chat SSE into a Responses SSE
/// stream, with the same watchdog/ping/no-replay discipline as the Anthropic
/// pump. Emits an SSE comment keepalive (`: ping`) —never a synthetic event
/// frame, which would fail the client's typed deserialization.
#[allow(clippy::too_many_arguments)]
pub(crate) fn responses_stream_response(
    response: reqwest::Response,
    expose_thinking: bool,
    model: String,
    session_fp: Option<String>,
    shadow: Option<Arc<crate::reasoning_shadow::ReasoningShadowStore>>,
    timeouts: StreamTimeouts,
    summary: crate::obs::StreamSummary,
    key_name: String,
    request_id: String,
) -> axum::response::Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<PumpItem>(64);
    let (stats_tx, stats_rx) = tokio::sync::oneshot::channel::<PumpStats>();
    let started_at = summary.started;

    let _pump_task = tokio::spawn(async move {
        let mut converter = ResponsesConverter::new(&model, expose_thinking);
        let mut watch = StreamWatch::new(timeouts, tokio::time::Instant::now());
        let mut buffer: Vec<u8> = Vec::with_capacity(8192);
        let mut stats = PumpStats::default();
        let mut byte_stream = response.bytes_stream();
        let mut ping = tokio::time::interval(PING_INTERVAL);
        ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut out: Vec<u8> = Vec::with_capacity(4096);
        let mut done = false;

        while !done {
            tokio::select! {
                biased;
                _ = tx.closed() => {
                    // Client disconnect: dropping the pump cancels the
                    // upstream body. Nothing committed may be replayed.
                    return;
                }
                _ = ping.tick() => {
                    out.extend_from_slice(crate::responses::types::sse_ping().as_bytes());
                }
                _ = tokio::time::sleep_until(watch.next_deadline()) => {
                    if let Some(stall) = watch.check(tokio::time::Instant::now()) {
                        tracing::warn!(
                            request_id = %request_id,
                            error_class = stall.as_str(),
                            "upstream Responses stream stalled; request will not be replayed"
                        );
                        converter.finish(Some(stall.message()), &mut out);
                        stats.error = true;
                        done = true;
                    }
                }
                chunk = byte_stream.next() => {
                    match chunk {
                        None => {
                            let error = (converter.finish_reason().is_none())
                                .then_some("upstream stream ended unexpectedly");
                            if let Some(message) = error {
                                converter.finish(Some(message), &mut out);
                                stats.error = true;
                            } else {
                                converter.finish(None, &mut out);
                            }
                            done = true;
                        }
                        Some(Err(err)) => {
                            tracing::warn!(
                                request_id = %request_id,
                                error_class = crate::upstream::transport_error_class(&err),
                                "upstream Responses stream interrupted; request will not be replayed"
                            );
                            converter.finish(
                                Some("upstream stream was interrupted"),
                                &mut out,
                            );
                            stats.error = true;
                            done = true;
                        }
                        Some(Ok(bytes)) => {
                            let now = tokio::time::Instant::now();
                            watch.on_upstream_bytes(now);
                            let elapsed = started_at.elapsed().as_millis();
                            if !stats.got_first_byte {
                                stats.got_first_byte = true;
                                stats.first_event_ms = Some(elapsed);
                            }
                            stats.last_byte_ms = Some(elapsed);
                            buffer.extend_from_slice(&bytes);
                            while let Some(newline) = buffer.iter().position(|&b| b == b'\n') {
                                let line: Vec<u8> = buffer.drain(..=newline).collect();
                                let line = trim_ascii(&line[..line.len() - 1]);
                                if line.is_empty() {
                                    continue;
                                }
                                if line.len() > MAX_UPSTREAM_SSE_LINE_BYTES {
                                    continue;
                                }
                                watch.on_sse_event();
                                let Some(payload) = line.strip_prefix(b"data:") else {
                                    continue;
                                };
                                let payload = trim_ascii(payload);
                                if payload == b"[DONE]" {
                                    let error = (converter.finish_reason().is_none())
                                        .then_some("upstream stream ended unexpectedly");
                                    if let Some(message) = error {
                                        converter.finish(Some(message), &mut out);
                                        stats.error = true;
                                    } else {
                                        converter.finish(None, &mut out);
                                    }
                                    done = true;
                                    break;
                                }
                                let Ok(chunk_value) = serde_json::from_slice::<Value>(payload)
                                else {
                                    continue;
                                };
                                let semantic = converter.feed_chunk(&chunk_value, &mut out);
                                if semantic {
                                    let elapsed = started_at.elapsed().as_millis();
                                    watch.on_semantic(tokio::time::Instant::now());
                                    stats.last_semantic_ms = Some(elapsed);
                                    if stats.ttft_ms.is_none() {
                                        stats.ttft_ms = Some(elapsed);
                                    }
                                    if stats.first_reasoning_ms.is_none()
                                        && converter.has_reasoning()
                                    {
                                        stats.first_reasoning_ms = Some(elapsed);
                                    }
                                    if stats.first_text_ms.is_none() && converter.has_text() {
                                        stats.first_text_ms = Some(elapsed);
                                    }
                                    if stats.first_tool_ms.is_none() && converter.has_tools() {
                                        stats.first_tool_ms = Some(elapsed);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if !out.is_empty() {
                let payload = std::mem::take(&mut out);
                if tx.send(PumpItem::Bytes(payload)).await.is_err() {
                    return;
                }
            }
        }
        stats.duration_ms = started_at.elapsed().as_millis();
        stats.finish_reason = converter.finish_reason().map(str::to_owned);
        stats.text_bytes = converter.text_bytes();
        stats.reasoning_bytes = converter.reasoning_bytes();
        stats.tool_bytes = converter.tool_bytes();
        if let Some(usage) = converter.usage() {
            stats.usage = Some(openai_usage_shape(usage));
        }
        commit_shadow(&converter, shadow.as_ref(), session_fp.as_deref());
        if !out.is_empty() {
            let _ = tx.send(PumpItem::Bytes(std::mem::take(&mut out))).await;
        }
        let _ = tx.send(PumpItem::Finished).await;
        let _ = stats_tx.send(stats);
    });

    let _summary_task = tokio::spawn(async move {
        let Ok(stats) = stats_rx.await else {
            return;
        };
        emit_stream_summary(&summary, key_name.as_str(), &stats, &summary.request_id);
    });

    let byte_stream = futures_util::stream::unfold(rx, |mut rx| async move {
        loop {
            match rx.recv().await {
                Some(PumpItem::Bytes(bytes)) => {
                    return Some((Ok::<_, std::convert::Infallible>(Bytes::from(bytes)), rx));
                }
                Some(PumpItem::Finished) => continue,
                None => return None,
            }
        }
    });
    let mut response = axum::body::Body::from_stream(byte_stream).into_response();
    response.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("text/event-stream"),
    );
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-cache"),
    );
    response.headers_mut().insert(
        axum::http::header::HeaderName::from_static("x-accel-buffering"),
        axum::http::HeaderValue::from_static("no"),
    );
    response
}

/// Non-stream Responses: ONE upstream stream, aggregated locally into a
/// single Responses object by the same converter the streaming path uses —/// never a second generation for shape conversion.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn aggregate_responses(
    response: reqwest::Response,
    expose_thinking: bool,
    model: String,
    session_fp: Option<String>,
    shadow: Option<Arc<crate::reasoning_shadow::ReasoningShadowStore>>,
    timeouts: StreamTimeouts,
    started_at: Instant,
    summary: &mut crate::obs::StreamSummary,
    request_id: &str,
    key_name: &str,
) -> Result<Value, crate::anthropic::ProtocolError> {
    let mut converter = ResponsesConverter::new(&model, expose_thinking);
    let mut watch = StreamWatch::new(timeouts, tokio::time::Instant::now());
    let mut buffer: Vec<u8> = Vec::with_capacity(8192);
    let mut byte_stream = response.bytes_stream();
    let mut ttft_ms: Option<u128> = None;
    let result = 'outer: loop {
        tokio::select! {
            _ = tokio::time::sleep_until(watch.next_deadline()) => {
                if let Some(stall) = watch.check(tokio::time::Instant::now()) {
                    break 'outer Err(crate::anthropic::ProtocolError::stall(stall));
                }
            }
            chunk = byte_stream.next() => {
                match chunk {
                    None => {
                        if converter.finish_reason().is_none() {
                            break 'outer Err(crate::anthropic::ProtocolError::upstream(
                                "upstream stream ended unexpectedly",
                            ));
                        }
                        converter.finish(None, &mut Vec::new());
                        break 'outer Ok(());
                    }
                    Some(Err(_)) => {
                        break 'outer Err(crate::anthropic::ProtocolError::upstream(
                            "upstream stream was interrupted",
                        ));
                    }
                    Some(Ok(bytes)) => {
                        let now = tokio::time::Instant::now();
                        watch.on_upstream_bytes(now);
                        buffer.extend_from_slice(&bytes);
                        while let Some(newline) = buffer.iter().position(|&b| b == b'\n') {
                            let line: Vec<u8> = buffer.drain(..=newline).collect();
                            let line = trim_ascii(&line[..line.len() - 1]);
                            if line.is_empty() {
                                continue;
                            }
                            if line.len() > MAX_UPSTREAM_SSE_LINE_BYTES {
                                continue;
                            }
                            watch.on_sse_event();
                            let Some(payload) = line.strip_prefix(b"data:") else {
                                continue;
                            };
                            let payload = trim_ascii(payload);
                            if payload == b"[DONE]" {
                                if converter.finish_reason().is_none() {
                                    break 'outer Err(crate::anthropic::ProtocolError::upstream(
                                        "upstream stream ended unexpectedly",
                                    ));
                                }
                                converter.finish(None, &mut Vec::new());
                                break 'outer Ok(());
                            }
                            let Ok(chunk_value) = serde_json::from_slice::<Value>(payload) else {
                                continue;
                            };
                            let semantic = converter.feed_chunk(&chunk_value, &mut Vec::new());
                            if semantic {
                                let elapsed = started_at.elapsed().as_millis();
                                watch.on_semantic(tokio::time::Instant::now());
                                if ttft_ms.is_none() {
                                    ttft_ms = Some(elapsed);
                                }
                            }
                        }
                    }
                }
            }
        }
    };
    match result {
        Ok(()) => {
            commit_shadow(&converter, shadow.as_ref(), session_fp.as_deref());
            let stats = PumpStats {
                duration_ms: started_at.elapsed().as_millis(),
                finish_reason: converter.finish_reason().map(str::to_owned),
                text_bytes: converter.text_bytes(),
                reasoning_bytes: converter.reasoning_bytes(),
                tool_bytes: converter.tool_bytes(),
                usage: converter.usage().map(openai_usage_shape),
                ..PumpStats::default()
            };
            emit_stream_summary(summary, key_name, &stats, request_id);
            Ok(converter.nonstream_response())
        }
        Err(error) => Err(error),
    }
}

/// Emit the single per-request summary for a completed Responses stream.
fn emit_stream_summary(
    summary: &crate::obs::StreamSummary,
    key_name: &str,
    stats: &PumpStats,
    request_id: &str,
) {
    let mut builder = crate::obs::SummaryBuilder::new(
        request_id,
        "responses",
        summary.requested_model.clone(),
        summary.upstream_model.clone(),
        summary.model_family,
        summary.session.clone(),
        summary.downstream_stream,
        summary.upstream_strategy,
        summary.started,
        summary.slow_ttft_ms,
        summary.slow_duration_ms,
    );
    let outcome: &'static str = if stats.error {
        builder.mark_anomalous("upstream_error");
        "upstream_error"
    } else {
        "complete"
    };
    builder.emit(crate::obs::SummaryEmit {
        sink: summary.sink.as_ref(),
        selected_key_name: key_name,
        route: summary.route,
        attempts: 1,
        failover_count: 0,
        reasoning_effort: summary.reasoning_effort,
        expose_thinking: summary.expose_thinking,
        client_max_tokens: summary.client_max_tokens,
        effective_max_tokens: summary.effective_max_tokens,
        request_bytes: summary.request_bytes,
        upstream_request_bytes: summary.upstream_request_bytes,
        system_bytes: summary.system_bytes,
        messages_bytes: summary.messages_bytes,
        tools_bytes: summary.tools_bytes,
        historical_reasoning_bytes_removed: summary.historical_reasoning_bytes_removed,
        billing_header_bytes_removed: summary.billing_header_bytes_removed,
        canonicalized_arguments: summary.canonicalized_arguments,
        usage: stats.usage.as_ref(),
        local_tokens: summary.local_tokens.get(),
        context_limit_tokens: summary.context_limit_tokens,
        first_reasoning_ms: stats.first_reasoning_ms,
        first_text_ms: stats.first_text_ms,
        first_tool_call_ms: stats.first_tool_ms,
        first_semantic_ms: stats
            .first_reasoning_ms
            .or(stats.first_text_ms)
            .or(stats.first_tool_ms),
        first_sse_event_ms: stats.first_event_ms,
        first_upstream_byte_ms: stats.first_event_ms,
        upstream_headers_ms: summary.upstream_headers_ms,
        last_upstream_progress_ms: stats.last_byte_ms,
        last_semantic_progress_ms: stats.last_semantic_ms,
        upstream_first_event_ms: stats.first_event_ms,
        upstream_duration_ms: Some(stats.duration_ms),
        text_bytes: stats.text_bytes,
        reasoning_bytes: stats.reasoning_bytes,
        tool_call_bytes: stats.tool_bytes,
        text_events: 0,
        reasoning_events: 0,
        tool_call_events: 0,
        response_shape: None,
        upstream_status: Some(200),
        outcome,
        web_search: crate::obs::WebSearchStats::default(),
    });
}

fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = bytes.len();
    while start < end && (bytes[start] == b' ' || bytes[start] == b'\t' || bytes[start] == b'\r') {
        start += 1;
    }
    while end > start
        && (bytes[end - 1] == b' ' || bytes[end - 1] == b'\t' || bytes[end - 1] == b'\r')
    {
        end -= 1;
    }
    &bytes[start..end]
}
