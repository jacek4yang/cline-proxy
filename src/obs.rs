//! Adaptive bounded observability (issue #16).
//!
//! One normal request produces exactly ONE `RequestSummary`. Detailed
//! lifecycle events live only in a request-local flight recorder; the
//! recorder is attached to the summary when the request was anomalous and
//! dropped otherwise. Records cross to the writer through a **bounded**
//! channel with `try_send` — a full queue drops the record and counts it
//! instead of ever backpressuring a request. A single dedicated
//! `std::thread` owns all disk IO (JSON serialization, buffered writes,
//! rotation, quota); the request path never touches the filesystem and
//! never serializes JSON for logging.
//!
//! Privacy: summaries carry names, counts, sizes, timings, hashes, and
//! fingerprints only — never prompts, reasoning, tool output, raw session
//! ids, or credentials.
//!
//! Failure policy: logging fails open. Any writer/disk error degrades file
//! logging to disabled with a rate-limited console warning; the proxy is
//! unaffected.

use std::collections::VecDeque;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, OnceLock};
use std::time::Duration;

use crate::console::{format_duration_ms, format_tokens};

/// JSONL schema for `RequestSummary`. Bump when field meaning changes.
pub const SCHEMA_VERSION: u32 = 2;

/// Approximate upper bound of one summary's serialized size, used to
/// document queue memory: 8192 × ~1 KiB ≈ 8 MiB worst case, typically
/// far less (most fields are small integers).
pub const QUEUE_CAPACITY: usize = 8192;

// --- flight recorder -------------------------------------------------------

/// One lifecycle event. Numeric/enum only — no strings beyond the variant.
#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FlightEventKind {
    Accepted,
    SystemNormalized,
    ShadowRestored,
    Optimized,
    ExactCountSkippedBusy,
    UpstreamSelected,
    UpstreamHeaders,
    AggregationStarted,
    FirstUpstreamEvent,
    FirstReasoning,
    FirstText,
    FirstToolCall,
    UsageObserved,
    StreamCommitted,
    UpstreamInterrupted,
    ProtocolError,
    Failover,
    ClientDisconnected,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct FlightEvent {
    /// Milliseconds since request start (u32 wraps at 49 days).
    pub at_ms: u32,
    pub kind: FlightEventKind,
}

/// Request-local, fixed-capacity flight recorder. Recording past the cap
/// silently stops (the earliest events are the interesting ones).
#[derive(Default)]
pub struct FlightRecorder {
    events: Vec<FlightEvent>,
    dropped: u8,
}

impl FlightRecorder {
    pub fn new() -> Self {
        Self {
            events: Vec::with_capacity(16),
            dropped: 0,
        }
    }

    pub fn record(&mut self, at_ms: u128, kind: FlightEventKind) {
        if self.events.len() >= 32 {
            self.dropped = self.dropped.saturating_add(1);
            return;
        }
        self.events.push(FlightEvent {
            at_ms: u32::try_from(at_ms).unwrap_or(u32::MAX),
            kind,
        });
    }

    pub fn into_parts(self) -> (Option<Vec<FlightEvent>>, u8) {
        let events = if self.events.is_empty() {
            None
        } else {
            Some(self.events)
        };
        (events, self.dropped)
    }
}

// --- request summary -------------------------------------------------------

/// Process-wide JSONL instance id (one UUID per process).
pub fn process_instance_id() -> &'static str {
    static ID: OnceLock<String> = OnceLock::new();
    ID.get_or_init(|| uuid::Uuid::new_v4().simple().to_string())
        .as_str()
}

/// Local GLM tokenizer result. Distinct from upstream-billed usage.
#[derive(Debug, Clone, Copy)]
pub struct LocalTokenCount {
    pub tokens: u64,
    pub method: &'static str,
    pub duration_ms: u64,
}

/// Shared slot so a background exact count can land before the summary emits.
#[derive(Clone, Default)]
pub struct LocalTokenSlot {
    inner: Arc<std::sync::Mutex<Option<LocalTokenCount>>>,
}

impl LocalTokenSlot {
    pub fn store(&self, count: LocalTokenCount) {
        if let Ok(mut guard) = self.inner.lock() {
            *guard = Some(count);
        }
    }

    pub fn get(&self) -> Option<LocalTokenCount> {
        self.inner.lock().ok().and_then(|guard| *guard)
    }
}

/// The single record emitted per completed request. `None` fields mean
/// "not applicable / upstream did not report" — never zero-padding.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RequestSummary {
    pub schema_version: u32,
    pub instance_id: String,
    pub ts_unix_ms: u64,
    pub request_id: String,
    pub protocol: &'static str,
    pub requested_model: String,
    pub upstream_model: String,
    pub model_family: &'static str,
    /// 16-hex HMAC fingerprint, or `"unstable"` when no session identity.
    pub session: Option<String>,
    pub downstream_stream: bool,
    pub upstream_strategy: &'static str,
    pub selected_key_name: String,
    pub attempts: u64,
    pub failover_count: u64,
    pub reasoning_effort: &'static str,
    pub thinking_exposure: &'static str,
    pub client_max_tokens: Option<u64>,
    pub effective_max_tokens: Option<u64>,
    pub request_bytes: usize,
    pub upstream_request_bytes: usize,
    pub system_bytes: usize,
    pub messages_bytes: usize,
    pub tools_bytes: usize,
    pub historical_reasoning_bytes_removed: u64,
    pub billing_header_bytes_removed: u64,
    pub canonicalized_arguments: usize,
    pub local_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_token_count_method: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_token_count_duration_ms: Option<u64>,
    pub reserved_output_tokens: Option<u64>,
    pub total_context_budget: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_limit_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_utilization_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_headroom_tokens: Option<i64>,
    /// Upstream-billed prompt tokens when the provider reported usage.
    pub prompt_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    /// cached_tokens / prompt_tokens, percent with one decimal (subset
    /// semantics verified against live Cline traffic; issue #8).
    pub cache_hit_ratio: Option<f64>,
    pub reasoning_ratio: Option<f64>,
    /// First semantic (reasoning/text/tool) output. Not a role-only frame.
    pub ttft_ms: Option<u64>,
    pub first_reasoning_ms: Option<u64>,
    pub first_text_ms: Option<u64>,
    pub first_tool_call_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_semantic_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_sse_event_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_upstream_byte_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream_headers_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_upstream_progress_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_semantic_progress_ms: Option<u64>,
    pub duration_ms: u64,
    pub upstream_first_event_ms: Option<u64>,
    pub upstream_duration_ms: Option<u64>,
    pub text_bytes: u64,
    pub reasoning_bytes: u64,
    pub tool_call_bytes: u64,
    pub text_events: u64,
    pub reasoning_events: u64,
    pub tool_call_events: u64,
    pub response_shape: Option<&'static str>,
    pub upstream_status: Option<u16>,
    pub outcome: &'static str,
    /// Present only for anomalous requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flight: Option<Vec<FlightEvent>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flight_dropped: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<&'static str>,
}

/// Everything that can go through the log queue. Kept small: the anomaly
/// variant is boxed so a rare large trace cannot inflate every record.
#[derive(Debug)]
pub enum LogRecord {
    Request(Box<RequestSummary>),
    /// Low-frequency runtime line (startup, degraded logging, metrics).
    Runtime {
        ts_unix_ms: u64,
        message: String,
        fields: Vec<(&'static str, String)>,
    },
}

// --- bounded sink ----------------------------------------------------------

/// Cloneable handle to the bounded log queue. `emit` never blocks and
/// never allocates beyond the boxed record.
#[derive(Clone)]
pub struct LogSink {
    tx: mpsc::SyncSender<LogRecord>,
    pub dropped: Arc<AtomicU64>,
    pub emitted: Arc<AtomicU64>,
}

impl LogSink {
    pub fn emit(&self, record: LogRecord) {
        match self.tx.try_send(record) {
            Ok(()) => {
                self.emitted.fetch_add(1, Ordering::Relaxed);
            }
            Err(mpsc::TrySendError::Full(_)) => {
                // Drop quietly; the writer reports the counter periodically.
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn dropped_total(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

// --- file rotation & quota -------------------------------------------------

#[derive(Debug)]
struct SegmentMeta {
    path: PathBuf,
    bytes: u64,
}

/// Rolling JSONL writer with a hard directory quota. Segments:
/// `events-000001.jsonl`, … — rotated at `max_file_size_mb`, oldest
/// deleted when total bytes exceed the quota, down to the cleanup
/// watermark. Retention state is maintained in memory after one startup
/// scan (no per-record directory IO).
#[derive(Debug)]
pub struct FileWriter {
    directory: PathBuf,
    max_file_bytes: u64,
    max_total_bytes: u64,
    cleanup_target_bytes: u64,
    segments: VecDeque<SegmentMeta>,
    total_bytes: u64,
    writer: Option<BufWriter<std::fs::File>>,
    active_path: Option<PathBuf>,
    /// Set when the writer entered degraded mode (fail-open).
    degraded: bool,
    last_degraded_log: std::time::Instant,
}

impl FileWriter {
    /// One startup scan; missing directory is created, not an error.
    fn scan(directory: &PathBuf, max_file_bytes: u64) -> std::io::Result<Self> {
        std::fs::create_dir_all(directory)?;
        let mut segments: Vec<SegmentMeta> = Vec::new();
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                let bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
                segments.push(SegmentMeta { path, bytes });
            }
        }
        // Oldest first: name suffix order == creation order.
        segments.sort_by(|a, b| a.path.file_name().cmp(&b.path.file_name()));
        let total_bytes = segments.iter().map(|segment| segment.bytes).sum();
        Ok(Self {
            directory: directory.clone(),
            max_file_bytes,
            max_total_bytes: 0, // set by caller
            cleanup_target_bytes: 0,
            segments: segments.into(),
            total_bytes,
            writer: None,
            active_path: None,
            degraded: false,
            last_degraded_log: std::time::Instant::now(),
        })
    }

    fn next_segment_index(&self) -> u32 {
        self.segments
            .iter()
            .filter_map(|segment| {
                segment
                    .path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .and_then(|stem| stem.strip_prefix("events-"))
                    .and_then(|suffix| suffix.parse::<u32>().ok())
            })
            .max()
            .map_or(1, |last| last.saturating_add(1).max(1))
    }

    /// Always create a new exclusive segment. Never append to a file left
    /// by a previous process (underfilled, full, or corrupt last line).
    fn open_fresh_segment(&mut self) -> std::io::Result<()> {
        let mut next_index = self.next_segment_index();
        loop {
            let path = self.directory.join(format!("events-{next_index:06}.jsonl"));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => {
                    self.segments.push_back(SegmentMeta {
                        bytes: 0,
                        path: path.clone(),
                    });
                    self.writer = Some(BufWriter::with_capacity(512 * 1024, file));
                    self.active_path = Some(path);
                    return Ok(());
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    next_index = next_index.saturating_add(1);
                    if next_index == 0 {
                        return Err(std::io::Error::other("log segment index overflow"));
                    }
                    continue;
                }
                Err(error) => return Err(error),
            }
        }
    }

    fn rotate_if_needed(&mut self) -> std::io::Result<()> {
        let active_full = self.active_path.as_ref().is_some_and(|active| {
            self.segments
                .iter()
                .find(|segment| &segment.path == active)
                .is_some_and(|segment| segment.bytes >= self.max_file_bytes)
        });
        if self.writer.is_none() || active_full {
            // Flush + drop the writer BEFORE opening the next file
            // (Windows keeps open handles locked).
            self.writer = None;
            self.active_path = None;
            self.open_fresh_segment()?;
        }
        Ok(())
    }

    fn enforce_quota(&mut self) {
        while self.total_bytes > self.max_total_bytes {
            let Some(oldest) = self.segments.front() else {
                break;
            };
            // Never delete the active segment.
            if self.active_path.as_ref() == Some(&oldest.path) {
                break;
            }
            let Some(oldest) = self.segments.pop_front() else {
                break;
            };
            self.total_bytes = self.total_bytes.saturating_sub(oldest.bytes);
            if std::fs::remove_file(&oldest.path).is_err() {
                // Accounting already moved on; periodic reconciliation
                // will correct drift.
            }
            if self.total_bytes <= self.cleanup_target_bytes {
                break;
            }
        }
    }

    fn write_line(&mut self, bytes: &[u8]) {
        if self.degraded {
            return;
        }
        let result = (|| -> std::io::Result<()> {
            self.rotate_if_needed()?;
            let writer = self
                .writer
                .as_mut()
                .ok_or_else(|| std::io::Error::other("writer closed"))?;
            writer.write_all(bytes)?;
            writer.write_all(b"\n")?;
            if let Some(segment) = self.segments.back_mut() {
                segment.bytes = segment.bytes.saturating_add(bytes.len() as u64 + 1);
            }
            self.total_bytes = self.total_bytes.saturating_add(bytes.len() as u64 + 1);
            self.enforce_quota();
            Ok(())
        })();
        if let Err(error) = result {
            // Fail open: stop file logging, warn at most once a minute.
            self.degraded = true;
            if self.last_degraded_log.elapsed() > Duration::from_secs(60) {
                self.last_degraded_log = std::time::Instant::now();
                tracing::warn!(
                    error = %error,
                    "file logging disabled after write failure; proxy continues"
                );
            }
        }
    }

    fn flush(&mut self) {
        if let Some(writer) = self.writer.as_mut() {
            let _ = writer.flush(); // best-effort; buffered data may be lost
        }
    }
}

// --- summary builder -------------------------------------------------------

use std::time::Instant;

/// Builds and emits exactly ONE `RequestSummary` per completed request.
/// Decoupled from the HTTP layer: the caller supplies the policy/routing
/// facts it collected; this composes, detects anomalies adaptively, emits
/// the compact console line, and pushes to the bounded sink.
pub struct SummaryBuilder {
    ts_unix_ms: u64,
    pub request_id: String,
    protocol: &'static str,
    requested_model: String,
    upstream_model: String,
    session: Option<String>,
    downstream_stream: bool,
    upstream_strategy: &'static str,
    started: Instant,
    flight: FlightRecorder,
    anomaly: bool,
    error_kind: Option<&'static str>,
    /// Adaptive thresholds, read from config at construction.
    slow_ttft_ms: u64,
    slow_duration_ms: u64,
    model_family: &'static str,
}

impl SummaryBuilder {
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::fn_params_excessive_bools)]
    pub fn new(
        request_id: &str,
        protocol: &'static str,
        requested_model: String,
        upstream_model: String,
        model_family: &'static str,
        session: Option<String>,
        downstream_stream: bool,
        upstream_strategy: &'static str,
        started: Instant,
        slow_ttft_ms: u64,
        slow_duration_ms: u64,
    ) -> Self {
        Self {
            ts_unix_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            request_id: request_id.to_owned(),
            protocol,
            requested_model,
            upstream_model,
            model_family,
            session,
            downstream_stream,
            upstream_strategy,
            started,
            flight: FlightRecorder::new(),
            anomaly: false,
            error_kind: None,
            slow_ttft_ms,
            slow_duration_ms,
        }
    }

    pub fn record(&mut self, kind: FlightEventKind) {
        self.flight.record(self.started.elapsed().as_millis(), kind);
    }

    pub fn mark_anomalous(&mut self, error_kind: &'static str) {
        self.anomaly = true;
        self.error_kind = Some(error_kind);
    }

    /// Compose the summary from everything learned during the request and
    /// emit it. `usage` is the upstream OpenAI usage value (may be partial).
    pub fn emit(mut self, facts: SummaryEmit<'_>) {
        let extract = |usage: Option<&serde_json::Value>, path: &[&str]| -> Option<u64> {
            usage
                .and_then(|usage| value_at(usage, path))
                .and_then(serde_json::Value::as_u64)
        };
        let prompt_tokens = extract(facts.usage, &["prompt_tokens"])
            .or_else(|| extract(facts.usage, &["input_tokens"]));
        let cached_tokens = extract(facts.usage, &["prompt_tokens_details", "cached_tokens"])
            .or_else(|| extract(facts.usage, &["cache_read_input_tokens"]));
        let completion_tokens = extract(facts.usage, &["completion_tokens"])
            .or_else(|| extract(facts.usage, &["output_tokens"]));
        let reasoning_tokens = extract(
            facts.usage,
            &["completion_tokens_details", "reasoning_tokens"],
        );
        let cache_hit_ratio = match (cached_tokens, prompt_tokens) {
            (Some(cached), Some(prompt)) if prompt > 0 => {
                Some((cached.min(prompt) as f64 / prompt as f64 * 1000.0).round() / 10.0)
            }
            _ => None,
        };
        let reasoning_ratio = match (reasoning_tokens, completion_tokens) {
            (Some(reasoning), Some(completion)) if completion > 0 => {
                Some((reasoning.min(completion) as f64 / completion as f64 * 1000.0).round() / 10.0)
            }
            _ => None,
        };
        let first_semantic_ms = facts
            .first_semantic_ms
            .or(facts.first_reasoning_ms)
            .or(facts.first_text_ms)
            .or(facts.first_tool_call_ms);
        let ttft_ms = first_semantic_ms;
        let elapsed = self.started.elapsed();
        if elapsed.as_millis() > u128::from(self.slow_duration_ms) {
            self.mark_anomalous("slow_duration");
        } else if ttft_ms.is_some_and(|ttft| ttft > u128::from(self.slow_ttft_ms)) {
            self.mark_anomalous("slow_ttft");
        }
        let (flight, flight_dropped) = if self.anomaly {
            let (events, dropped) = self.flight.into_parts();
            (events, Some(dropped))
        } else {
            (None, None)
        };
        let local_input_tokens = facts.local_tokens.map(|count| count.tokens);
        let reserved_output_tokens = facts.effective_max_tokens;
        let total_context_budget = match (local_input_tokens, reserved_output_tokens) {
            (Some(input), Some(output)) => Some(input.saturating_add(output)),
            _ => None,
        };
        let (context_utilization_ratio, context_headroom_tokens) =
            match (total_context_budget, facts.context_limit_tokens) {
                (Some(budget), Some(limit)) if limit > 0 => {
                    let ratio = (budget as f64 / limit as f64 * 1000.0).round() / 10.0;
                    let headroom = limit as i64 - budget as i64;
                    (Some(ratio), Some(headroom))
                }
                _ => (None, None),
            };
        let summary = RequestSummary {
            schema_version: SCHEMA_VERSION,
            instance_id: process_instance_id().to_owned(),
            ts_unix_ms: self.ts_unix_ms,
            request_id: self.request_id.clone(),
            protocol: self.protocol,
            requested_model: self.requested_model.clone(),
            upstream_model: self.upstream_model.clone(),
            model_family: self.model_family,
            session: self.session.clone(),
            downstream_stream: self.downstream_stream,
            upstream_strategy: self.upstream_strategy,
            selected_key_name: facts.selected_key_name.to_owned(),
            attempts: facts.attempts,
            failover_count: facts.failover_count,
            reasoning_effort: facts.reasoning_effort,
            thinking_exposure: if facts.expose_thinking {
                "exposed"
            } else {
                "suppressed"
            },
            client_max_tokens: facts.client_max_tokens,
            effective_max_tokens: facts.effective_max_tokens,
            request_bytes: facts.request_bytes,
            upstream_request_bytes: facts.upstream_request_bytes,
            system_bytes: facts.system_bytes,
            messages_bytes: facts.messages_bytes,
            tools_bytes: facts.tools_bytes,
            historical_reasoning_bytes_removed: facts.historical_reasoning_bytes_removed,
            billing_header_bytes_removed: facts.billing_header_bytes_removed,
            canonicalized_arguments: facts.canonicalized_arguments,
            local_input_tokens,
            local_token_count_method: facts.local_tokens.map(|count| count.method),
            local_token_count_duration_ms: facts.local_tokens.map(|count| count.duration_ms),
            reserved_output_tokens,
            total_context_budget,
            context_limit_tokens: facts.context_limit_tokens,
            context_utilization_ratio,
            context_headroom_tokens,
            prompt_tokens,
            cached_tokens,
            completion_tokens,
            reasoning_tokens,
            cache_hit_ratio,
            reasoning_ratio,
            ttft_ms: cap_ms(ttft_ms),
            first_reasoning_ms: cap_ms(facts.first_reasoning_ms),
            first_text_ms: cap_ms(facts.first_text_ms),
            first_tool_call_ms: cap_ms(facts.first_tool_call_ms),
            first_semantic_ms: cap_ms(first_semantic_ms),
            first_sse_event_ms: cap_ms(facts.first_sse_event_ms),
            first_upstream_byte_ms: cap_ms(facts.first_upstream_byte_ms),
            upstream_headers_ms: cap_ms(facts.upstream_headers_ms),
            last_upstream_progress_ms: cap_ms(facts.last_upstream_progress_ms),
            last_semantic_progress_ms: cap_ms(facts.last_semantic_progress_ms),
            duration_ms: elapsed.as_millis().min(u64::MAX as u128) as u64,
            upstream_first_event_ms: cap_ms(
                facts.first_sse_event_ms.or(facts.upstream_first_event_ms),
            ),
            upstream_duration_ms: cap_ms(facts.upstream_duration_ms),
            text_bytes: facts.text_bytes,
            reasoning_bytes: facts.reasoning_bytes,
            tool_call_bytes: facts.tool_call_bytes,
            text_events: facts.text_events,
            reasoning_events: facts.reasoning_events,
            tool_call_events: facts.tool_call_events,
            response_shape: facts.response_shape,
            upstream_status: facts.upstream_status,
            outcome: facts.outcome,
            flight,
            flight_dropped,
            error_kind: self.error_kind,
        };
        let line = compact_request_line(&summary, self.session.as_deref());
        tracing::info!("{line}");
        if let Some(sink) = facts.sink {
            sink.emit(LogRecord::Request(Box::new(summary)));
        }
    }
}

fn cap_ms(value: Option<u128>) -> Option<u64> {
    value.map(|value| value.min(u64::MAX as u128) as u64)
}

fn compact_request_line(summary: &RequestSummary, session: Option<&str>) -> String {
    let symbol = match summary.outcome {
        "complete" => "\u{2713}",
        _ => "\u{2717}",
    };
    let family = if summary.model_family == "glm53" {
        "GLM53"
    } else {
        "GENERIC"
    };
    let input = summary
        .local_input_tokens
        .or(summary.prompt_tokens)
        .map(format_tokens)
        .unwrap_or_else(|| "?".to_string());
    let budget = summary
        .total_context_budget
        .map(|tokens| format!(" budget={}", format_tokens(tokens)))
        .unwrap_or_default();
    let cache = summary
        .cache_hit_ratio
        .map(|ratio| format!("{ratio}%"))
        .unwrap_or_else(|| "n/a".to_string());
    let out = match summary.completion_tokens {
        Some(tokens) => tokens.to_string(),
        None if summary.text_events + summary.reasoning_events + summary.tool_call_events > 0 => {
            format!(
                "t{}+r{}+k{}",
                summary.text_events, summary.reasoning_events, summary.tool_call_events
            )
        }
        None => "?".to_string(),
    };
    let ttft = summary
        .ttft_ms
        .map(format_duration_ms)
        .unwrap_or_else(|| "-".to_string());
    let tool = summary
        .first_tool_call_ms
        .map(|ms| format!(" tool={}", format_duration_ms(ms)))
        .unwrap_or_default();
    let err = summary
        .error_kind
        .map(|kind| format!(" err={kind}"))
        .unwrap_or_default();
    format!(
        "{symbol} {family} agent={} key={} in={input}{budget} cache={cache} out={out} ttft={ttft}{tool} dur={}{err}",
        session.unwrap_or("-"),
        summary.selected_key_name,
        format_duration_ms(summary.duration_ms),
    )
}

/// Facts supplied at summary emission. Named so new diagnostic fields do
/// not explode positional argument lists.
pub struct SummaryEmit<'a> {
    pub sink: Option<&'a LogSink>,
    pub selected_key_name: &'a str,
    pub attempts: u64,
    pub failover_count: u64,
    pub reasoning_effort: &'static str,
    pub expose_thinking: bool,
    pub client_max_tokens: Option<u64>,
    pub effective_max_tokens: Option<u64>,
    pub request_bytes: usize,
    pub upstream_request_bytes: usize,
    pub system_bytes: usize,
    pub messages_bytes: usize,
    pub tools_bytes: usize,
    pub historical_reasoning_bytes_removed: u64,
    pub billing_header_bytes_removed: u64,
    pub canonicalized_arguments: usize,
    pub usage: Option<&'a serde_json::Value>,
    pub local_tokens: Option<LocalTokenCount>,
    pub context_limit_tokens: Option<u64>,
    pub first_reasoning_ms: Option<u128>,
    pub first_text_ms: Option<u128>,
    pub first_tool_call_ms: Option<u128>,
    pub first_semantic_ms: Option<u128>,
    pub first_sse_event_ms: Option<u128>,
    pub first_upstream_byte_ms: Option<u128>,
    pub upstream_headers_ms: Option<u128>,
    pub last_upstream_progress_ms: Option<u128>,
    pub last_semantic_progress_ms: Option<u128>,
    pub upstream_first_event_ms: Option<u128>,
    pub upstream_duration_ms: Option<u128>,
    pub text_bytes: u64,
    pub reasoning_bytes: u64,
    pub tool_call_bytes: u64,
    pub text_events: u64,
    pub reasoning_events: u64,
    pub tool_call_events: u64,
    pub response_shape: Option<&'static str>,
    pub upstream_status: Option<u16>,
    pub outcome: &'static str,
}

/// Walk a path into a JSON value (tiny helper for usage fields).
fn value_at<'a>(value: &'a serde_json::Value, path: &[&str]) -> Option<&'a serde_json::Value> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
}

/// Static (handler-known) summary context for a streaming request. The
/// stream owns it; at close it feeds the dynamic snapshot to a
/// [`SummaryBuilder`]. Cloned once per request.
#[derive(Clone)]
pub struct StreamSummary {
    pub sink: Option<LogSink>,
    pub session: Option<String>,
    pub requested_model: String,
    pub upstream_model: String,
    pub model_family: &'static str,
    pub downstream_stream: bool,
    pub upstream_strategy: &'static str,
    pub started: Instant,
    pub slow_ttft_ms: u64,
    pub slow_duration_ms: u64,
    // Policy facts from the optimization step.
    pub reasoning_effort: &'static str,
    pub expose_thinking: bool,
    pub client_max_tokens: Option<u64>,
    pub effective_max_tokens: Option<u64>,
    pub request_bytes: usize,
    pub upstream_request_bytes: usize,
    pub system_bytes: usize,
    pub messages_bytes: usize,
    pub tools_bytes: usize,
    pub historical_reasoning_bytes_removed: u64,
    pub billing_header_bytes_removed: u64,
    pub canonicalized_arguments: usize,
    pub local_tokens: LocalTokenSlot,
    pub context_limit_tokens: Option<u64>,
    pub upstream_headers_ms: Option<u128>,
}

/// Dynamic per-stream data at close.
pub struct StreamSnap<'a> {
    pub request_id: &'a str,
    pub key_name: &'a str,
    pub first_reasoning_ms: Option<u128>,
    pub first_text_ms: Option<u128>,
    pub first_tool_call_ms: Option<u128>,
    pub first_sse_event_ms: Option<u128>,
    pub first_semantic_ms: Option<u128>,
    pub first_upstream_byte_ms: Option<u128>,
    pub last_upstream_progress_ms: Option<u128>,
    pub last_semantic_progress_ms: Option<u128>,
    pub usage: Option<&'a serde_json::Value>,
    pub text_bytes: u64,
    pub reasoning_bytes: u64,
    pub tool_call_bytes: u64,
    pub text_events: u64,
    pub reasoning_events: u64,
    pub tool_call_events: u64,
}

impl StreamSummary {
    pub fn finish(&self, snap: StreamSnap<'_>, outcome: &'static str) {
        let mut builder = SummaryBuilder::new(
            snap.request_id,
            "anthropic",
            self.requested_model.clone(),
            self.upstream_model.clone(),
            self.model_family,
            self.session.clone(),
            self.downstream_stream,
            self.upstream_strategy,
            self.started,
            self.slow_ttft_ms,
            self.slow_duration_ms,
        );
        if outcome != "complete" {
            builder.mark_anomalous(outcome);
        }
        if snap.first_reasoning_ms.is_some() {
            builder.record(FlightEventKind::FirstReasoning);
        }
        if snap.first_text_ms.is_some() {
            builder.record(FlightEventKind::FirstText);
        }
        if snap.first_tool_call_ms.is_some() {
            builder.record(FlightEventKind::FirstToolCall);
        }
        if snap.usage.is_some() {
            builder.record(FlightEventKind::UsageObserved);
        }
        builder.emit(SummaryEmit {
            sink: self.sink.as_ref(),
            selected_key_name: snap.key_name,
            attempts: 1,
            failover_count: 0,
            reasoning_effort: self.reasoning_effort,
            expose_thinking: self.expose_thinking,
            client_max_tokens: self.client_max_tokens,
            effective_max_tokens: self.effective_max_tokens,
            request_bytes: self.request_bytes,
            upstream_request_bytes: self.upstream_request_bytes,
            system_bytes: self.system_bytes,
            messages_bytes: self.messages_bytes,
            tools_bytes: self.tools_bytes,
            historical_reasoning_bytes_removed: self.historical_reasoning_bytes_removed,
            billing_header_bytes_removed: self.billing_header_bytes_removed,
            canonicalized_arguments: self.canonicalized_arguments,
            usage: snap.usage,
            local_tokens: self.local_tokens.get(),
            context_limit_tokens: self.context_limit_tokens,
            first_reasoning_ms: snap.first_reasoning_ms,
            first_text_ms: snap.first_text_ms,
            first_tool_call_ms: snap.first_tool_call_ms,
            first_semantic_ms: snap.first_semantic_ms,
            first_sse_event_ms: snap.first_sse_event_ms,
            first_upstream_byte_ms: snap.first_upstream_byte_ms,
            upstream_headers_ms: self.upstream_headers_ms,
            last_upstream_progress_ms: snap.last_upstream_progress_ms,
            last_semantic_progress_ms: snap.last_semantic_progress_ms,
            upstream_first_event_ms: snap.first_sse_event_ms,
            upstream_duration_ms: Some(self.started.elapsed().as_millis()),
            text_bytes: snap.text_bytes,
            reasoning_bytes: snap.reasoning_bytes,
            tool_call_bytes: snap.tool_call_bytes,
            text_events: snap.text_events,
            reasoning_events: snap.reasoning_events,
            tool_call_events: snap.tool_call_events,
            response_shape: None,
            upstream_status: Some(200),
            outcome,
        });
    }

    /// Emit a summary when the stream never started (transport / pre-body).
    #[allow(clippy::too_many_arguments)]
    pub fn emit_outcome(
        &self,
        request_id: &str,
        key_name: &str,
        attempts: u64,
        failover_count: u64,
        usage: Option<&serde_json::Value>,
        outcome: &'static str,
        error_kind: Option<&'static str>,
        timings: OutcomeTimings,
    ) {
        let mut builder = SummaryBuilder::new(
            request_id,
            "anthropic",
            self.requested_model.clone(),
            self.upstream_model.clone(),
            self.model_family,
            self.session.clone(),
            self.downstream_stream,
            self.upstream_strategy,
            self.started,
            self.slow_ttft_ms,
            self.slow_duration_ms,
        );
        if let Some(kind) = error_kind {
            builder.mark_anomalous(kind);
        } else if outcome != "complete" {
            builder.mark_anomalous(outcome);
        }
        builder.emit(SummaryEmit {
            sink: self.sink.as_ref(),
            selected_key_name: key_name,
            attempts,
            failover_count,
            reasoning_effort: self.reasoning_effort,
            expose_thinking: self.expose_thinking,
            client_max_tokens: self.client_max_tokens,
            effective_max_tokens: self.effective_max_tokens,
            request_bytes: self.request_bytes,
            upstream_request_bytes: self.upstream_request_bytes,
            system_bytes: self.system_bytes,
            messages_bytes: self.messages_bytes,
            tools_bytes: self.tools_bytes,
            historical_reasoning_bytes_removed: self.historical_reasoning_bytes_removed,
            billing_header_bytes_removed: self.billing_header_bytes_removed,
            canonicalized_arguments: self.canonicalized_arguments,
            usage,
            local_tokens: self.local_tokens.get(),
            context_limit_tokens: self.context_limit_tokens,
            first_reasoning_ms: timings.first_reasoning_ms,
            first_text_ms: timings.first_text_ms,
            first_tool_call_ms: timings.first_tool_call_ms,
            first_semantic_ms: timings.first_semantic_ms,
            first_sse_event_ms: timings.first_sse_event_ms,
            first_upstream_byte_ms: timings.first_upstream_byte_ms,
            upstream_headers_ms: self.upstream_headers_ms,
            last_upstream_progress_ms: timings.last_upstream_progress_ms,
            last_semantic_progress_ms: timings.last_semantic_progress_ms,
            upstream_first_event_ms: timings.first_sse_event_ms,
            upstream_duration_ms: timings.upstream_duration_ms,
            text_bytes: timings.text_bytes,
            reasoning_bytes: timings.reasoning_bytes,
            tool_call_bytes: timings.tool_call_bytes,
            text_events: timings.text_events,
            reasoning_events: timings.reasoning_events,
            tool_call_events: timings.tool_call_events,
            response_shape: timings.response_shape,
            upstream_status: timings.upstream_status,
            outcome,
        });
    }
}

/// Optional timings for [`StreamSummary::emit_outcome`].
#[derive(Default)]
pub struct OutcomeTimings {
    pub first_reasoning_ms: Option<u128>,
    pub first_text_ms: Option<u128>,
    pub first_tool_call_ms: Option<u128>,
    pub first_semantic_ms: Option<u128>,
    pub first_sse_event_ms: Option<u128>,
    pub first_upstream_byte_ms: Option<u128>,
    pub last_upstream_progress_ms: Option<u128>,
    pub last_semantic_progress_ms: Option<u128>,
    pub upstream_duration_ms: Option<u128>,
    pub text_bytes: u64,
    pub reasoning_bytes: u64,
    pub tool_call_bytes: u64,
    pub text_events: u64,
    pub reasoning_events: u64,
    pub tool_call_events: u64,
    pub response_shape: Option<&'static str>,
    pub upstream_status: Option<u16>,
}

// --- writer thread ---------------------------------------------------------

/// Writer-thread configuration.
#[derive(Debug, Clone)]
pub struct WriterConfig {
    pub directory: PathBuf,
    pub max_file_size_mb: u64,
    pub max_total_size_mb: u64,
    pub cleanup_target_percent: u64,
    pub flush_interval_ms: u64,
}

/// Spawn the dedicated writer thread. Returns the queue handle and a
/// shutdown guard: dropping the guard signals the thread to finish its
/// bounded drain (≤2 s) and exit.
pub fn spawn_writer(
    config: WriterConfig,
) -> std::io::Result<(LogSink, std::thread::JoinHandle<()>)> {
    let (tx, rx) = mpsc::sync_channel::<LogRecord>(QUEUE_CAPACITY);
    let dropped = Arc::new(AtomicU64::new(0));
    let emitted = Arc::new(AtomicU64::new(0));
    let dropped_handle = Arc::clone(&dropped);
    let emitted_handle = Arc::clone(&emitted);

    let mut file_writer = FileWriter::scan(
        &config.directory,
        config.max_file_size_mb.saturating_mul(1024 * 1024),
    )?;
    // Each process owns a fresh active segment. Quota accounting still
    // includes segments discovered above.
    file_writer.open_fresh_segment()?;
    file_writer.max_total_bytes = config.max_total_size_mb.saturating_mul(1024 * 1024);
    file_writer.cleanup_target_bytes = file_writer
        .max_total_bytes
        .saturating_mul(config.cleanup_target_percent.min(100))
        / 100;
    if file_writer.total_bytes > file_writer.max_total_bytes {
        file_writer.enforce_quota();
        tracing::info!(
            directory = %config.directory.display(),
            total_bytes_after_startup_cleanup = file_writer.total_bytes,
            "log quota enforced at startup"
        );
    }

    let flush_interval = Duration::from_millis(config.flush_interval_ms.max(50));
    let handle = std::thread::Builder::new()
        .name("cline-log-writer".into())
        .spawn(move || {
            // Shutdown: the LogSink sender handles dropping triggers the
            // Disconnected branch below for a bounded drain.
            let mut scratch: Vec<u8> = Vec::with_capacity(1024);
            let mut last_dropped_reported = 0u64;
            loop {
                // Bounded wait: periodic flush + dropped-record reporting.
                match rx.recv_timeout(flush_interval) {
                    Ok(record) => {
                        scratch.clear();
                        match &record {
                            LogRecord::Request(summary) => {
                                let _ = serde_json::to_writer(&mut scratch, summary);
                            }
                            LogRecord::Runtime { ts_unix_ms, message, fields } => {
                                use std::io::Write as _;
                                let _ = write!(scratch, "{{\"ts_unix_ms\":{ts_unix_ms},\"message\":");
                                let _ = serde_json::to_writer(&mut scratch, message);
                                let _ = write!(scratch, ",\"fields\":{{");
                                for (index, (key, value)) in fields.iter().enumerate() {
                                    if index > 0 {
                                        let _ = write!(scratch, ",");
                                    }
                                    let _ = serde_json::to_writer(&mut scratch, key);
                                    let _ = write!(scratch, ":");
                                    let _ = serde_json::to_writer(&mut scratch, value);
                                }
                                let _ = write!(scratch, "}}}}");
                            }
                        }
                        file_writer.write_line(&scratch);
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        // Bounded drain on shutdown: take whatever is
                        // queued, then exit.
                        while let Ok(record) = rx.try_recv() {
                            scratch.clear();
                            if let LogRecord::Request(summary) = &record {
                                let _ = serde_json::to_writer(&mut scratch, summary);
                                file_writer.write_line(&scratch);
                            }
                        }
                        break;
                    }
                }
                file_writer.flush();
                let dropped_now = dropped_handle.load(Ordering::Relaxed);
                if dropped_now > last_dropped_reported {
                    last_dropped_reported = dropped_now;
                    tracing::warn!(
                        dropped_log_records = dropped_now,
                        emitted_log_records = emitted_handle.load(Ordering::Relaxed),
                        "log records dropped: queue full (observability degraded, requests unaffected)"
                    );
                }
            }
            file_writer.flush();
        })?;
    Ok((
        LogSink {
            tx,
            dropped,
            emitted,
        },
        handle,
    ))
}

impl LogRecord {
    #[cfg(test)]
    fn request_marker(&self) -> &str {
        match self {
            LogRecord::Request(summary) => &summary.request_id,
            LogRecord::Runtime { .. } => "runtime",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn summary(request_id: &str) -> RequestSummary {
        RequestSummary {
            schema_version: SCHEMA_VERSION,
            instance_id: "test-instance".into(),
            ts_unix_ms: 0,
            request_id: request_id.to_owned(),
            protocol: "anthropic",
            requested_model: "claude-sonnet-4-6".into(),
            upstream_model: "z-ai/glm-5.3-flash".into(),
            model_family: "glm53",
            session: Some("ab12cd34ef56ab12".into()),
            downstream_stream: false,
            upstream_strategy: "stream_and_aggregate",
            selected_key_name: "key-1".into(),
            attempts: 1,
            failover_count: 0,
            reasoning_effort: "high",
            thinking_exposure: "suppressed",
            client_max_tokens: Some(32000),
            effective_max_tokens: Some(16384),
            request_bytes: 188_000,
            upstream_request_bytes: 187_000,
            system_bytes: 90_000,
            messages_bytes: 90_000,
            tools_bytes: 7_000,
            historical_reasoning_bytes_removed: 0,
            billing_header_bytes_removed: 120,
            canonicalized_arguments: 2,
            local_input_tokens: Some(307_678),
            local_token_count_method: Some("exact_glm53_optimized"),
            local_token_count_duration_ms: Some(12),
            reserved_output_tokens: Some(16384),
            total_context_budget: Some(324_062),
            context_limit_tokens: None,
            context_utilization_ratio: None,
            context_headroom_tokens: None,
            prompt_tokens: Some(307_678),
            cached_tokens: Some(307_648),
            completion_tokens: Some(209),
            reasoning_tokens: Some(114),
            cache_hit_ratio: Some(99.9),
            reasoning_ratio: Some(54.5),
            ttft_ms: Some(88_789),
            first_reasoning_ms: Some(88_700),
            first_text_ms: None,
            first_tool_call_ms: Some(88_789),
            first_semantic_ms: Some(88_700),
            first_sse_event_ms: Some(88_650),
            first_upstream_byte_ms: Some(88_640),
            upstream_headers_ms: Some(1_200),
            last_upstream_progress_ms: Some(92_300),
            last_semantic_progress_ms: Some(92_250),
            duration_ms: 92_376,
            upstream_first_event_ms: Some(88_650),
            upstream_duration_ms: Some(92_300),
            text_bytes: 400,
            reasoning_bytes: 800,
            tool_call_bytes: 120,
            text_events: 3,
            reasoning_events: 8,
            tool_call_events: 1,
            response_shape: Some("openai"),
            upstream_status: Some(200),
            outcome: "complete",
            flight: None,
            flight_dropped: None,
            error_kind: None,
        }
    }

    #[test]
    fn summary_serializes_without_content_and_with_expected_fields() {
        let bytes = serde_json::to_vec(&summary("req_1")).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // Sizes/counters only.
        assert!(value["request_bytes"].as_u64().unwrap() > 0);
        assert!(value.get("prompt").is_none());
        assert!(value.get("messages").is_none());
        assert!(value.get("content").is_none());
        // Flight fields omitted when absent.
        assert!(value.get("flight").is_none());
        // Sensible serialized size (bounded queue math).
        assert_eq!(value["schema_version"], SCHEMA_VERSION);
        assert!(value.get("prompt").is_none());
        assert!(
            bytes.len() < 4096,
            "summary serialized to {} bytes",
            bytes.len()
        );
    }

    fn writer_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "cline-proxy-obs-{label}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ))
    }

    fn wait_writer(sink: LogSink, handle: std::thread::JoinHandle<()>) {
        drop(sink);
        let _ = handle.join();
    }

    #[test]
    fn writer_restart_with_underfilled_segment_opens_fresh_file() {
        let directory = writer_dir("underfilled");
        std::fs::create_dir_all(&directory).unwrap();
        let old_path = directory.join("events-000001.jsonl");
        std::fs::write(&old_path, b"{\"old\":true}\n").unwrap();
        let (sink, handle) = spawn_writer(WriterConfig {
            directory: directory.clone(),
            max_file_size_mb: 1,
            max_total_size_mb: 4,
            cleanup_target_percent: 85,
            flush_interval_ms: 50,
        })
        .unwrap();
        sink.emit(LogRecord::Request(Box::new(summary("req_restart"))));
        wait_writer(sink, handle);
        let old = std::fs::read(&old_path).unwrap();
        assert_eq!(old, b"{\"old\":true}\n");
        let names: Vec<_> = std::fs::read_dir(&directory)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(names.iter().any(|name| name != "events-000001.jsonl"));
        let mut found = false;
        for entry in std::fs::read_dir(&directory)
            .unwrap()
            .filter_map(Result::ok)
        {
            if entry.path() == old_path {
                continue;
            }
            let text = std::fs::read_to_string(entry.path()).unwrap();
            for line in text.lines() {
                let value: serde_json::Value = serde_json::from_str(line).unwrap();
                if value["request_id"] == "req_restart" {
                    found = true;
                    assert_eq!(value["schema_version"], SCHEMA_VERSION);
                }
            }
        }
        assert!(found, "new segment missing restarted summary");
        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn writer_restart_with_full_segment_opens_next_index() {
        let directory = writer_dir("full");
        std::fs::create_dir_all(&directory).unwrap();
        let old_path = directory.join("events-000001.jsonl");
        std::fs::write(&old_path, vec![b'x'; 64 * 1024]).unwrap();
        let (sink, handle) = spawn_writer(WriterConfig {
            directory: directory.clone(),
            max_file_size_mb: 1,
            max_total_size_mb: 8,
            cleanup_target_percent: 85,
            flush_interval_ms: 50,
        })
        .unwrap();
        sink.emit(LogRecord::Request(Box::new(summary("req_full"))));
        wait_writer(sink, handle);
        assert_eq!(std::fs::read(&old_path).unwrap().len(), 64 * 1024);
        let new_path = directory.join("events-000002.jsonl");
        let text = std::fs::read_to_string(&new_path).unwrap();
        assert!(text.contains("req_full"));
        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn writer_restart_after_partial_final_line_does_not_truncate_old() {
        let directory = writer_dir("partial");
        std::fs::create_dir_all(&directory).unwrap();
        let old_path = directory.join("events-000001.jsonl");
        std::fs::write(&old_path, b"{\"old\":true}\n{\"partial").unwrap();
        let (sink, handle) = spawn_writer(WriterConfig {
            directory: directory.clone(),
            max_file_size_mb: 1,
            max_total_size_mb: 4,
            cleanup_target_percent: 85,
            flush_interval_ms: 50,
        })
        .unwrap();
        sink.emit(LogRecord::Request(Box::new(summary("req_partial"))));
        wait_writer(sink, handle);
        let old = std::fs::read(&old_path).unwrap();
        assert_eq!(old, b"{\"old\":true}\n{\"partial");
        let new_text = std::fs::read_to_string(directory.join("events-000002.jsonl")).unwrap();
        for line in new_text.lines() {
            let _: serde_json::Value = serde_json::from_str(line).unwrap();
        }
        assert!(new_text.contains("req_partial"));
        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn compact_info_line_uses_local_tokens_and_not_question_mark() {
        let mut record = summary("req_fail");
        record.prompt_tokens = None;
        record.completion_tokens = None;
        record.local_input_tokens = Some(199_634);
        record.total_context_budget = Some(216_018);
        record.outcome = "upstream_error";
        record.error_kind = Some("upstream_stream_idle_timeout");
        record.text_events = 2;
        record.reasoning_events = 4;
        record.tool_call_events = 0;
        let line = compact_request_line(&record, Some("abcd"));
        assert!(line.contains("in=199.6K"), "{line}");
        assert!(line.contains("budget=216.0K"), "{line}");
        assert!(line.contains("out=t2+r4+k0"), "{line}");
        assert!(!line.contains("in=?"), "{line}");
        assert!(!line.contains("out=0"), "{line}");
        assert!(line.as_bytes().iter().all(|b| *b != 0x1b));
    }

    #[test]
    fn local_exact_tokens_survive_missing_upstream_usage() {
        let mut builder = SummaryBuilder::new(
            "req",
            "anthropic",
            "claude-sonnet-4-6".into(),
            "z-ai/glm-5.3-flash".into(),
            "glm53",
            None,
            true,
            "native_streaming",
            Instant::now(),
            15_000,
            60_000,
        );
        builder.mark_anomalous("upstream_error");
        let (tx, rx) = mpsc::sync_channel(4);
        let sink = LogSink {
            tx,
            dropped: Arc::new(AtomicU64::new(0)),
            emitted: Arc::new(AtomicU64::new(0)),
        };
        builder.emit(SummaryEmit {
            sink: Some(&sink),
            selected_key_name: "k",
            attempts: 1,
            failover_count: 0,
            reasoning_effort: "high",
            expose_thinking: false,
            client_max_tokens: Some(32_000),
            effective_max_tokens: Some(16_384),
            request_bytes: 10,
            upstream_request_bytes: 10,
            system_bytes: 1,
            messages_bytes: 1,
            tools_bytes: 0,
            historical_reasoning_bytes_removed: 0,
            billing_header_bytes_removed: 0,
            canonicalized_arguments: 0,
            usage: None,
            local_tokens: Some(LocalTokenCount {
                tokens: 199_634,
                method: "exact_glm53_optimized",
                duration_ms: 40,
            }),
            context_limit_tokens: None,
            first_reasoning_ms: None,
            first_text_ms: None,
            first_tool_call_ms: None,
            first_semantic_ms: None,
            first_sse_event_ms: Some(13_100),
            first_upstream_byte_ms: Some(13_050),
            upstream_headers_ms: Some(400),
            last_upstream_progress_ms: Some(13_100),
            last_semantic_progress_ms: None,
            upstream_first_event_ms: Some(13_100),
            upstream_duration_ms: Some(603_500),
            text_bytes: 0,
            reasoning_bytes: 0,
            tool_call_bytes: 0,
            text_events: 0,
            reasoning_events: 0,
            tool_call_events: 0,
            response_shape: None,
            upstream_status: Some(200),
            outcome: "upstream_error",
        });
        let LogRecord::Request(summary) = rx.try_recv().unwrap() else {
            panic!("expected request record");
        };
        assert_eq!(summary.local_input_tokens, Some(199_634));
        assert_eq!(summary.prompt_tokens, None);
        assert_eq!(summary.total_context_budget, Some(199_634 + 16_384));
        assert_eq!(summary.first_sse_event_ms, Some(13_100));
        assert_eq!(summary.ttft_ms, None);
        assert_eq!(summary.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn anthropic_usage_fields_are_not_misread_as_missing() {
        let usage = json!({"input_tokens":19,"output_tokens":14,"cache_read_input_tokens":0});
        let builder = SummaryBuilder::new(
            "req",
            "anthropic",
            "claude-sonnet-4-6".into(),
            "z-ai/glm-5.3-flash".into(),
            "glm53",
            None,
            false,
            "stream_and_aggregate",
            Instant::now(),
            15_000,
            60_000,
        );
        let (tx, rx) = mpsc::sync_channel(4);
        let sink = LogSink {
            tx,
            dropped: Arc::new(AtomicU64::new(0)),
            emitted: Arc::new(AtomicU64::new(0)),
        };
        builder.emit(SummaryEmit {
            sink: Some(&sink),
            selected_key_name: "k",
            attempts: 1,
            failover_count: 0,
            reasoning_effort: "high",
            expose_thinking: false,
            client_max_tokens: Some(16),
            effective_max_tokens: Some(16),
            request_bytes: 10,
            upstream_request_bytes: 10,
            system_bytes: 0,
            messages_bytes: 10,
            tools_bytes: 0,
            historical_reasoning_bytes_removed: 0,
            billing_header_bytes_removed: 0,
            canonicalized_arguments: 0,
            usage: Some(&usage),
            local_tokens: Some(LocalTokenCount {
                tokens: 19,
                method: "exact_glm53_optimized",
                duration_ms: 2,
            }),
            context_limit_tokens: None,
            first_reasoning_ms: None,
            first_text_ms: Some(100),
            first_tool_call_ms: None,
            first_semantic_ms: Some(100),
            first_sse_event_ms: Some(90),
            first_upstream_byte_ms: Some(80),
            upstream_headers_ms: Some(20),
            last_upstream_progress_ms: Some(200),
            last_semantic_progress_ms: Some(200),
            upstream_first_event_ms: Some(90),
            upstream_duration_ms: Some(250),
            text_bytes: 2,
            reasoning_bytes: 0,
            tool_call_bytes: 0,
            text_events: 1,
            reasoning_events: 0,
            tool_call_events: 0,
            response_shape: Some("openai"),
            upstream_status: Some(200),
            outcome: "complete",
        });
        let LogRecord::Request(summary) = rx.try_recv().unwrap() else {
            panic!("expected request record");
        };
        assert_eq!(summary.prompt_tokens, Some(19));
        assert_eq!(summary.completion_tokens, Some(14));
        assert_eq!(summary.cached_tokens, Some(0));
        assert_eq!(summary.local_input_tokens, Some(19));
    }

    #[test]
    fn flight_recorder_is_capped_and_stable() {
        let mut recorder = FlightRecorder::new();
        for index in 0..64u128 {
            recorder.record(index, FlightEventKind::Accepted);
        }
        let (events, dropped) = recorder.into_parts();
        assert_eq!(events.unwrap().len(), 32);
        assert_eq!(dropped, 32);
    }

    #[test]
    fn queue_saturation_never_blocks_and_counts_drops() {
        let (tx, rx) = mpsc::sync_channel::<LogRecord>(2);
        let dropped = Arc::new(AtomicU64::new(0));
        let sink = LogSink {
            tx,
            dropped: Arc::clone(&dropped),
            emitted: Arc::new(AtomicU64::new(0)),
        };
        for index in 0..10 {
            sink.emit(LogRecord::Request(Box::new(summary(&format!(
                "req_{index}"
            )))));
        }
        assert_eq!(dropped.load(Ordering::Relaxed), 8);
        assert_eq!(rx.try_recv().unwrap().request_marker(), "req_0");
    }

    #[tokio::test]
    async fn writer_roundtrip_jsonl_and_rotation_with_tiny_quota() {
        let directory =
            std::env::temp_dir().join(format!("cline-proxy-obs-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let (sink, handle) = spawn_writer(WriterConfig {
            directory: directory.clone(),
            max_file_size_mb: 1, // tiny quota forces rotation/cleanup
            max_total_size_mb: 1,
            cleanup_target_percent: 85,
            flush_interval_ms: 50,
        })
        .unwrap();
        // Sequential stress: enough records to force several segments.
        for index in 0..20_000u64 {
            let mut record = summary(&format!("req_{index}"));
            record.duration_ms = index;
            sink.emit(LogRecord::Request(Box::new(record)));
        }
        // Drop the sender to trigger the bounded shutdown drain.
        drop(sink);
        let _ = handle.join();
        let files: Vec<_> = std::fs::read_dir(&directory)
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert!(!files.is_empty(), "jsonl segments written");
        // Total on disk within the quota (segments + overhead).
        let total: u64 = files
            .iter()
            .map(|entry| entry.metadata().unwrap().len())
            .sum();
        assert!(total <= 1024 * 1024 + 64 * 1024, "quota exceeded: {total}");
        // Every surviving record parseable with the correct schema; the
        // oldest SURVIVING records are later than req_0 because the quota
        // cleanup deleted the earliest segments (by design).
        let mut parsed = 0u64;
        for entry in &files {
            for line in std::fs::read_to_string(entry.path()).unwrap().lines() {
                let value: serde_json::Value = serde_json::from_str(line).unwrap();
                assert!(value["request_id"].as_str().unwrap().starts_with("req_"));
                assert!(value["duration_ms"].as_u64().is_some());
                parsed += 1;
            }
        }
        assert!(parsed > 0, "records survived the quota cleanup");
        let _ = json!(null); // keep json import used
        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn writer_failure_degrades_without_panicking() {
        // A directory path that is actually a FILE: create_dir_all fails.
        let blocker =
            std::env::temp_dir().join(format!("cline-proxy-obs-blocker-{}", std::process::id()));
        std::fs::write(&blocker, b"not a directory").unwrap();
        let error = FileWriter::scan(&blocker, 1024).unwrap_err();
        assert!(
            error.kind() == std::io::ErrorKind::Other
                || error.kind() == std::io::ErrorKind::AlreadyExists
                || error.kind() == std::io::ErrorKind::PermissionDenied
        );
        std::fs::remove_file(&blocker).ok();
    }
}
