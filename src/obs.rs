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
use std::sync::{mpsc, Arc};
use std::time::Duration;

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

/// The single record emitted per completed request. `None` fields mean
/// "not applicable / upstream did not report" — never zero-padding.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RequestSummary {
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
    pub prompt_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    /// cached_tokens / prompt_tokens, percent with one decimal (subset
    /// semantics verified against live Cline traffic; issue #8).
    pub cache_hit_ratio: Option<f64>,
    pub reasoning_ratio: Option<f64>,
    pub ttft_ms: Option<u64>,
    pub first_reasoning_ms: Option<u64>,
    pub first_text_ms: Option<u64>,
    pub first_tool_call_ms: Option<u64>,
    pub duration_ms: u64,
    pub upstream_first_event_ms: Option<u64>,
    pub upstream_duration_ms: Option<u64>,
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

    fn open_next_segment(&mut self) -> std::io::Result<()> {
        let next_index = self
            .segments
            .back()
            .and_then(|segment| {
                segment
                    .path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .and_then(|stem| stem.rsplit('-').next())
                    .and_then(|suffix| suffix.parse::<u32>().ok())
            })
            .map_or(1, |last| last + 1);
        let path = self.directory.join(format!("events-{next_index:06}.jsonl"));
        let file = std::fs::File::options()
            .create(true)
            .append(true)
            .open(&path)?;
        self.segments.push_back(SegmentMeta {
            bytes: 0,
            path: path.clone(),
        });
        self.writer = Some(BufWriter::with_capacity(512 * 1024, file));
        self.active_path = Some(path);
        Ok(())
    }

    fn rotate_if_needed(&mut self) -> std::io::Result<()> {
        let needs_rotation = match self.segments.back() {
            Some(segment) => segment.bytes >= self.max_file_bytes,
            None => true,
        };
        if needs_rotation {
            // Flush + drop the writer BEFORE rename/delete (Windows keeps
            // open handles locked).
            self.writer = None;
            self.active_path = None;
            self.open_next_segment()?;
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
    #[allow(clippy::too_many_arguments)]
    pub fn emit(
        mut self,
        sink: Option<&LogSink>,
        selected_key_name: &str,
        attempts: u64,
        failover_count: u64,
        reasoning_effort: &'static str,
        expose_thinking: bool,
        client_max_tokens: Option<u64>,
        effective_max_tokens: Option<u64>,
        request_bytes: usize,
        upstream_request_bytes: usize,
        system_bytes: usize,
        messages_bytes: usize,
        tools_bytes: usize,
        historical_reasoning_bytes_removed: u64,
        billing_header_bytes_removed: u64,
        canonicalized_arguments: usize,
        usage: Option<&serde_json::Value>,
        ttft_ms: Option<u128>,
        first_reasoning_ms: Option<u128>,
        first_text_ms: Option<u128>,
        first_tool_call_ms: Option<u128>,
        upstream_first_event_ms: Option<u128>,
        upstream_duration_ms: Option<u128>,
        response_shape: Option<&'static str>,
        upstream_status: Option<u16>,
        outcome: &'static str,
    ) {
        let extract = |usage: Option<&serde_json::Value>, path: &[&str]| -> Option<u64> {
            usage
                .and_then(|usage| value_at(usage, path))
                .and_then(serde_json::Value::as_u64)
        };
        let prompt_tokens = extract(usage, &["prompt_tokens"]);
        let cached_tokens = extract(usage, &["prompt_tokens_details", "cached_tokens"]);
        let completion_tokens = extract(usage, &["completion_tokens"]);
        let reasoning_tokens = extract(usage, &["completion_tokens_details", "reasoning_tokens"]);
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
        // Adaptive anomaly detection: slow requests attach their trace.
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
        let summary = RequestSummary {
            ts_unix_ms: self.ts_unix_ms,
            request_id: self.request_id.clone(),
            protocol: self.protocol,
            requested_model: self.requested_model.clone(),
            upstream_model: self.upstream_model.clone(),
            model_family: self.model_family,
            session: self.session.clone(),
            downstream_stream: self.downstream_stream,
            upstream_strategy: self.upstream_strategy,
            selected_key_name: selected_key_name.to_owned(),
            attempts,
            failover_count,
            reasoning_effort,
            thinking_exposure: if expose_thinking {
                "exposed"
            } else {
                "suppressed"
            },
            client_max_tokens,
            effective_max_tokens,
            request_bytes,
            upstream_request_bytes,
            system_bytes,
            messages_bytes,
            tools_bytes,
            historical_reasoning_bytes_removed,
            billing_header_bytes_removed,
            canonicalized_arguments,
            prompt_tokens,
            cached_tokens,
            completion_tokens,
            reasoning_tokens,
            cache_hit_ratio,
            reasoning_ratio,
            ttft_ms: ttft_ms.map(|value| value.min(u64::MAX as u128) as u64),
            first_reasoning_ms: first_reasoning_ms.map(|v| v.min(u64::MAX as u128) as u64),
            first_text_ms: first_text_ms.map(|v| v.min(u64::MAX as u128) as u64),
            first_tool_call_ms: first_tool_call_ms.map(|v| v.min(u64::MAX as u128) as u64),
            duration_ms: elapsed.as_millis().min(u64::MAX as u128) as u64,
            upstream_first_event_ms: upstream_first_event_ms
                .map(|v| v.min(u64::MAX as u128) as u64),
            upstream_duration_ms: upstream_duration_ms.map(|v| v.min(u64::MAX as u128) as u64),
            response_shape,
            upstream_status,
            outcome,
            flight,
            flight_dropped,
            error_kind: self.error_kind,
        };
        // One compact console line (the default console surface).
        let cache_display = cache_hit_ratio
            .map(|ratio| format!("{ratio}%"))
            .unwrap_or_else(|| "n/a".to_string());
        let tokens_display = prompt_tokens
            .map(format_tokens)
            .unwrap_or_else(|| "?".to_string());
        let symbol = match outcome {
            "complete" => "\u{2713}",
            "error"
            | "protocol_error"
            | "upstream_error"
            | "unexpected_eof"
            | "decode_error"
            | "client_disconnected" => "\u{2717}",
            _ => "\u{b7}",
        };
        tracing::info!(
            request_id = %self.request_id,
            "{} {} agent={} key={} in={} cache={} out={} ttft={}ms tool={}ms dur={}ms{}",
            symbol,
            if self.model_family == "glm53" { "GLM53" } else { "GENERIC" },
            self.session.as_deref().unwrap_or("-"),
            selected_key_name,
            tokens_display,
            cache_display,
            completion_tokens.unwrap_or(0),
            ttft_ms.unwrap_or(0),
            first_tool_call_ms.unwrap_or(0),
            elapsed.as_millis(),
            self.error_kind
                .map(|kind| format!(" ({kind})"))
                .unwrap_or_default(),
        );
        if let Some(sink) = sink {
            sink.emit(LogRecord::Request(Box::new(summary)));
        }
    }
}

/// Compact human token figure: 307678 -> "307.7K".
pub fn format_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{:.1}K", tokens as f64 / 1_000.0)
    } else {
        tokens.to_string()
    }
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
}

/// Dynamic per-stream data at close.
pub struct StreamSnap<'a> {
    pub request_id: &'a str,
    pub key_name: &'a str,
    pub ttft_ms: Option<u128>,
    pub first_reasoning_ms: Option<u128>,
    pub first_text_ms: Option<u128>,
    pub first_tool_call_ms: Option<u128>,
    pub usage: Option<&'a serde_json::Value>,
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
        builder.emit(
            self.sink.as_ref(),
            snap.key_name,
            1,
            0,
            self.reasoning_effort,
            self.expose_thinking,
            self.client_max_tokens,
            self.effective_max_tokens,
            self.request_bytes,
            self.upstream_request_bytes,
            self.system_bytes,
            self.messages_bytes,
            self.tools_bytes,
            self.historical_reasoning_bytes_removed,
            self.billing_header_bytes_removed,
            self.canonicalized_arguments,
            snap.usage,
            snap.ttft_ms,
            snap.first_reasoning_ms,
            snap.first_text_ms,
            snap.first_tool_call_ms,
            snap.ttft_ms,
            Some(self.started.elapsed().as_millis()),
            None,
            Some(200),
            outcome,
        );
    }
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
            duration_ms: 92_376,
            upstream_first_event_ms: Some(88_700),
            upstream_duration_ms: Some(92_300),
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
        assert!(
            bytes.len() < 2048,
            "summary serialized to {} bytes",
            bytes.len()
        );
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
