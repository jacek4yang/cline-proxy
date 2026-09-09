# ADR 0007: Adaptive bounded observability and resource bounds

Date: 2026-09-09
Status: Accepted
Scope: `src/obs.rs`, request lifecycle logging, file writer
Related: issue #16, issue #14 (non-stream strategy), docs/OBSERVABILITY.md

## Context

Before this ADR every request emitted 8-20 tracing INFO lifecycle lines
(accepted, optimization, upstream selected, first event, stream closed,
token accounting, ...). Under sustained multi-agent traffic this costs
per-request formatting, unbounded console noise, and unbounded file
growth. The model path itself is verified and stable (300 K-token
sessions, 99.8-100% warm cache) — the runtime around it needed to become
quiet, bounded, and diagnosable without competing with inference.

## Decision

1. **One summary per request.** A request-local `RequestTrace` /
   `SummaryBuilder` aggregates counters, timings, tokens, and routing in
   memory (field updates only). On completion exactly ONE
   `RequestSummary` is emitted: one compact console line + one JSONL
   record. Detailed lifecycle lines drop to `debug`.

2. **RAM flight recorder for anomalies.** Detailed lifecycle events
   (enum + relative-ms, zero content, cap 32) live only per-request.
   Anomalies (errors, 429, transport failure, SSE parse failure, TTFT /
   duration over configured thresholds) attach the trace to the summary;
   normal requests drop it. Anomalies only affect log retention — never
   routing, reasoning, or retries.

3. **Bounded non-blocking pipeline.** Records cross to the writer via a
   bounded channel (8192) with `try_send`; a full queue drops the record
   and counts it. A single dedicated `std::thread` (not an async task)
   owns JSON serialization, buffered writes (512 KB), rotation (64 MB),
   and quota enforcement — the request path never touches the filesystem
   and never serializes JSON for logging.

4. **Hard disk quota.** The managed log directory never exceeds
   `max_total_size_mb` (default 1 GB); past the quota the oldest
   segments are deleted down to `cleanup_target_percent` (85%).
   Retention state is scanned once at startup, then maintained in
   memory. No per-record directory IO.

5. **No fsync.** A crash may lose the last buffered records; durability
   belongs to `runtime-state.json` (the only state that matters across
   restarts), not to observability.

6. **Compression optional, default none.** The quota already bounds
   disk; rotating + re-reading + compressing raw JSONL costs CPU and IO
   that inference could use. If a future benchmark proves a win,
   zstd-1 with one background worker is the design slot.

7. **Failure-open.** Any logging failure degrades file logging to
   disabled with a rate-limited warning. The proxy never panics and no
   request ever fails because of logging.

## Alternatives considered

- Keep wide tracing lines at INFO, add log rotation in a subscriber
  layer: still 8-20 formatted records per request and per-chunk
  temptation; the summary approach removes the work instead of moving it.
- Async writer task: rejected — `tokio::spawn` with file IO occupies
  Tokio workers and can pile up under disk stalls; a dedicated thread is
  the correct primitive for a single ordered buffered writer.
- fsync per record / per second: rejected — observability is not a
  transaction.
- Logging content (prompts, reasoning): rejected permanently (privacy,
  size); sizes, counts, hashes, fingerprints only.

## Consequences

- Normal request ≈ 1 JSONL record + 1 console line; queue memory ≈ 8 MiB
  worst case, far less in practice (summaries serialize < 2 KiB).
- Observability can only degrade (drop records, disable file logging) —
  it cannot backpressure inference by construction.
- Real multi-agent production traffic becomes analyzable per session /
  key / turn from JSONL (`jq` recipes in docs/OBSERVABILITY.md); no new
  concurrency stress tests were added — production usage is the test.
