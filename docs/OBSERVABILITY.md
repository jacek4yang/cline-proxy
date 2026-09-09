# Observability

Adaptive bounded observability (issue #16, ADR 0007). Design law:

> Observability must degrade before inference degrades.

## Normal request = ONE summary

Every request aggregates its lifecycle in memory (`RequestTrace` fields,
counters, timings) and emits exactly one record on completion:

- **Console** (default): one compact line —

  ```text
  11:22:08 ✓ GLM53 agent=91af/31c8 key=aromacode in=188.5K cache=99.9% out=257 ttft=18012ms tool=18090ms dur=19874ms
  ```

  `✓` complete · `✗` error · `·` other. `agent` is the 16-hex HMAC
  session fingerprint (`-` when the client sent no session identity —
  never a raw id). Detailed lifecycle events remain available at
  `debug` level (`runtime.log_level: "debug"`, or `RUST_LOG`).

- **File** (`logging.directory`, default `./logs`): one JSONL object per
  request in `events-NNNNNN.jsonl` segments — schema fields:
  `request_id, protocol, requested/upstream_model, model_family, session,
  downstream_stream, upstream_strategy, selected_key_name, attempts,
  failover_count, reasoning_effort, thinking_exposure, *_max_tokens,
  request/upstream_request/system/messages/tools_bytes,
  historical_reasoning_bytes_removed, billing_header_bytes_removed,
  canonicalized_arguments, prompt/cached/completion/reasoning_tokens,
  cache_hit_ratio, reasoning_ratio, ttft/first_*/duration_ms,
  upstream_*_ms, response_shape, upstream_status, outcome`.

  jq examples:

  ```bash
  jq 'select(.outcome=="complete") | [.prompt_tokens,.cached_tokens,.cache_hit_ratio]' logs/events-000001.jsonl
  jq 'select(.error_kind) | [.request_id,.error_kind,.duration_ms]' logs/*.jsonl
  jq 'select(.cache_hit_ratio != null and .cache_hit_ratio < 50)' logs/*.jsonl
  ```

## Adaptive flight traces

Detailed lifecycle events (small enum + relative ms; zero content) are
recorded into a request-local **RAM** flight recorder (cap 32). The trace
is attached to the summary only for anomalies:

- HTTP error, effective 429, upstream timeout/transport failure,
- SSE parse failure, protocol error, empty stream,
- TTFT over `logging.slow_ttft_ms` (default 15 s),
- duration over `logging.slow_duration_ms` (default 60 s).

Normal requests drop the recorder. Anomalies **only** affect log
retention — never routing, reasoning, or retry behavior.

## Hard bounds

| resource | bound | behavior at bound |
|---|---|---|
| log queue | 8192 records (~8 MiB worst case) | `try_send`; full → record dropped + `dropped_log_records` counter; writer warns at most per flush cycle |
| disk | `max_total_size_mb` (default 1024) | oldest segments deleted to `cleanup_target_percent` (85%) of quota; active segment never deleted |
| segment | `max_file_size_mb` (default 64) | rotation: close → new segment (Windows-safe: no rename of open files) |
| writer | 1 dedicated `std::thread` | all serialization + disk IO off the request path; buffered 512 KB; flush every `flush_interval_ms` (1 s) |
| fsync | never | a crash may lose the last buffered records — observability is not a transaction |

Retention accounting scans the directory once at startup, then is
maintained in memory (no per-record `read_dir`).

## Failure-open

Any writer/disk failure (disk full, permissions, path missing) disables
file logging with a rate-limited console warning. The proxy never panics
or fails a request because of logs. `logging.directory: null`/`""`
disables file logging entirely; console summary always works.

## Privacy

Summaries carry names, counts, sizes, timings, and fingerprints only —
never prompts, source, reasoning, tool output, raw session ids, API
keys, or authorization values. Flight events are enum+timestamp only.
