# Observability

Adaptive bounded observability (issue #16, ADR 0007; recovery issue #24,
ADR 0008). Design law:

> Observability must degrade before inference degrades.

## Console destination and color

Diagnostic logs go to **stderr**. `runtime.log_color` (default `auto`):

| mode | ANSI |
|---|---|
| `auto` | only if stderr is a terminal and `NO_COLOR` is unset |
| `always` | force ANSI (`--color always`) |
| `never` | never (`--no-color` or `--color never`) |

Redirected `proxy.log` must contain **no** CSI escape sequences under `auto`.

## Normal request = ONE summary

Every request aggregates its lifecycle in memory (`RequestTrace` fields,
counters, timings) and emits exactly one record on completion:

- **Console** (default): one compact line —

  ```text
  12:30:18 ✓ GLM53 agent=0feedb6e key=gxe-outlook in=199.6K budget=216.0K cache=99.8% out=312 ttft=8.4s tool=9.1s dur=11.7s
  12:32:54 ✗ GLM53 agent=0feedb6e key=gxe-outlook in=201.2K budget=217.6K cache=n/a out=? ttft=- dur=193.4s err=upstream_semantic_idle_timeout
  ```

  `✓` complete · `✗` otherwise. `in=` prefers **local exact tokens** over
  upstream `prompt_tokens`. `out=` is provider completion tokens when
  present; if usage is missing but the stream produced output, it shows
  `t{text}+r{reasoning}+k{tool}` event counts rather than `0`. `ttft` is
  first **semantic** (reasoning/text/tool) output, not a role-only frame.
  `agent` is the 16-hex HMAC session fingerprint (`-` when unstable).

  Wide lifecycle lines (`stable prefix telemetry`, request optimization,
  exact token accounting) are **DEBUG**.

- **File** (`logging.directory`, default `./logs`): one JSONL object per
  request in `events-NNNNNN.jsonl` segments. **schema_version = 3**.
  Each process opens a **fresh** segment (`create_new`); existing files
  are counted for quota and never appended to or truncated.

  Notable fields: `schema_version`, `instance_id`, `route`
  (`direct`/`socks5`/`socks5h`),
  `local_input_tokens` / `local_token_count_method` /
  `local_token_count_duration_ms` (tokenizer; not billed usage),
  `prompt_tokens` / `cached_tokens` / `completion_tokens` (upstream
  usage), `reserved_output_tokens`, `total_context_budget`,
  `first_sse_event_ms`, `first_semantic_ms` (`ttft_ms` aliases this),
  `upstream_headers_ms`, `text/reasoning/tool_call_{bytes,events}`.

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
maintained in memory (no per-record `read_dir`). A restart always
creates a new active segment even if the last file is underfilled.

## Failure-open

Any writer/disk failure (disk full, permissions, path missing) disables
file logging with a rate-limited console warning. The proxy never panics
or fails a request because of logs. `logging.directory: null`/`""`
disables file logging entirely; console summary always works.

## Privacy

Summaries carry names, counts, sizes, timings, and fingerprints only —
never prompts, source, reasoning, tool output, raw session ids, API
keys, or authorization values. Flight events are enum+timestamp only.
