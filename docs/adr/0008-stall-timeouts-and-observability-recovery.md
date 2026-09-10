# ADR 0008: Stall timeouts and observability recovery

Date: 2026-09-10
Status: Accepted
Scope: `src/obs.rs`, `src/stream_watch.rs`, `src/console.rs`, `src/context_guard.rs`, `src/anthropic.rs`, `src/main.rs`
Related: issue #24, ADR 0007

## Context

Production evidence (`D:\Workspace\cline-proxy-bin`) showed four independent
operational failures on an otherwise correct model path:

1. Redirected `proxy.log` contained tens of thousands of ANSI CSI sequences.
2. INFO dumped lifecycle copies of JSONL (`stable prefix telemetry`,
   `request optimization`, `exact GLM token accounting`).
3. `FileWriter::scan()` discovered existing JSONL segments but left
   `writer = None`, so the next write could become `writer closed`.
4. The only stall detector was Reqwest `read_timeout = timeout_secs`
   (default 600s). A committed stream that produced a first event and then
   went silent could hang Claude Code for ~10 minutes. Failed summaries
   also dropped already-known local exact token counts (`in=?`).

A 453,504-token Cline request in the same JSONL **completed**, so a ~200k
serving cap must not be hard-coded from one stall.

## Decision

1. **Fresh JSONL segment per process.** Startup scans existing files for
   quota accounting, then `create_new`s `events-NNNNNN.jsonl`. Old files
   are never appended to or truncated. `writer closed` is not a normal
   restart state.

2. **Color = auto on stderr.** ANSI only when stderr is a terminal and
   `NO_COLOR` is unset. `--color always|never` and `--no-color` override.
   JSONL never carries CSI.

3. **JSONL schema_version = 2** with `instance_id`, local vs upstream
   tokens, first-sse vs first-semantic timings, and output byte/event
   counts. Console INFO is one compact line per request.

4. **Explicit stall timers** (defaults; `0` disables a timer):

   | timer | default | meaning |
   |---|---:|---|
   | `first_event_timeout_secs` | 180 | headers → first SSE/data |
   | `first_semantic_timeout_secs` | 180 | headers → reasoning/text/tool |
   | `stream_idle_timeout_secs` | 120 | max gap between upstream bytes |
   | `semantic_idle_timeout_secs` | 180 | max gap between semantic deltas |

   Downstream Anthropic pings do not reset these. A stalled **committed**
   stream emits one Anthropic error event and is never replayed.
   `timeout_secs` remains the Reqwest read-timeout backstop (600).

5. **Optional context guard.**
   `glm53.context.upstream_context_window_tokens` is `null` by default.
   When set, input+output+margin must fit; output may be reduced to
   remaining headroom; input that cannot fit is rejected before generation.
   No guessed Cline route limit is configured.

## Consequences

- Claude Code sees a protocol error in seconds-to-minutes on a hung
  upstream instead of a ~600s socket timeout.
- Restarting the proxy does not disable JSONL.
- Redirected logs are grep-able.
- Context enforcement waits on evidence; operators can set a window
  once a real Cline catalog/route limit is known.
