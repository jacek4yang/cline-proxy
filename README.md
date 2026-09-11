# cline-proxy

`cline-proxy` is a small Rust gateway that exposes OpenAI Chat Completions and
Anthropic Messages APIs over one or more Cline API keys. Its primary use is
running Claude Code against Cline while filling one configured key at a time.

> **Routing invariant: ONLY an effective upstream HTTP 429 may trigger Cline
> API-key switching.** This includes a direct HTTP 429 and a high-confidence
> proxy wrapper that explicitly reports an upstream HTTP 429. Generic 5xx,
> network failures, timeouts, ambiguous error text, and interrupted streams
> never switch keys or replay a request.

## Architecture

The binary contains five intentionally small subsystems:

- Axum serves `/v1/messages`, `/v1/messages/count_tokens`,
  `/v1/chat/completions`, `/v1/models`, `/healthz`, and `/readyz`.
- One long-lived rustls `reqwest::Client` provides HTTP/2, gzip, pooling,
  keep-alive, connect timeout, and read-inactivity timeout behavior.
- A concurrency-safe pool holds the sticky active key, per-key effective-429
  cooldown metadata (Healthy → Cooling → HalfOpen with single-flight
  probing), and runtime counters. No lock is held over a network await.
- A debounced writer persists key cooldowns and the active key to a local
  JSON state file (name-keyed, wall-clock deadlines, atomic rename) so
  restarts never re-probe keys with known quota cooldowns.
- The Anthropic adapter converts structured messages, images, tool use/results,
  reasoning, usage, stop reasons, and stateful OpenAI SSE into Anthropic SSE.
- Central error and redaction paths remove configured secrets, Bearer values,
  API-key fields, cookies, and JWT-like values.

There is no OAuth, credential refresh, database, Redis, web UI, or balance
poller. Configuration is read once from JSON at startup; `/admin/status` is a
read-only, authenticated operational snapshot.

## Build and install

Rust 1.88 or newer is required by the locked dependency graph.

```bash
cargo build --release
install -m 0755 target/release/cline-proxy ~/.local/bin/cline-proxy
```

The binary can also be run directly with `cargo run --release`.

Safe local overrides (the binary reads the config file; flags never echo secrets):

```bash
./target/release/cline-proxy \
  --config /path/to/config.json \
  --bind 127.0.0.1:8799 \
  --state-file /tmp/cline-proxy-state.json \
  --log-directory /tmp/cline-proxy-logs \
  --no-color
```

`--color auto|always|never` controls ANSI on stderr. Default `auto`: color
only when stderr is a terminal and `NO_COLOR` is unset.

## Configuration

Copy the example and replace every placeholder locally:

```bash
cp config.example.json config.json
chmod 600 config.json
```

The default path is `./config.json`. Override it with `--config PATH` or
`CLINE_PROXY_CONFIG=PATH`.

The key list is ordered. Disabled entries remain configured but are not loaded:

```json
"cline_api_keys": [
  { "name": "cline-1", "api_key": "YOUR_CLINE_API_KEY_1", "enabled": true },
  { "name": "cline-2", "api_key": "YOUR_CLINE_API_KEY_2", "enabled": true },
  { "name": "spare",   "api_key": "YOUR_CLINE_API_KEY_3", "enabled": false }
]
```

Startup rejects an empty or non-header-safe gateway key, zero enabled Cline
keys, empty or duplicate key entries, an invalid bind address or URL, embedded
URL credentials, zero timeout/size limits, invalid tracing filters, and
malformed or reserved upstream headers. Validation errors identify a key by
index only and never print its value.

### Cline-facing headers

`upstream.headers` deliberately describes the Cline client identity. The
example sends:

```text
HTTP-Referer: https://cline.bot
User-Agent: Cline/4.1.16
X-Client-Type: cline-vscode
X-Client-Version: 4.1.16
X-Core-Version: 4.1.16
X-Platform: vscode
X-Platform-Version: 1.106.0
X-Title: Cline
```

These are HTTP semantic compatibility settings, not a claim that rustls
reproduces a Chromium, Node, or VS Code TLS ClientHello fingerprint.

The gateway owns `Authorization`, `Host`, `Content-Length`,
`Transfer-Encoding`, `Connection`, `Accept`, `Content-Type`,
`Accept-Encoding`, `Cookie`, `x-api-key`, `x-task-id`, and `x-request-id`;
configuration cannot override them (`x-task-id` is set dynamically per
session, `x-request-id` per logical request). Client headers are not blindly
copied upstream. Each upstream request is
constructed with `Authorization: Bearer <selected Cline key>` after all other
headers have been selected.

### Models and aliases

`models.default` plus alias names and targets form the deterministic local
`/v1/models` result. Aliases are applied consistently to OpenAI chat,
Anthropic Messages, and token-count request accounting. For example:

```json
"models": {
  "default": "z-ai/glm-5.3-flash",
  "aliases": {
    "claude-sonnet-4-6": "z-ai/glm-5.3-flash"
  }
}
```

No live model-list request is needed for readiness or model discovery.

### GLM-5.3-Flash request policy (reasoning, output, context)

Requests are optimized for coding-agent workloads before they reach Cline.
Full rationale and evidence: `docs/GLM53_FLASH.md` and
`docs/adr/0004-glm53-bounded-reasoning-and-context-policy.md`.

```json
"glm53": {
  "reasoning": {
    "default_effort": "high",
    "adaptive_effort": "high",
    "strip_historical_thinking": true,
    "expose_thinking": "requested_only"
  },
  "limits": { "max_output_tokens": 16384 },
  "context": { "safe_compaction": true },
  "telemetry": {
    "exact_input_tokens": true,
    "max_concurrent_token_counts": 1
  }
}
```

- **Explicit reasoning effort, always.** The official GLM template coerces
  unset effort to `max`; cline-proxy never sends unset. Without explicit
  client controls the effort is `high` (strong analysis/planning without
  multi-minute `max` runaways). `disabled`->`low`, `adaptive`->`high`,
  small `budget_tokens` (<8192)->`low`, large->`high`, and only an explicit
  `output_config.effort: "max"` produces `max`. Precedence: explicit
  `output_config.effort` > explicit `thinking` > proxy default.
- **Historical thinking is stripped per reasoning epoch.** The epoch
  boundary is the newest *human* user message — tool results continue the
  current epoch rather than starting one. Reasoning from previous epochs
  is removed; the current epoch keeps its in-turn reasoning continuity
  across the tool loop. Text, tool calls, call ids, and order are always
  untouched. This is the reliable local equivalent of GLM's
  `clear_thinking`, refined by real tool-loop semantics (ADR 0006).
- **Reasoning shadow store** (`shadow_current_turn`, default on): when
  the client did not request thinking, the proxy briefly keeps the
  in-turn reasoning that issued tool calls and restores it onto the
  matching assistant turn of the next request in the same epoch — tool
  loops keep their reasoning continuity without Claude Code ever storing
  or replaying reasoning. Memory-only, bounded (256 sessions / 64 MiB /
  1 MiB per entry / 10-minute TTL), never truncated, never logged,
  never persisted; disabled automatically when no stable session identity
  exists.
- **Thinking exposure is `requested_only`:** upstream reasoning is surfaced
  to the client as Anthropic thinking blocks only when the request
  explicitly carries `thinking`. This prevents Claude Code from storing and
  re-sending reasoning (the main multi-turn amplification source).
- **Output is capped**: `effective_max_tokens = min(client, 16384)` by
  default; a request without a bound gets the cap.
- **Safe compaction only**: lossless structural normalization (single text
  block -> string, empty blocks dropped, Anthropic-only `metadata` dropped).
  Tool results are never truncated and tool schemas are never edited;
  Claude Code keeps full ownership of context compaction.
- **Model-scoped policy.** The GLM semantics above apply only when the
  resolved *upstream* model is a GLM id (`glm*` segment detection).
  Every other OpenAI-compatible model passes through byte-for-byte: no
  injected `reasoning_effort`, no output cap, no message rewriting, no
  metadata removal. Unknown models fail safe toward compatibility, never
  toward GLM semantics.
- **Telemetry**: per-request byte breakdown and policy decisions
  (`request optimization` log, including the model family), per-stream
  reasoning/text/tool-call byte accounting with first-tool-call latency,
  upstream usage tokens (`prompt/completion/cached/reasoning`) when
  provided, and an exact GLM token count of the optimized request. The
  tokenizer is CPU-bound, so the count runs on the **blocking pool under a
  semaphore** (`max_concurrent_token_counts`, default 1) — never on a Tokio
  worker and never queued unboundedly; when the slot is busy the count is
  skipped and logged. Logs contain sizes/counts only, never prompt content.

### Adaptive bounded observability

One request = one summary. A request-local trace aggregates counters,
timings, tokens, and routing in memory; on completion exactly one compact
console line and one JSONL record are emitted through a **bounded queue
→ dedicated writer thread** (disk IO never runs on a Tokio worker, a
full queue drops the record instead of ever slowing a request). Detailed
lifecycle events live in a request-local RAM flight recorder and are
attached only to anomalous requests (errors, 429, transport failures,
TTFT/duration over `logging.slow_ttft_ms`/`slow_duration_ms`).
File logging has a hard disk quota (`max_total_size_mb`, default 1 GB)
with 85% cleanup watermark and 64 MB rotation; there is no fsync;
failures degrade file logging to disabled — the proxy never fails a
request because of logs. Logs carry names, counts, sizes, timings, and
fingerprints only, never prompts, reasoning, tool output, or raw session
ids. See `docs/OBSERVABILITY.md`, `docs/PERFORMANCE.md`, and ADR 0007.

### Prompt-prefix stability (cache locality)

Upstream prompt caches key on byte-exact prefixes, so per-turn byte drift
in an otherwise identical conversation wastes prefill. cline-proxy
addresses the drift sources it can (issue #8, ADR 0005):

- **Volatile billing header**: Claude Code prepends an
  `x-anthropic-billing-header: ...` line to the system text whose
  attribution metadata changes between requests. A *leading* line of
  exactly that shape is stripped (LF/CRLF/CR aware); a header mentioned
  later in the text is never touched. Applied in Anthropic system
  normalization so the wire body, `/v1/messages/count_tokens`, and
  telemetry all see the same normalized system.
- **Canonical tool-argument JSON**: historical assistant
  `tool_calls[].function.arguments` strings are re-serialized with
  deterministic key order so equivalent arguments are byte-identical
  across turns. Arrays keep order; malformed strings, plain-text tool
  results, shell output, and source code are never rewritten.
- **Stable prefix telemetry**: each request logs `prefix_hash` and
  `prefix_bytes` (hash of normalized system + messages + tools). Equal
  hashes prove local byte stability, not an upstream cache hit.
- **Session fingerprint**: when Claude Code supplies a session identity
  (`metadata.user_id`/`session_id`), a 16-hex-char HMAC fingerprint is
  logged (`session=...`). Raw ids are never logged; without identity the
  field is `unstable` and no session-scoped behavior is attempted.
- **Upstream X-Task-ID**: when a session identity exists, the proxy sends
  that same fingerprint as a dynamic `X-Task-ID` header to Cline and keeps
  it byte-identical across credential failover (only Authorization
  rotates on an effective 429). Raw session ids are never forwarded.
  Without identity the header is not sent and affinity is never guessed
  from connection, IP, key, or recent requests. Cline's official client
  sends an equivalent per-conversation header; the proxy preserves the
  shape, but Cline server-side routing/cache use of `X-Task-ID` is
  undocumented and not guaranteed.
- **Cache ratios**: `cache_hit_ratio` and `reasoning_ratio` are logged
  from upstream usage. Verified against live traffic: Cline reports
  `cached_tokens` (subset of `prompt_tokens`), and a real ~300 K-token
  Claude Code session sustained **99.8-100.0% cache hit ratios** across
  consecutive turns with these stability mechanisms enabled
  (`docs/DEVELOPMENT_STATE.md`). `prompt_cache_key` is *not* sent —
  high cache hits are achieved without it.

## Key stickiness and persisted quota state

Routing is **strict sticky sequential**. The active key is used for every
request until it confirms an effective upstream HTTP 429; then it enters
cooldown and the next configured key becomes active until *its* quota is
exhausted. Successes, 5xx, timeouts, connection resets, TLS/DNS errors, and
stream interruptions never rotate the key, and a recovered key whose cooldown
expired never steals the active role back from a healthy key.

Key cooldowns and the active selection survive process restarts:

- `runtime.state_file` (default `./runtime-state.json`; set to `null` or `""`
  to disable) stores one versioned JSON snapshot.
- The file is keyed by configured key **name** (never by position), so
  reordering the key list cannot cool the wrong key.
- Deadlines are wall-clock Unix milliseconds; nothing secret is ever written:
  no API keys, no authorization values, no raw upstream error text.
- Writes are atomic (temp file + rename) and debounced off the request path;
  healthy requests never touch the disk. A final flush runs during graceful
  shutdown.
- A missing, truncated, or unreadable state file is **non-fatal**: the gateway
  logs a sanitized warning and starts with empty state. Persistence health is
  visible in `/admin/status`.
- On startup, expired cooldowns are restored as probe-eligible (never as
  active cooldowns), and the persisted active key is restored if it still
  exists and is enabled — so a restart no longer re-probes keys that are
  hours away from quota recovery.

A confirmed effective 429 is further classified as `daily_quota`, `transient`,
or `unknown` (`rate_limit_kind` in `/admin/status`). Classification happens
only *after* the 429 is confirmed and never widens failover conditions.

## Run

```bash
cargo run --release -- --config config.json
# or
./target/release/cline-proxy --config config.json
```

Upstream stall detection (issue #24) is separate from the Reqwest read
timeout (`upstream.timeout_secs`, default 600, last-resort inactivity
backstop). Defaults:

```text
first_event_timeout_secs      = 180   # headers → first SSE/data
first_semantic_timeout_secs   = 180   # headers → reasoning/text/tool
stream_idle_timeout_secs      = 120   # max gap between upstream bytes
semantic_idle_timeout_secs    = 180   # max gap between semantic deltas
```

`0` disables an individual timer. Downstream Anthropic pings do not reset
these. A stalled committed stream emits one Anthropic error and is never
replayed. Optional `glm53.context.upstream_context_window_tokens` is unset
by default — do not guess a Cline serving limit.

Outbound Cline traffic uses one shared rustls HTTP/2 client. `upstream.proxy`
is optional:

```text
null / omitted     → direct (environment HTTP(S)_PROXY is ignored)
socks5://host:port → SOCKS5, local DNS
socks5h://host:port → SOCKS5, proxy DNS
```

Override without copying the config: `--upstream-proxy socks5://127.0.0.1:10888`
or `--upstream-proxy direct`. Proxy credentials, if present, are never logged.
SOCKS/connect failures do not rotate Cline keys and never replay a committed
stream.

`SIGINT` and `SIGTERM` stop new accepts and allow active requests to drain for
`runtime.shutdown_timeout_secs` before remaining connections are aborted.

Set `RUST_LOG` to override `runtime.log_level`. `runtime.log_format` accepts
`pretty` or `json`.

## Gateway authentication

API routes require the gateway's `server.api_key`, supplied in either form:

```text
Authorization: Bearer <gateway key>
x-api-key: <gateway key>
```

Health endpoints are intentionally unauthenticated. The gateway key is never
forwarded. The Cline keys are never returned to clients.

`GET /admin/status` (same gateway authentication) returns a read-only
operational snapshot: version, uptime, the active key and its age, and per-key
state (`healthy` / `cooling` / `half_open` / `probing`), cooldown deadlines,
rate-limit kind, model scope, and request counters. It never returns key
material or raw upstream error text.

## Claude Code

Point Claude Code's Anthropic base URL at the local gateway. Provide
`CLINE_PROXY_GATEWAY_KEY` through your normal secret-management mechanism; it
must equal `server.api_key`:

### Exact token counting

`POST /v1/messages/count_tokens` returns the **exact** GLM-5.3-Flash prompt
token count, computed in-process with the official tokenizer and official
chat template (zai-org/GLM-5.3-Flash, revision pinned in
`docs/GLM53_FLASH.md` — no Python, no network at runtime). Responses carry
`x-cline-proxy-token-count: exact_glm53_optimized` (counting the optimized request the gateway would actually send; the oracle-faithful count is used when the strip policy is disabled). See
`docs/adr/0003-glm53-exact-tokenizer.md` for the parity guarantees, the
reasoning-effort mapping, and the two documented exclusions (non-text
documents are rejected rather than undercounted; images count as their
template placeholder).

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8788
export ANTHROPIC_AUTH_TOKEN="${CLINE_PROXY_GATEWAY_KEY:?set CLINE_PROXY_GATEWAY_KEY}"
export ANTHROPIC_MODEL=claude-sonnet-4-6
export ANTHROPIC_DEFAULT_OPUS_MODEL=claude-sonnet-4-6
export ANTHROPIC_DEFAULT_SONNET_MODEL=claude-sonnet-4-6
export ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-sonnet-4-6
export API_TIMEOUT_MS=600000
claude
```

If a Claude Code version uses `ANTHROPIC_API_KEY` instead, set it to the same
gateway key. The gateway accepts both Bearer and `x-api-key` authentication.

Messages compatibility includes system and mid-conversation system content,
text, base64/URL images, documents where the upstream supports OpenAI file
parts, assistant `tool_use`, user `tool_result`, tools/input schemas,
tool choice, parallel tools, sampling and stop fields, response usage, and
reasoning fields that can be represented as Anthropic thinking blocks.

Streaming uses a bounded, stateful SSE decoder. It emits `message_start`,
content block start/delta/stop, `message_delta`, and `message_stop`; tool
arguments can be split at arbitrary chunk boundaries and multiple call indexes
can be interleaved. Malformed or truncated upstream streams produce one
Anthropic error event and are never replayed. Dropping the downstream body
drops the reqwest body so abandoned streaming work is cancelled upstream.

## Manual integration tests

Set `CLINE_PROXY_GATEWAY_KEY` to `server.api_key` through your normal
secret-management mechanism. These commands do not use or reveal a Cline key.

Model list:

```bash
curl -sS http://127.0.0.1:8788/v1/models \
  -H "Authorization: Bearer ${CLINE_PROXY_GATEWAY_KEY:?set CLINE_PROXY_GATEWAY_KEY}"
```

OpenAI non-stream:

```bash
curl -sS http://127.0.0.1:8788/v1/chat/completions \
  -H "Authorization: Bearer ${CLINE_PROXY_GATEWAY_KEY:?set CLINE_PROXY_GATEWAY_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Reply with OK"}]}'
```

OpenAI stream:

```bash
curl -N http://127.0.0.1:8788/v1/chat/completions \
  -H "Authorization: Bearer ${CLINE_PROXY_GATEWAY_KEY:?set CLINE_PROXY_GATEWAY_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-sonnet-4-6","stream":true,"messages":[{"role":"user","content":"Count to three"}]}'
```

Anthropic stream:

```bash
curl -N http://127.0.0.1:8788/v1/messages \
  -H "x-api-key: ${CLINE_PROXY_GATEWAY_KEY:?set CLINE_PROXY_GATEWAY_KEY}" \
  -H 'anthropic-version: 2023-06-01' \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-sonnet-4-6","max_tokens":128,"stream":true,"messages":[{"role":"user","content":"Reply with OK"}]}'
```

Health checks:

```bash
curl -fsS http://127.0.0.1:8788/healthz
curl -fsS http://127.0.0.1:8788/readyz
```

## Exact rate-limit behavior

Selection is sticky and sequential. With keys 1, 2, and 3, normal requests
continue using key 1. If and only if key 1 produces an effective HTTP 429, that
key enters cooldown and the same logical request may try key 2. Later requests
stay on key 2 until it is rate-limited. When the current key is limited,
selection scans forward and wraps to any older key whose cooldown has expired.

An effective 429 is either the configured upstream's direct HTTP 429 response,
or an outer HTTP 5xx whose bounded error body provides high-confidence proxy
evidence. Structured wrappers recognize explicit `upstream_status`,
`upstreamStatus`, `status_code`, `statusCode`, `http_status`, and `httpStatus`
fields; `error.status` is also accepted inside an error object. Text wrappers
recognize `upstream returned [HTTP] 429`, `upstream status 429`, `upstream
response status: 429`, and `upstream error: 429` only when the same body also
contains rate-limit or quota semantics. Merely containing the number `429` is
never sufficient.

For one request, every enabled key is tried at most once. The maximum number of
upstream attempts is therefore the enabled key count. If all are cooling or all
return 429, the client receives HTTP 429 with `all Cline API keys are currently
rate-limited` and, when known, an approximate earliest `Retry-After`.

Cooldown precedence is:

1. `Retry-After` delta-seconds or HTTP date.
2. Clear structured JSON retry fields.
3. Text such as `Try again in 2h 30m`, `Retry in 3h`, or `Retry after 30ms`.
4. `upstream.fallback_429_cooldown_secs`.

The parser accepts `d`, `h`, `m`, `s`, and `ms`, combines components, checks
overflow, caps unreasonable values, handles invalid UTF-8 lossily, and never
panics on malformed input.

Every response not classified as an effective 429—including 400, 401, 402,
403, 404, 408, 409, 422 and generic 5xx responses—is sanitized and returned
without switching keys. DNS, TLS, connection, reset, timeout, body-read,
invalid JSON, and SSE errors also return without switching. Transport-error
strings are never inspected for status codes. There are no hidden generic 5xx
retries or automatic auth retries.

Failover classification happens on the initial HTTP error response before any
response body is exposed downstream. Error inspection is capped at 64 KiB and
the same buffered, sanitized body is used if the error is returned to the
client. After an OpenAI or Anthropic stream body exists, no retry path is
reachable. This prevents duplicate text, tool calls, commands, file edits, or
patches.

## Logging and security

Logs include request ID, protocol, model/alias, streaming mode, selected key
name/index, status, duration, attempt/failover count, cooldown source and
duration, first Anthropic event, periodic stream counters, completion, and
sanitized error class. Rate-limit logs preserve both the proxy's outer status
and the classified effective status. Prompts and full request headers are not
logged.

Configured Cline and gateway key strings are removed from upstream error bodies
in addition to structural redaction of Authorization, API keys, access/refresh
tokens, cookies, Bearer values, and JWT-like strings. Key names are operational
labels and may appear in logs; do not put secret material in a key name.

Protect `config.json` as a credential file, bind to loopback unless remote
access is deliberately secured, use a long random gateway key, and put TLS in
front of the service before exposing it beyond a trusted host. This gateway
does not encrypt the config file or provide credential management.

## Troubleshooting

- Startup `server.api_key must not be empty`: choose a local gateway key; it is
  separate from all Cline keys.
- `all Cline API keys are currently rate-limited`: wait for the earliest
  cooldown or add another enabled key and restart.
- A 401/403 does not move to another key by design. Correct or replace that
  configured Cline key, then restart.
- `could not connect to upstream`: check DNS, TLS roots, firewall/proxy policy,
  and `upstream.base_url`. The request is not retried on another key.
- `upstream stream ended unexpectedly`: the provider closed without a final
  stop reason; the gateway intentionally did not replay generation.
- Token counts differ from provider billing: this endpoint is documented as a
  local approximation because no Cline tokenizer endpoint is assumed.

## Development

The test suite is offline and uses deterministic loopback mock upstreams. It
covers normal OpenAI/Anthropic requests, SSE and cancellation, reasoning,
usage, tool loops and fragmented parallel tools, direct and proxy-wrapped 429
routing, false-positive wrapper rejection, cooldowns, bounded attempts,
concurrency, header ownership, and secret redaction.

```bash
cargo fmt --all -- --check
cargo check --all-targets
cargo test --all-targets
cargo clippy --all-targets --all-features -- -D warnings
```

Some restricted sandboxes require permission for `cargo test` to bind
ephemeral `127.0.0.1` ports. No test contacts `api.cline.bot`.
