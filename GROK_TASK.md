# GROK_TASK.md — Production recovery and hardening of cline-proxy

## Objective

The current `cline-proxy` is not reliably usable with Claude Code in the user's real environment. Treat this as a production-recovery task, not a cosmetic refactor.

Repository:

```text
https://github.com/jacek4yang/cline-proxy
```

Real runtime/evidence directory:

```text
D:\Workspace\cline-proxy-bin
```

Primary client:

```text
Claude Code
```

Primary upstream model:

```text
z-ai/glm-5.3-flash
```

The user has configured Claude Code as if this model has a 1M context window. You must establish whether the Cline serving route actually supports that usable context before trusting the client declaration.

Use Grok 4.6 with High reasoning for the main investigation. Continue through implementation, tests, real secret-safe validation, PR/CI, fixes, and merge. Do not stop after producing a plan.

No release/tag is requested.

---

# 1. Known evidence that must be reproduced or explained

## 1.1 Stream stall / ~600 second failure

Observed real request:

```text
optimized local input tokens: 199,634
first upstream/stream event: ~13.1 s
no useful final response
request duration: ~603.5 s
final class: stream_transport
downstream stream had already been committed
```

Current source uses a Reqwest read timeout tied to:

```text
upstream.timeout_secs
default = 600
```

The failure duration strongly matches this ~600 second inactivity boundary.

Do not assume this means “GLM reasoned for ten minutes”. Determine:

- time to upstream headers;
- time to first raw SSE event;
- time to first *semantic* event;
- last upstream byte;
- last semantic progress;
- whether only role/metadata/heartbeat frames arrived;
- whether the provider stalled, the network stalled, or a context boundary was hit.

## 1.2 Possible context-window mismatch

A real request had:

```text
optimized_input_tokens = 199,634
effective_max_tokens = 16,384
nominal total budget = 216,018
```

The client is configured for 1M context, but that does NOT prove Cline's deployed `z-ai/glm-5.3-flash` route has a 1M usable context.

Investigate whether there is a real serving limit around 200K, 256K, or another boundary.

Do not hard-code 200K merely because one request stalled near it.

## 1.3 JSONL writer restart bug

Current `FileWriter::scan()` discovers existing `.jsonl` segments but initializes:

```text
writer = None
active_path = None
```

`rotate_if_needed()` does not open a writer if the last discovered segment is below rotation size, so the next `write_line()` can produce:

```text
writer closed
```

Observed:

```text
WARN file logging disabled after write failure; proxy continues error=writer closed
```

This is a deterministic bug and must receive a regression test.

## 1.4 ANSI escape sequences in redirected `proxy.log`

The supplied screenshot shows literal sequences such as:

```text
ESC[2m
ESC[32m
ESC[0m
```

inside:

```text
D:\Workspace\cline-proxy-bin\proxy.log
```

This is not acceptable for a redirected/plain log file.

The terminal should remain readable and optionally colored, but a redirected stream/file must not contain ANSI escapes by default.

## 1.5 Excessive INFO lifecycle noise

The screenshot also shows wide INFO records such as:

```text
stable prefix telemetry
request optimization
exact GLM token accounting
```

with very large per-request field sets.

The intended operational UX is:

```text
normal request -> one compact INFO summary
important runtime state -> concise INFO
warnings/errors -> WARN/ERROR
detailed lifecycle -> DEBUG/TRACE
structured durable evidence -> JSONL
```

Do not make INFO a second copy of all JSONL fields.

## 1.6 Diagnostic loss on failed streams

The failed request summary showed roughly:

```text
in=?
cache=n/a
out=0
```

even though local exact token accounting had already established:

```text
input_tokens=199634
```

A failed upstream stream may never return usage, but diagnostics must not discard already-known local facts.

Separate local token measurement from upstream-billed usage.

---

# 2. Secret-safety protocol — mandatory

The actual file:

```text
D:\Workspace\cline-proxy-bin\config.json
```

contains real API credentials.

You are authorized to USE it for controlled local testing, but you are NOT authorized to reveal it.

## 2.1 Never display the raw file

Do NOT execute commands whose output exposes it, including:

```powershell
type D:\Workspace\cline-proxy-bin\config.json
Get-Content D:\Workspace\cline-proxy-bin\config.json
gc D:\Workspace\cline-proxy-bin\config.json
cat D:\Workspace\cline-proxy-bin\config.json
```

Do not open the raw config through an agent file-reading tool that would place secret values in model context.

## 2.2 Safe inspection

If structure is needed, create/use a LOCAL sanitizer script that:

- parses JSON locally;
- emits only safe field paths;
- emits number of keys;
- emits enabled flags;
- emits configured key names;
- emits secret lengths or HMAC/fingerprint only if useful;
- replaces every secret value with `[REDACTED]`;
- never prints raw Authorization/header/key values.

The sanitizer must itself contain no secret literal.

## 2.3 Real execution

Prefer letting `cline-proxy.exe` / the newly built binary read the real config directly:

```powershell
.\target\release\cline-proxy.exe --config D:\Workspace\cline-proxy-bin\config.json
```

If isolation requires bind/state/log overrides, add safe CLI overrides so the same real config can be used without copying its secrets.

Preferred additions if needed:

```text
--config <path>
--bind <127.0.0.1:temporary-port>
--state-file <temporary path>
--log-directory <temporary path>
```

CLI overrides must never echo secret values.

## 2.4 Claude Code smoke wrapper

If actual Claude Code E2E needs the local proxy auth token, use a generic PowerShell helper that reads the config locally and sets process environment variables without printing the value.

The script source must not contain any key.

Do not use verbose shell tracing.

## 2.5 Git hygiene

Before every commit:

```powershell
git status --short
git ls-files
```

Verify no runtime secrets are tracked.

Check ignored runtime files without printing contents.

Add narrow `.gitignore` rules for secret-bearing/local-runtime files.

Never commit:

```text
config.json
runtime-state.json
proxy.log
logs/
*-smoke-config.json
*-live-config.json
```

If historical credentials are detected, state that rotation is required; never print the credential.

---

# 3. Phase 0 — establish ground truth

Before editing:

```powershell
git status
git branch --show-current
git remote -v
git fetch --all --prune
git checkout main
git pull --ff-only
git log --oneline --decorate -30
gh repo view
gh pr list --state all
gh issue list --state all
```

Read:

```text
AGENTS.md
GROK_TASK.md
docs/DEVELOPMENT_STATE.md
docs/GLM53_FLASH.md
docs/OBSERVABILITY.md
docs/PERFORMANCE.md
all accepted ADRs
```

Audit at least:

```text
src/main.rs
src/server.rs
src/upstream.rs
src/anthropic.rs
src/obs.rs
src/config.rs
src/optimize.rs
src/reasoning_shadow.rs
src/pool.rs
src/rate_limit.rs
Cargo.toml
config.example.json
.github/workflows/
.gitignore
```

Run baseline:

```powershell
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo test --release --workspace --all-targets --all-features
```

Record exact baseline commit and test counts.

---

# 4. Phase 1 — audit real logs as a dataset

Audit:

```text
D:\Workspace\cline-proxy-bin\proxy.log
D:\Workspace\cline-proxy-bin\logs\*.jsonl
D:\Workspace\cline-proxy-bin\runtime-state.json
```

Do not mutate them.

Build a small local diagnostic tool/script if useful. It must emit no prompt/reasoning/tool content and no secrets.

## 4.1 Group by request_id/session/key

Compute safe metrics:

```text
request count
success/error counts
duration p50/p95/max
upstream headers time
first raw SSE event
first reasoning/text/tool event
first semantic event
semantic idle duration
stream_transport count
unexpected_eof count
protocol errors
effective 429s
failovers
cache ratios
local exact token counts
upstream prompt tokens
input-size buckets
context-budget buckets
writer failures
dropped log records
```

Correlate slow/error requests with:

```text
input token count
input bytes
effective max output
total context budget
cache hit
selected key name
stream strategy
time of day
reasoning exposure/effort
```

## 4.2 Produce a safe local diagnosis report

Create a report under a non-secret development path, e.g.:

```text
target/diagnostics/runtime-audit.md
```

It may contain aggregate numbers and request IDs/fingerprints, but no raw content or credentials.

Do not commit production-specific request IDs unless there is a clear reason; prefer a summarized report for the PR.

---

# 5. Phase 2 — fix the log writer restart defect first

Create a failing regression test that reproduces:

```text
existing underfilled events-*.jsonl
-> process/writer restart
-> first RequestSummary
-> "writer closed"
```

Then fix it.

Preferred design:

```text
each process/writer startup opens a fresh active segment
```

rather than ambiguously “discovering” an old segment but owning no file handle.

Requirements:

- Windows-safe;
- active segment is always explicit;
- no collision if a prior process/crash left files behind;
- quota accounting includes old segments;
- active file is never deleted by cleanup;
- restart with underfilled segment works;
- restart with full segment works;
- restart after corrupt/partial final JSONL line does not prevent new logging;
- startup does not truncate old logs;
- writer flush/shutdown remains bounded;
- `writer closed` cannot occur as a normal restart state.

Use `create_new`/unique sequence logic if needed to avoid file collisions.

Do not introduce high-frequency directory scans.

---

# 6. Phase 3 — terminal/logging output best practices

## 6.1 ANSI/color policy

Current `tracing_subscriber::fmt()` path must become explicit.

Implement an ergonomic policy such as:

```text
auto   -> ANSI only when the actual output stream is a terminal and NO_COLOR is not set
always -> force ANSI for interactive use
never  -> no ANSI
```

Default:

```text
auto
```

At minimum use Rust terminal detection (`std::io::IsTerminal`) or an equivalent lightweight implementation.

If output is redirected to `proxy.log`, it must contain NO ANSI escape bytes by default.

Support `NO_COLOR` if practical.

Optional CLI:

```text
--no-color
```

is acceptable if cleanly integrated, but do not add overlapping settings without need.

## 6.2 Console destination

Prefer normal diagnostic/operational logs on `stderr`, leaving `stdout` available for future machine-readable command output.

Be consistent.

## 6.3 INFO policy

Default INFO should contain:

```text
startup/config summary without secrets
runtime-state restore summary
listening address
one compact line per completed request
concise WARN/ERROR
```

Move high-cardinality/wide lifecycle telemetry to DEBUG/TRACE, including, unless a strong operational reason exists:

```text
stable prefix telemetry
full request optimization breakdown
exact-token-accounting breakdown
upstream transport selection detail
per-stage lifecycle chatter
```

The one request summary must retain the important numbers.

## 6.4 Structured JSONL

JSONL is the durable evidence source.

Add a schema version, e.g.:

```text
schema_version
```

and preferably a per-process safe:

```text
instance_id
```

Keep durable JSONL free of ANSI and terminal decoration.

Do not duplicate prompts, reasoning, tool outputs, source code, or credentials.

---

# 7. Phase 4 — repair observability semantics

## 7.1 Separate local token accounting from upstream usage

A request can fail after local exact counting but before upstream usage arrives.

Do not emit:

```text
in=?
```

when an exact local count is already known.

Represent facts separately, for example:

```text
local_input_tokens
local_token_count_method
local_token_count_duration_ms

upstream_prompt_tokens
cached_tokens
completion_tokens
reasoning_tokens
```

or an equally clear schema.

Never mislabel a local tokenizer count as provider-billed usage.

## 7.2 Context budget fields

When local input count is known, derive:

```text
reserved_output_tokens = effective_max_tokens
total_context_budget = local_input_tokens + reserved_output_tokens
```

If a configured/provider-confirmed context limit exists, also expose:

```text
context_limit_tokens
context_utilization_ratio
context_headroom_tokens
```

These must be diagnostic values, not guessed facts.

## 7.3 First event vs first semantic output

Current first SSE event may only be role/metadata.

Track separately:

```text
upstream_headers_ms
first_upstream_byte_ms
first_sse_event_ms
first_reasoning_ms
first_text_ms
first_tool_call_ms
first_semantic_ms
last_upstream_progress_ms
last_semantic_progress_ms
```

Define `ttft_ms` precisely. Prefer first user-visible/semantic output rather than a meaningless role-only frame.

If changing existing JSONL semantics, bump `schema_version` and document it.

## 7.4 Failed-stream output facts

If a stream produced text/reasoning/tool data before failure but no final usage, preserve safe counts:

```text
text_bytes/events
reasoning_bytes/events
tool_call_bytes/events
```

Do not misleadingly report `out=0` solely because provider usage is absent.

---

# 8. Phase 5 — replace the single 600s stall policy with explicit timeouts

The current single upstream read timeout is too coarse.

Design and implement separate concepts.

Suggested starting defaults; validate against existing real evidence before finalizing:

```text
connect_timeout_secs          = 20
first_upstream_event_timeout  = 180
first_semantic_timeout_secs   = 180
stream_idle_timeout_secs      = 120
semantic_idle_timeout_secs    = 180
overall_request_timeout_secs  = optional / generous
```

Do NOT blindly use these numbers if real logs show a safer boundary.

## 8.1 Definitions

`first_upstream_event_timeout`:
time allowed to receive initial upstream data/SSE.

`first_semantic_timeout`:
time allowed to receive actual reasoning/text/tool progress after headers.

`stream_idle_timeout`:
maximum time with no upstream bytes once the stream is established.

`semantic_idle_timeout`:
maximum time without reasoning/text/tool progress, even if the upstream sends heartbeat/metadata frames.

Do not let local downstream ping frames reset upstream-progress timers.

## 8.2 Streaming client behavior

Continue sending downstream Anthropic pings if useful for connection health, but a local ping must not disguise a stalled upstream.

On a stalled committed stream:

```text
emit one safe Anthropic error event if protocol permits
record precise error_kind
close
NEVER replay on another key
```

## 8.3 Non-stream local aggregation

Apply the same upstream stall semantics to `StreamAndAggregate`.

Do not wait for the old generic 600 second read timeout if the provider has made no meaningful progress beyond the configured boundary.

Do not automatically pay for a second generation after a stalled first generation.

## 8.4 Error classes

Produce stable categories:

```text
upstream_first_event_timeout
upstream_first_semantic_timeout
upstream_stream_idle_timeout
upstream_semantic_idle_timeout
upstream_transport
unexpected_eof
```

Add regression tests using paused/mock streams rather than waiting real minutes.

---

# 9. Phase 6 — establish the REAL Cline context capability

This is critical.

Do not infer the route limit from GLM model architecture alone.

## 9.1 First seek metadata/evidence

Before spending tokens:

- inspect Cline model/catalog endpoints or public metadata available to the current integration;
- inspect response/provider headers if relevant;
- inspect repository/public Cline model information if useful;
- determine whether a `contextWindow`, `maxInputTokens`, or equivalent exists for the exact route.

Record whether evidence is:

```text
model-native
catalog-advertised
provider-advertised
empirically observed
```

## 9.2 Controlled real probe only if necessary

If authoritative route metadata is absent or ambiguous, run a controlled context probe using the REAL config without exposing keys.

Constraints:

- actual config is read locally by the test process;
- one isolated proxy instance;
- one healthy key where practical;
- output budget tiny (e.g. 1–32 tokens);
- simple deterministic instruction;
- do not ask for long reasoning;
- no tool calls;
- stop as soon as the boundary is sufficiently characterized;
- do NOT blindly send 1M tokens;
- do NOT fan out concurrent probes;
- do NOT rotate through every key to multiply quota usage.

Suggested adaptive sequence, not a mandatory fixed list:

```text
128K
192K
224K
256K
```

Then binary-search only around the observed transition.

Only proceed to:

```text
384K / 512K / higher
```

if lower probes clearly succeed and the remaining uncertainty matters.

Generate synthetic benign context locally. Do not include repository secrets/source in the synthetic prompt.

Use the official GLM tokenizer already in the project to verify actual optimized token counts.

## 9.3 Distinguish failure modes

For each probe record:

```text
optimized input tokens
effective output reserve
total context budget
HTTP status
time to headers
first SSE event
first semantic event
completion success
provider error class
stall timeout class
duration
```

A `200 -> initial event -> stall` is not the same as a clean context-length rejection.

Do not call a threshold “the context limit” until repeated evidence supports it.

---

# 10. Phase 7 — add a provider context guard only after evidence

If the Cline route's usable context is confirmed smaller than the Claude Code client declaration, add a configurable provider context budget.

Example semantics:

```text
upstream_context_window_tokens: optional
context_safety_margin_tokens: configurable
```

Do not hard-code an unverified value.

## 10.1 Enforcement

Desired invariant:

```text
optimized_input_tokens
+ effective_max_tokens
+ safety_margin
<= configured_upstream_context_window
```

But do not force expensive exact counting onto every small request merely for this rule.

Design a low-overhead fast path.

Possible approach:

- ordinary requests well below the boundary do not synchronously exact-count;
- near the configured boundary, perform exact count in bounded `spawn_blocking`;
- do not create an unbounded tokenizer queue;
- if exact enforcement cannot be obtained cheaply, prefer a conservative, documented strategy.

## 10.2 Output adaptation

If input fits but the configured output cap would exceed remaining context, it is acceptable to reduce the effective output budget to remaining safe headroom, provided:

- the reduction is explicit in telemetry;
- a meaningful minimum output allowance remains;
- semantics are documented.

If the input itself leaves insufficient safe headroom, return a clear Anthropic-compatible context error BEFORE paying for an upstream generation.

Do NOT silently delete/summarize/truncate user messages, source code, tool evidence, or history in the proxy.

## 10.3 Claude Code compatibility

Research the current Claude Code behavior relevant to context overflow/auto-compaction.

If proxy-side context errors can trigger useful client compaction, verify it.

If Claude Code's configured 1M declaration itself must be reduced, document the exact client-side configuration needed rather than pretending the proxy can change the client's model metadata.

---

# 11. Phase 8 — real Claude Code E2E using the actual config

After deterministic tests pass, validate the actual chain.

Do this on an isolated local proxy instance using the real config path and secret-safe overrides.

At minimum:

## Case A — small streaming turn

```text
Claude Code -> Anthropic stream=true -> cline-proxy -> Cline -> GLM
```

Expect a normal completion/tool lifecycle.

## Case B — non-stream route

Exercise the existing upstream-stream/local-aggregate path and verify valid final Anthropic JSON.

## Case C — tool use

Perform a harmless small repository task requiring a read/tool operation.

Verify:

```text
tool_use
tool_result continuation
reasoning shadow continuity
no replay
```

Do not log tool arguments/content.

## Case D — real longer-context turn

Use a naturally available long Claude Code session or a controlled synthetic request within the established safe context range.

Verify:

```text
no ~600s unexplained stall
clear timeout if provider stalls
context diagnostics populated
cache telemetry remains correct
```

Do not intentionally hit the provider repeatedly once a failure boundary is known.

## Case E — restart logging

1. start isolated proxy;
2. produce at least one JSONL summary;
3. stop cleanly;
4. restart using same test log directory;
5. produce another summary;
6. verify no `writer closed`;
7. verify both old and new logs remain readable.

## Case F — redirected console

Run something equivalent to:

```powershell
.\cline-proxy.exe ... *> proxy-test.log
```

or the appropriate redirection for the chosen stderr/stdout design.

Verify byte-wise that no ANSI CSI escape sequence exists under `color=auto`.

---

# 12. Phase 9 — full repository audit for adjacent correctness problems

Do not interpret “fix the known bugs” as permission to stop auditing.

Review the code for related failures:

- all `unwrap`/`expect` in runtime paths;
- timeout configuration and overflow/zero handling;
- shutdown behavior of the log writer;
- disconnected log channel semantics;
- rotation/quota behavior after restart;
- partial JSONL writes;
- duplicate/stale active segment detection;
- Windows file handle behavior;
- large response aggregation limit;
- SSE decoder limits;
- tool-call accumulator correctness;
- usage-only chunks;
- reasoning/text/tool progress detection;
- non-stream aggregation memory bound;
- client disconnect cancellation;
- request body size limits;
- HTTP client reuse/keepalive;
- 429 classification;
- active-key restore/cooldown behavior;
- stale runtime-state handling;
- session fingerprint privacy;
- redaction of upstream errors;
- duplicated large `Value`/`Bytes` allocations;
- INFO logging that bypasses the one-summary policy;
- any accidental raw upstream body logging.

Fix only concrete problems supported by code/tests/evidence. Do not rewrite stable subsystems for style.

---

# 13. Logging implementation quality bar

Final logging behavior should look approximately like this interactively:

```text
12:30:01 INFO  cline-proxy  listening=0.0.0.0:8788 keys=12 model=glm53 logs=jsonl
12:30:18 ✓ GLM53 agent=0feedb6e key=gxe-outlook in=199.6K budget=216.0K cache=99.8% out=312 ttft=8.4s tool=9.1s dur=11.7s
12:32:54 ✗ GLM53 agent=0feedb6e key=gxe-outlook in=201.2K budget=217.6K semantic_idle=180.0s dur=193.4s err=upstream_semantic_idle_timeout
```

Exact formatting is up to you.

Default INFO must not dump 15–25 key/value fields across multiple wrapped lines.

`DEBUG` may expose safe technical fields.

JSONL should contain the complete safe machine-readable summary.

---

# 14. Performance/resource constraints

Do not “fix” observability by slowing the gateway.

Maintain:

```text
request hot path:
  no disk IO
  no await on log queue
  no per-chunk log serialization

log queue:
  bounded
  try_send/drop
  drop counter

writer:
  one dedicated thread
  buffered sequential IO

tokenizer:
  bounded spawn_blocking
```

New timeout/progress tracking should be O(1) per chunk.

Avoid `String` formatting on every chunk.

Context probing is a diagnostic tool and must not remain active in normal request flow.

Do not add a heavyweight metrics database.

---

# 15. Test matrix

Add focused deterministic tests covering at least:

```text
writer restart with underfilled existing segment
writer restart with full segment
writer restart after partial/corrupt final line
no ANSI when non-terminal/redirected
ANSI allowed only when explicitly terminal/forced
NO_COLOR / explicit never mode if implemented
INFO summary remains compact
local exact input tokens survive upstream failure
upstream usage remains distinct from local count
first raw event != first semantic event
first semantic timeout
stream idle timeout
semantic idle timeout despite heartbeat/nonsemantic frames
semantic timer resets on reasoning delta
semantic timer resets on text delta
semantic timer resets on tool-call delta
local Anthropic ping does NOT reset upstream timer
committed stream timeout never replays
non-stream aggregation timeout never triggers accidental duplicate generation
context budget arithmetic
configured context guard
output reserve reduction near context edge if implemented
pre-upstream context rejection when input cannot fit
no prompt/reasoning/tool content in logs
no raw secret/session id in logs
JSONL schema version
```

Use Tokio paused time where practical so timeout tests run instantly.

---

# 16. Benchmark expectations

No uncontrolled concurrent upstream load test is required.

Run local/release microbenchmarks for:

```text
1KB request
100KB request
1MB / 1.3MB request
SSE progress bookkeeping
RequestSummary construction
log try_send
writer sequential throughput
```

Compare before/after.

If logging/timeout bookkeeping adds >~5% to local hot-path benchmark, profile and fix it.

Do not fabricate <1% claims from noisy measurements.

---

# 17. GitHub workflow

Create one umbrella issue for production recovery, then split PRs if it improves review safety.

Suggested split:

```text
PR A: fix(obs): restart-safe writer + ANSI/console cleanup + diagnostic schema
PR B: fix(stream): explicit upstream/semantic stall detection
PR C: feat(context): evidence-backed provider context guard (ONLY if the real limit is established)
PR D: docs/test tooling: secret-safe live diagnostics and recovery docs
```

If context investigation proves no proxy context guard is necessary, do NOT create PR C just to match this plan.

Every PR:

```text
branch from current main
regression test first
focused tests
full gates
git diff main...HEAD
self-review
push
open PR
wait CI
fix findings
merge
pull latest main before next PR
```

Do not publish a release.

---

# 18. DEVELOPMENT_STATE.md requirements

At the end, rewrite the authoritative top snapshot to current main.

Include:

```text
current main SHA
production/runtime architecture
known Cline transport/context evidence
verified context limit OR explicitly "not established"
timeout policy and definitions
logging color policy
JSONL schema version
writer restart behavior
secret-safe testing protocol
all merged issue/PR numbers
full test counts
live E2E evidence
remaining known limitations
next action = normal production observation unless evidence shows another bug
```

Do not leave completed work in “Current target”.

---

# 19. Documentation requirements

Update as appropriate:

```text
README.md
config.example.json
docs/DEVELOPMENT_STATE.md
docs/OBSERVABILITY.md
docs/PERFORMANCE.md
docs/GLM53_FLASH.md
ADR(s)
```

Document config migrations/defaults.

If logging schema changes, document old/new meaning.

If the actual Cline context limit cannot be conclusively determined, say so clearly.

---

# 20. Final verification

From the final merged main:

```powershell
git checkout main
git pull --ff-only
git status

cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo test --release --workspace --all-targets --all-features
```

Verify no open task PRs/issues created by this work unless intentionally documented.

Verify no release/tag was created.

Run secret-safe final scan of tracked files and current diff.

---

# 21. Final report format

Return a factual report:

```text
PRODUCTION RECOVERY REPORT

Repository:
Final main SHA:
Workspace clean:

ROOT CAUSES
1.
2.
3.

REAL LOG AUDIT
requests analyzed:
slow/error distributions:
context correlation:
stream-stall evidence:
writer evidence:

CONTEXT CAPABILITY
client advertised:
model-native:
Cline catalog/provider:
empirically verified:
configured proxy limit:
confidence:

STREAM/TIMEOUT FIX
old behavior:
new behavior:
first-event timeout:
first-semantic timeout:
stream idle:
semantic idle:
committed-stream behavior:

LOGGING FIX
writer restart:
console:
ANSI:
INFO policy:
JSONL schema:
disk bounds:
failure behavior:

TOKEN/CONTEXT OBSERVABILITY
local exact tokens:
provider usage:
total budget:
utilization:

SECRET SAFETY
real config used:
raw secrets printed: NO
secret-bearing files committed: NO
temporary secret files remaining: NO

CLAUDE CODE LIVE E2E
small stream:
non-stream:
tool loop:
long context:
restart logging:
redirected logging:

PERFORMANCE
before:
after:
regressions:

TESTS
fmt:
check:
clippy:
debug:
release:
CI:

MERGED ISSUES/PRS
...

KNOWN LIMITATIONS
...

NOT VERIFIED
...

NEXT STEP
normal real Claude Code usage and evidence-driven follow-up
```

Do not claim that every future upstream problem is impossible. State exactly what is proven.

---

# 22. Completion rule

Do not stop after analysis.

Continue until all feasible work is:

```text
audited
reproduced
fixed
tested
validated with the real config without exposing secrets
reviewed
merged
documented
```

If a real external limitation prevents a fix (for example Cline itself cannot reliably serve the requested context), implement the best fail-fast/diagnostic behavior in the proxy, document the proven upstream limitation, and tell the user exactly what Claude Code/client context configuration should be changed.

Never hide an upstream limitation behind a ten-minute hang.
