# AGENTS.md — cline-proxy project rules

## Mission

`cline-proxy` is a small, production-oriented Rust gateway for Claude Code / Anthropic Messages clients using Cline as an upstream OpenAI-compatible transport, especially `z-ai/glm-5.3-flash`.

Priorities: protocol correctness and Claude Code usability; model/tool-loop capability; fail-safe upstream behavior; bounded latency/CPU/RAM/disk; privacy-safe observability; maintainability. Do not redesign working model behavior without production evidence.

## Mandatory startup workflow

At the beginning of every substantial task:

```powershell
git status
git branch --show-current
git remote -v
git fetch --all --prune
git log --oneline --decorate -30
gh repo view
gh pr list --state all
gh issue list --state all
```

Then read, when present:

```text
GROK_TASK.md
docs/DEVELOPMENT_STATE.md
docs/GLM53_FLASH.md
docs/OBSERVABILITY.md
docs/PERFORMANCE.md
docs/adr/
```

Never trust an old SHA or a previous agent report without checking the repository.

## Local production/runtime evidence

The user's real runtime directory is:

```text
D:\Workspace\cline-proxy-bin
```

Important evidence includes `config.json`, `runtime-state.json`, `proxy.log`, `logs\*.jsonl`, and `cline-proxy.exe`. Treat it as live evidence: do not overwrite, truncate, or mutate runtime evidence merely to make tests pass.

Use isolated ports, temporary state/log directories, and test processes for validation.

## Absolute secret-safety rules

`D:\Workspace\cline-proxy-bin\config.json` contains real credentials.

You MAY use the real config to execute local integration tests only when the task explicitly authorizes it, but:

- NEVER print, `type`, `cat`, `Get-Content`, echo, serialize, dump, paste, or otherwise expose the raw config.
- NEVER include API-key values, Authorization values, cookies, bearer tokens, raw session IDs, or full secrets in tool output, commits, PRs, issues, logs, screenshots, test snapshots, panic messages, or final reports.
- NEVER copy secret values into source code, Markdown, shell history, command-line arguments, tracked test fixtures, `.env` files in the repository, or generated prompts.
- NEVER ask a subagent to read the raw config.
- If config structure must be inspected, use a local sanitizer that emits only safe field names, booleans, counts, key *names*, lengths, or non-reversible fingerprints; do not emit secret values.
- Prefer passing `--config D:\Workspace\cline-proxy-bin\config.json` to the program so the binary reads the file locally.
- If a temporary secret-bearing config is absolutely unavoidable, create it only under a secure OS temp directory, restrict access when practical, never print it, and delete it in a `finally`/cleanup path. Prefer adding safe CLI overrides instead.
- Before every commit, verify secret-bearing runtime files are untracked/ignored.
- If a credential appears in Git history, report the exposure without reproducing the credential.

Keep secret/runtime classes such as `config.json`, `runtime-state.json`, `proxy.log`, `logs/`, `*-smoke-config.json`, `*-live-config.json`, and `*.secret.json` ignored. Do not blindly ignore all JSON.

## Existing model-path invariants

Unless a regression test proves they are wrong, preserve:

- explicit bounded GLM reasoning; do not allow omitted effort to silently become `max`;
- current reasoning epoch semantics: latest real human turn defines the epoch; `tool_result` continues that epoch;
- historical hidden reasoning is not replay-amplified;
- bounded memory-only reasoning shadow store; never persist hidden reasoning;
- unknown/unstable session identity disables cross-request shadow restore rather than guessing;
- stable prompt-prefix normalization and canonical structured tool arguments;
- user/tool/source text is never semantically truncated merely to reduce tokens;
- current output cap policy unless evidence requires a change;
- Cline downstream non-stream requests use one upstream streaming generation plus local aggregation;
- malformed 2xx/protocol errors are not API-key quota events;
- only effective HTTP 429 participates in key cooldown/failover;
- once a downstream stream is committed, never replay it on another key;
- prompt/source/tool output/reasoning content must never be written to observability logs.

## Performance and async rules

The request/SSE hot path must remain lean:

- no filesystem I/O;
- no blocking I/O on Tokio worker threads;
- no `await` on logging/telemetry queues;
- no unbounded queues/tasks/caches;
- no JSON log serialization per SSE chunk;
- no global mutex per SSE chunk;
- no whole-stream buffering for logging;
- no repeated large `serde_json::Value` clones without measured justification;
- no new HTTP client per request;
- no regex compilation in the request path;
- no per-request `fsync`;
- no synchronous compression in the request path.

CPU-heavy optional work belongs in bounded `spawn_blocking` or a bounded dedicated worker. Optional telemetry must degrade before inference performance degrades.

Every long-lived collection/queue/cache needs an explicit bound/TTL, full behavior, backpressure policy, and shutdown owner. If it is unbounded, fix the design.

## Logging rules

Use `tracing` for diagnostics and typed observability for request summaries.

Desired separation: interactive terminal -> compact human output; redirected terminal -> plain text with NO ANSI; persistent analytics -> bounded JSONL; detailed lifecycle -> DEBUG/TRACE.

Color must be `auto` by default: only emit ANSI when the output stream is an actual terminal. Respect `NO_COLOR` and/or an explicit no-color setting if implemented.

Do not persist a duplicate verbose text log when structured JSONL already carries the same information unless a concrete operational requirement exists.

Normal successful requests should produce one summary record, not many lifecycle records.

## Error classification

Do not collapse distinct failures into one generic error. Preserve safe categories such as:

```text
connect_timeout
first_event_timeout
first_semantic_timeout
semantic_idle_timeout
read_timeout
stream_transport
unexpected_eof
invalid_sse
upstream_protocol_error
upstream_application_error
context_budget_exceeded
effective_429
all_keys_rate_limited
client_disconnect
log_writer_failure
```

HTTP 200 means transport success only. Logical success requires a valid completion/stream.

## Context-window policy

Do not assume that Claude Code advertising `1M` proves Cline's deployed route has a 1M usable context window.

Distinguish:

```text
model-native context capability
client-advertised context
provider/catalog limit
actual serving-route limit
configured proxy safety limit
```

Do not hard-code a guessed Cline limit without evidence.

When context enforcement is added, it must be based on documented/provider metadata or controlled real probes. Prefer a configurable provider limit and safe telemetry over hidden semantic truncation.

## Testing rules

For each bug: add a regression test when practical, implement the smallest architecture-correct fix, run focused tests, then full gates.

Final gates:

```powershell
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo test --release --workspace --all-targets --all-features
```

Use release mode for performance measurements.

Do not run destructive or uncontrolled high-concurrency/load tests against the user's real upstream keys. Controlled single-session/live tests are allowed when explicitly authorized by `GROK_TASK.md`.

## Real integration-test rules

When authorized to test using the real `config.json`:

- the program may read it locally;
- Grok must not display its raw contents;
- prefer one selected healthy key if the product can do so safely;
- use tiny output budgets for diagnostics;
- isolate bind port, logs, runtime-state, and test artifacts from the production instance;
- do not kill or hijack an existing proxy process;
- do not overwrite the user's production executable until validation is complete;
- record only safe metrics: model, key name, status, token counts, timings, response shape, content-block types, tool-call count, cache counters, and error categories;
- never record completion text/reasoning/tool arguments during secret-safe live diagnostics.

## Git/GitHub workflow

Do not push directly to protected `main`.

Workflow: evidence -> issue -> branch -> regression tests -> implementation -> focused validation -> full gates -> self-review -> PR -> CI -> fixes -> merge.

Keep commits reviewable. Do not mix unrelated dependency upgrades or broad refactors into a correctness fix.

Before self-review:

```powershell
git diff main...HEAD
```

Review specifically for secret exposure, blocking work, unbounded memory, duplicated parsing/serialization, changed retry semantics, stream replay, and accidental prompt/reasoning logging.

## Documentation / recovery

`docs/DEVELOPMENT_STATE.md` must contain one authoritative current snapshot at the top. Historical material must be clearly non-authoritative.

Before context compaction or a long interruption:

1. update `docs/DEVELOPMENT_STATE.md`;
2. record branch, issue, PR, completed/pending work, tests, and real evidence;
3. commit or clearly record intentional uncommitted state.

After compaction:

1. reread this `AGENTS.md`;
2. reread `GROK_TASK.md`;
3. reread `docs/DEVELOPMENT_STATE.md`;
4. verify git/GitHub state;
5. continue without asking the user to repeat the task.

## Scope discipline

Do not replace evidence-driven engineering with speculative rewrites.

Do not claim all problems are fixed merely because tests are green. Distinguish deterministic regression fixes, controlled live verification, and production behavior not yet verified. Never fabricate performance, context limits, cache ratios, or upstream behavior.
