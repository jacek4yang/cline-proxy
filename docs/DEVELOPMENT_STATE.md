# Development State

Last updated: 2026-09-10
Main SHA: 1191a8a (docs snapshot #26); SOCKS5 work on `feat/upstream-socks5` (issue #27)
Repository: https://github.com/jacek4yang/cline-proxy
Status: SOCKS5 + deterministic direct route in progress

> Agent recovery protocol — on context compaction or session end:
> 1. Read this file top to bottom.
> 2. `git status && git branch --show-current && git pull --ff-only`.
> 3. Verify main SHA above against `git log --oneline -1`.
> 4. `gh pr list --state open && gh issue list --state open`.
> 5. Continue from "Current target". NEVER trust branch names or
>    "pending merge" phrases from anything below the Historical record.

## Current production baseline

- main `18e7dcb` (axum 0.8.9 + production recovery). Adds stall timers,
  restart-safe JSONL writer, stderr/auto-color, JSONL schema v2, local vs
  upstream token fields, optional context guard (unset by default).
- Tests on the recovery branch: 202 lib + 4 glm53_policy + 4
  reasoning_shadow = **210** (debug and release); fmt/clippy `-D warnings`
  clean.
- Release hot path (1.3 MB): convert 1.0 ms, optimize 1.1 ms, cache 1.1 ms,
  serialize 0.5 ms — no >5% regression vs the prior 0.9/1.2/1.2/0.5 ms
  baseline. Exact tokenizer ~411 ms on that fixture, still spawn_blocking.
- Isolated live Cline (2026-09-10, real `config.json`, bind 127.0.0.1:18799,
  no secrets printed): stream 200 + deltas + stop; non-stream 200
  `end_turn`; tool_use 200 with 2 tool-call events, thinking suppressed;
  JSONL schema_version=2; writer restart opened a fresh segment and left
  old records intact; redirected console CSI count = 0.
- Context: model-native 1,048,576; Cline catalog unknown; empirical Cline
  success ≥ 453,504 prompt tokens (production JSONL). Proxy window
  **not configured**.
- Timeout policy: first_event 180s, first_semantic 180s, stream_idle 120s,
  semantic_idle 180s; Reqwest read_timeout remains 600s as backstop.
- Production deployment: `D:\Workspace\cline-proxy-bin`. Do not overwrite
  the production exe until this recovery is merged and the isolated live
  binary is copied deliberately.
- Release: INTENTIONALLY DEFERRED (no tags, no GitHub Release).

## Current architecture

```text
Claude Code
  → Anthropic Frontend (anthropic.rs)
      volatile billing-header strip · session fingerprint (HMAC)
      reasoning-shadow restore (epoch-aware)
  → GLM policy (optimize.rs; ModelFamily-scoped)
      explicit reasoning_effort (never unset→max) · output cap 16384
      historical-thinking strip (epoch boundary = newest HUMAN turn)
      safe compaction · canonical tool JSON
  → Cline upstream (upstream.rs; strict sticky keys, effective-429 only)
      downstream stream=true  → upstream stream=true → SSE translate
      downstream stream=false → upstream stream=true → local aggregation
        (strict envelope normalizer; issue #14)
  → SSE exposure gate (requested_only) · shadow store commit
  → StreamWatch (first-event / first-semantic / stream-idle / semantic-idle)
      committed stall → one Anthropic error, never replay
  → Adaptive observability (obs.rs schema v2): ONE compact INFO + JSONL
      fresh segment per process · stderr · color=auto
      RAM flight recorder attached to anomalies only
```

Modules: `anthropic.rs` (protocol), `cache.rs` (prefix stability),
`optimize.rs` (model policy), `reasoning_shadow.rs` (ephemeral reasoning),
`obs.rs` (summaries/writer), `stream_watch.rs`, `console.rs`,
`context_guard.rs`, `pool.rs`+`state.rs` (keys), `glm53/*`
(exact tokenizer/template).

## Completed milestones

| PR | Milestone |
|---|---|
| #4 | Persistent key runtime state (atomic JSON, restart skips cooling keys) |
| #5 | Exact GLM-5.3-Flash tokenizer (official assets, Rust==oracle parity) |
| #7 | Bounded reasoning (high default, never unset→max), model-scoped policy, bounded spawn_blocking token telemetry |
| #9 | Prompt-prefix stability (billing-header strip, canonical tool JSON, prefix hash, session fingerprint) |
| #11 | Reasoning epochs (human-turn boundary) + bounded reasoning shadow store |
| #12 | Real E2E cache baseline (99.8–100% warm hits) |
| #15 | P0 Cline non-stream fix (stream-and-aggregate; strict envelope normalizer; malformed-200 never rotates keys) |
| #17 | Adaptive bounded observability (one summary/request, dedicated JSONL writer, disk quota, flight recorder) |
| #20 | axum 0.7.9 → 0.8.9 |
| #24 | Production recovery: stall timers, restart-safe writer, ANSI/INFO, schema v2 (this work) |

Issues #3, #6, #8, #10, #13, #14, #16, #19 closed with their PRs.

## Open PRs / issues

- Dependabot: hmac 0.13, sha2 0.11, tokenizers 0.23 — not part of this
  recovery; leave for a later dedicated bump.
- No open recovery issues.

## Known limitations

- Cline native non-stream bodies are not reliably standard OpenAI
  (production evidence); the proxy therefore always streams upstream and
  aggregates locally for non-stream clients. The strict envelope
  normalizer covers root-`choices` and the known `success/data` envelope;
  unknown shapes are a safe 502 (no key rotation), not a guess.
- The reasoning shadow store is memory-only: a proxy restart drops
  in-epoch reasoning continuity (accepted; reasoning is ephemeral).
- All Cline keys share daily quotas; when every key is cooling the proxy
  returns a semantic 429 with Retry-After (by design).
- No formal Release yet; release workflow does not exist. Deferred on
  purpose.
- Real multi-agent concurrency/resource load analysis intentionally
  deferred: production traffic + JSONL summaries are the evidence source.
- Cline serving-route context window is **not established**. Do not set
  `upstream_context_window_tokens` until catalog/probe evidence exists.
  Claude Code's 1M client declaration is not a Cline route guarantee.
- Isolated live tests used HTTP against the local proxy, not an
  interactive Claude Code TUI session.

## Current target

1. Merge issue #27 SOCKS5/direct route.
2. Isolated A/B (n=4 tiny stream turns, real config, no secrets printed):
   direct headers median 1284 ms (685–4887); SOCKS5 headers median 1054 ms
   (534–1706). No transport errors. Overlap is large — do **not** claim
   SOCKS5 is faster. Local SOCKS5 is a viable Cline path.
3. After merge: timestamped config backup, set `upstream.proxy` to
   `socks5://127.0.0.1:10888`, backup+replace production exe, smoke.

## Next tasks (after this recovery)

- Observe real production logs (`logs/events-*.jsonl`); analyze only
  when evidence shows a problem.
- Do NOT proactively redesign the model path (feature freeze on
  reasoning/cache/prefix/routing semantics — all verified).
- Optional: set `glm53.context.upstream_context_window_tokens` only after
  a real Cline route limit is measured.

## Explicitly deferred

- Release/tags/binaries (intentionally).
- New concurrency/multi-agent load tests (intentionally).
- prompt_cache_key (unnecessary: 99.8–100% hits without it).
- Request gzip, deferred tools, adaptive effort heuristics (no evidence).

## Agent recovery protocol

See the blockquote at the top. In short: this file's top section is the
only authoritative state; everything under "Historical record" is
append-only archaeology and must not drive decisions.

## Historical record (NOT CURRENT STATE)

Everything below predates the normalization. Kept for provenance only.

---

### Real E2E baseline (2026-09-09, live Cline + Claude Code)

Observed on the production proxy running main `954c59a`, serving a real
Claude Code session with ~300 K-token context. Upstream **does** report
usage with `cached_tokens`; semantics confirmed as
`cached_tokens ⊆ prompt_tokens`.

| turn | prompt_tokens | cached_tokens | cache_hit_ratio | duration_ms | first_tool_call_ms | reasoning_tokens |
|------|--------------:|--------------:|----------------:|------------:|-------------------:|-----------------:|
| 1 (cold) | 307,678 | 0 | 0.0% | 92,376 | 88,789 | 114 |
| 2 | 308,067 | 307,648 | 99.9% | 19,042 | 19,036 | 255 |
| 3 | 308,470 | 308,032 | 99.9% | 12,167 | 12,167 | 0 |
| 4 | 308,751 | 308,416 | 99.9% | 31,075 | 31,075 | 436 |
| 5 | 309,463 | 308,736 | 99.8% | 9,973 | 9,973 | 0 |
| 6 | 309,587 | 309,440 | 100.0% | 10,312 | 10,312 | 0 |

- Prefix stability works at production scale; `prompt_cache_key` is
  unnecessary and stays unsent.
- reasoning_tokens bounded under `high` effort — no runaway.
- Non-stream upstream quirk [observed, resolved by PR #15]: Cline
  returned 200 with a non-choices body; now aggregated via upstream
  streaming.

### Historical milestones (pre-normalization, for provenance)

- PR #4 persistent key runtime state: `state.rs` v1 schema, atomic
  tmp+rename, `pool.rs` Healthy/Cooling/HalfOpen single-flight probe,
  debounced writer, restart regression test (5×429 → key6 → restart →
  first-attempt success). ADR 0001.
- PR #5 exact tokenizer: official assets pinned @ `eb9eb20` (MIT),
  Rust == Python oracle byte-for-byte across 15 golden fixtures,
  `/v1/messages/count_tokens` exact, CRLF parity via `.gitattributes`
  + include-time normalization. ADR 0003.
- PR #7 bounded reasoning: `resolve_reasoning_policy` single source,
  safe compaction, output cap, Windows CRLF fixture fix. ADR 0004.
- PR #9 prefix stability: `cache.rs` (billing-header strip LF/CRLF/CR,
  canonical JSON, SHA-256 prefix hash, HMAC session fingerprint).
  ADR 0005.
- PR #11 reasoning epochs + shadow store: human-turn boundary, bounded
  memory-only store (256 sessions / 64 MiB / 1 MiB / 10 min TTL),
  multi-agent isolation tests. ADR 0006.
- PR #15 non-stream fix: `NonStreamAccumulator` (byte-exact aggregation,
  parallel tool calls, usage last-write-wins, 16 MiB bound), strict
  envelope normalizer, transport-vs-logical log fix.
- PR #17 adaptive observability: `obs.rs` summary pipeline, writer
  thread, quotas, flight recorder. ADR 0007.
