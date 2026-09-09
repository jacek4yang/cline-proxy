# Development State

Last updated: 2026-09-09
Main SHA: 72c4426 (feat(obs): adaptive bounded observability #17)
Repository: https://github.com/jacek4yang/cline-proxy
Status: main green; all roadmap phases through observability merged

> Agent recovery protocol — on context compaction or session end:
> 1. Read this file top to bottom.
> 2. `git status && git branch --show-current && git pull --ff-only`.
> 3. Verify main SHA above against `git log --oneline -1`.
> 4. `gh pr list --state open && gh issue list --state open`.
> 5. Continue from "Current target". NEVER trust branch names or
>    "pending merge" phrases from anything below the Historical record.

## Current production baseline

- main SHA `72c4426`; 186 tests green (debug + release); fmt/clippy
  `-D warnings` clean.
- Real E2E (2026-09-09, live Cline + Claude Code, ~300 K-token session):
  99.8–100.0% warm cache hit ratio, TTFT ≈ first-tool-call, cold prefill
  ~89 s → cached turns ~10 s, reasoning tokens 0–436/turn under `high`
  effort. Table in the historical record below and issue #8.
- Proxy hot path (release, 1.3 MB request): convert 0.9 ms, optimize
  1.2 ms, cache pass 1.2 ms, serialize 0.5 ms. Exact tokenizer (≈400 ms)
  runs spawn_blocking + semaphore(1) + busy-skip, off the request path.
- Production deployment: `D:\Workspace\cline-proxy-bin` (binary updated
  per release; config with explicit glm53 policy + logging block).
- Release: INTENTIONALLY DEFERRED (no tags, no GitHub Release, no
  binaries published).

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
  → Adaptive observability (obs.rs): ONE summary/request
      bounded queue → dedicated writer thread → JSONL (1 GB quota)
      RAM flight recorder attached to anomalies only
```

Modules: `anthropic.rs` (protocol), `cache.rs` (prefix stability),
`optimize.rs` (model policy), `reasoning_shadow.rs` (ephemeral reasoning),
`obs.rs` (summaries/writer), `pool.rs`+`state.rs` (keys), `glm53/*`
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

Issues #3, #6, #8, #10, #13, #14, #16 closed with their PRs.

## Open PRs / issues

- PR #1 (dependabot): actions/checkout 4→7 — STALE (BEHIND), handled in
  the checkout phase (merge or supersede via fresh branch).
- PR #2 (dependabot): axum 0.7.9→0.8.9 — STALE (BEHIND), must NOT be
  merged directly; dedicated migration branch planned.
- No open issues.

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

## Current target

System-hardening sequence, in order (each phase: issue → branch → PR →
CI → self-review → merge → pull main):

1. ✅ P0 Cline non-stream correctness (PR #15).
2. ✅ Phase A adaptive observability + CPU/RAM bounds (PR #17).
3. ✅ Phase B: DEVELOPMENT_STATE normalization (this PR).
4. Phase C: actions/checkout v4→v7 (merge PR #1 after rebase or
   supersede via `ci/checkout-v7`).
5. Phase D: axum 0.7→0.8 migration (`chore/axum-0.8-migration`), minimum
   diff, full regression on streaming/auth/429/shutdown, then supersede
   PR #2.
6. STOP (release deferred).

## Next tasks (after Phase D)

- Observe real production logs (`logs/events-*.jsonl`); analyze only
  when evidence shows a problem.
- Do NOT proactively redesign the model path (feature freeze on
  reasoning/cache/prefix/routing semantics — all verified).

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
