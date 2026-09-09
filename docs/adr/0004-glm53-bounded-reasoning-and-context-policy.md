# ADR 0004: Bounded reasoning and zero historical-thinking accumulation

Date: 2026-09-08 · Status: Accepted · Issue: #6

## Context

Real Claude Code → cline-proxy → Cline → GLM-5.3-Flash sessions showed:

- `request_bytes≈139 KB, TTFT≈8.6 s` followed by **518 s** single turns
  producing 7.5 MB of upstream SSE / 20k events;
- requests growing to `≈1.3 MB` as sessions progressed.

Root causes found in the code (audit of `main` @ `803d457`):

1. **unset → max.** `convert_request` emitted `reasoning_effort` only when
   the request carried `thinking`. Claude Code does not send `thinking`, so
   the OpenAI body had no effort, and the official GLM chat template
   coerces any effort outside `{low, high}` — including unset — to `max`
   (`chat_template.jinja`). Every ordinary coding turn was a max-effort
   turn.
2. **historical thinking amplification.** Assistant `thinking` blocks were
   re-emitted as `reasoning_content` on the wire *and* upstream reasoning
   was always exposed to the client as `thinking` blocks, which Claude Code
   stores and replays every turn — a positive feedback loop on context
   size.
3. **no output bound.** Client `max_tokens` was forwarded unbounded.
4. **no attribution telemetry.** Only total `request_bytes` was logged; it
   was impossible to say where tokens went.

## Decision

1. **One resolver.** `src/glm53/reasoning.rs::resolve_reasoning_policy` is
   the single source of truth mapping explicit controls + proxy defaults to
   (effort, exposure). Precedence: explicit `output_config.effort` >
   explicit `thinking` > default. Default is **`high`** — never `max`
   (rejected at startup validation) and not `low` (coding quality is a
   requirement). Every upstream request carries an explicit effort.
2. **Strip history, keep the chain.** `reasoning_content` is removed from
   assistant messages before the last user/tool-result turn
   (`strip_historical_thinking`, default on). Text, `tool_calls`, ids, and
   order are untouched; a dedicated test walks the tool-call chain. This
   mirrors the official template's `clear_thinking=true` semantics but is
   enforced locally (Cline forwarding of that flag cannot be relied on).
3. **Exposure gate.** `ThinkingExposure::RequestedOnly` (default): upstream
   reasoning becomes Anthropic `thinking` blocks only for requests that
   explicitly asked. `always` preserves legacy behavior; `never` is a hard
   off.
4. **Output cap.** `glm53.limits.max_output_tokens` (default 16,384):
   `effective = min(client, cap)`; an absent client bound becomes the cap.
5. **Safe compaction only.** Lossless structural normalization (single text
   block → string, empty blocks dropped, Anthropic-only `metadata`
   dropped). Tool results are never truncated; tool schemas are never
   edited; no LLM-side summarization — Claude Code owns context
   management.
6. **Telemetry.** Byte breakdown + policy decisions per request; output
   composition and first-reasoning/text/tool-call latencies per stream;
   exact GLM token accounting of the optimized request in a background
   task using the embedded official tokenizer (one full pass; the removed
   reasoning chunks are tokenized individually rather than re-counting the
   whole request twice).
7. **Windows parity fix.** `chat_template.jinja` is normalized to LF at
   include time and pinned via `.gitattributes`; CRLF checkouts had
   silently broken fixture parity (+5 tokens) on Windows.

## Alternatives considered

- **Default `max`:** reproduces the runaway; rejected.
- **Default `low`:** measurably weaker planning/debugging; rejected.
- **Relying on Cline to forward GLM's `clear_thinking`:** unverifiable;
  local stripping is deterministic.
- **Truncating tool results to shrink 1.3 MB requests:** destroys evidence
  the agent needs (diagnostics, file contents); explicitly rejected for
  this change.
- **Proxy-side conversation summarization:** competes with Claude Code's
  own compaction; rejected.
- **Request-body gzip:** network-only savings, unverified endpoint
  compatibility; deferred.

## Consequences

- Ordinary Claude Code turns run at `high` effort with bounded output:
  faster tool-call onset, no multi-minute reasoning runs by default.
- Historical reasoning contributes ~0 tokens to later turns (verified by
  the Case C integration test: identical counts with 0×/1×/40× thinking).
- Unexposed reasoning still costs its generation tokens on the current
  turn; the exposure gate only prevents re-payment, which is why it ships
  together with the effort cap.
- `count_tokens` now reflects the optimized request; the oracle-faithful
  count remains when the strip policy is disabled.
- Trade-off to watch: with `requested_only`, Claude Code no longer sees
  GLM's thinking for UX purposes unless it asks (`thinking.enabled`).
  Quality-of-work is unaffected; visibility is opt-in.

## Verification

- 123 unit/integration tests (debug + release), clippy `-D warnings` clean.
- Integration scenarios (`tests/glm53_policy.rs`): simple edit loop,
  compile-error debug loop, 24-turn session.
- Benchmarks (`cargo bench --bench request_optimization`): convert +
  optimize + serialize ≤ ~4 ms at ~1 MB; exact token count ~350 ms
  (background task in production).
