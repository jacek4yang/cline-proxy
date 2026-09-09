# ADR 0005: Claude Code prompt stability and model capabilities

Date: 2026-09-09
Status: Accepted
Scope: `src/cache.rs`, Anthropic system normalization, stream telemetry
Related: issue #8, ADR 0004 (GLM bounded reasoning), issue #6

## Context

Upstream prompt caches are keyed on byte-exact prefixes. A coding-agent
session repeats almost all of its prompt every turn (system, tools,
history); any byte difference in that repeated section forfeits the cache
and re-pays full prefill. Three sources of per-turn byte drift are
observable in Claude Code traffic:

1. Claude Code prepends an `x-anthropic-billing-header:` line to the
   system text whose attribution metadata (`cch`, timestamps) varies
   between requests of the *same* conversation.
2. Tool-call arguments are JSON strings on the OpenAI wire; equivalent
   objects with different key insertion order serialize to different
   bytes.
3. Nothing currently measures whether the locally-built prefix is stable,
   so drift would be invisible until an upstream cache is probed.

## Decision

- **Strip only a *leading* `x-anthropic-billing-header:` line** from the
  system text, in Anthropic system normalization (so the wire body,
  count_tokens, and telemetry see identical normalized content). Never a
  full-text search: a header-shaped line authored later by the user or the
  model must survive. LF/CRLF/CR are all handled.
- **Canonicalize historical tool-call argument JSON** (deterministic,
  sorted-key serialization) *only* for `tool_calls[].function.arguments`
  strings and only for turns before the last user/tool message. Arrays
  keep order; numbers/strings are emitted verbatim; malformed argument
  strings pass through untouched. Plain-text tool results, source code,
  shell output, and diagnostics are never parsed or rewritten.
- **Stable prefix hash telemetry**: SHA-256 over the normalized
  system + messages + tools sections, logged as `prefix_hash` +
  `prefix_bytes`. This is a *local stability metric only* — equal hashes
  never prove an upstream cache hit; only upstream `cached_tokens` can.
- **Session fingerprint**: when `metadata.user_id`/`metadata.session_id`
  is present, an HMAC-SHA256 (server secret) of the raw id is logged as a
  16-hex-char `session` field. Raw ids are never logged or forwarded.
  Without a stable identity the field is `unstable` — no fallback
  guessing from connection state, keys, or recent requests.
- **Cache ratios from real usage**: `cache_hit_ratio =
  cached_tokens / prompt_tokens` and `reasoning_ratio =
  reasoning_tokens / completion_tokens` are logged only from figures the
  upstream actually reports. Documented OpenAI semantics (cached_tokens
  is a subset of prompt_tokens) are assumed and annotated in code; this
  is to be verified against real Cline usage before any conclusion.

All new behavior is config-gated under `glm53.context.*` and
`glm53.telemetry.*`, defaulting on, and model-scoped by the PR #7
`ModelFamily` split (non-GLM models never receive GLM semantics).

## Alternatives considered

- *Full system canonicalization* (merging all system text blocks into
  one string): deferred — it changes tokenization of the system section
  and must first be validated against the GLM template's handling of
  multiple system messages. The current conversion already emits a single
  system message for Anthropic block-array systems, so the practical
  drift source is the billing header, not message structure.
- *Session-scoped `prompt_cache_key` upstream*: not implemented. Whether
  the Cline endpoint accepts the parameter is unverified; sending an
  unknown parameter risks 400s. The field stays out of the wire until a
  deliberate capability probe proves support.
- *Deleting all Anthropic metadata earlier*: rejected — metadata carries
  the session identity needed above. Extraction happens before the GLM
  policy drops it from the wire.

## Consequences

- Two requests that differ only in billing-header metadata or argument
  key order produce byte-identical system sections and equal prefix
  hashes (enforced by tests, including an end-to-end router test).
- Hashing adds one serialization of the three request sections per
  request. At 1 MB this is milliseconds-scale and off the tokenizer path;
  benchmarks guard the regression.
- Cache-hit claims remain unverified until a real Cline A/B run measures
  `cached_tokens` with stripping on vs. off; the telemetry makes that
  run interpretable.
