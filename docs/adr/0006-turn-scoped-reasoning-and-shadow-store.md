# ADR 0006: Turn-scoped reasoning and the reasoning shadow store

Date: 2026-09-09
Status: Accepted
Scope: `src/optimize.rs`, `src/reasoning_shadow.rs`, Anthropic handler
Related: issue #10, ADR 0004 (bounded reasoning), issue #6

## Context

ADR 0004 stripped historical reasoning before "the last user/tool turn".
That boundary conflated two distinct things:

- a **human user turn** (a real request — the reasoning-epoch boundary),
- a **tool result** (Anthropic `role=user` with only `tool_result` blocks,
  or an OpenAI `tool` message — a *continuation* of the current assistant
  reasoning epoch).

Consequences: reasoning within a tool loop was erased mid-epoch (the
model re-reasoned from scratch on every tool round-trip), while the
design intent — "current turn thinks fully, old turns never pay for
their internal reasoning" — was only half-achieved. Additionally, when
Claude Code does not request thinking (`requested_only` exposure), all
in-turn reasoning is discarded even though the *current* tool loop would
benefit from it.

## Decision

### Reasoning epochs

The strip boundary becomes the newest **human** user message: a `user`
message carrying any content other than `tool_result` blocks (OpenAI
side: the newest plain `user` message; `tool` role never starts an
epoch). Assistant reasoning before that boundary is stripped; reasoning
within the current epoch is preserved on the wire. This preserves
tool-chain integrity (ids, arguments, order) exactly as before.

### Reasoning shadow store

A bounded, memory-only store keyed by (HMAC session fingerprint,
tool-call id). When the client did not request thinking and the upstream
response carries reasoning + tool calls, the reasoning is stored; the
next request of the same session that replays a matching `tool` result
gets the reasoning attached to the assistant turn as `reasoning_content`
before the strip step runs (restored reasoning is in the current epoch,
so it survives).

Resource rules (every long-lived collection must have explicit bounds):

- max 256 sessions, 64 MiB total, 1 MiB per entry, 10-minute TTL;
- LRU/byte eviction; expired entries dropped lazily on access;
- oversized reasoning is **skipped, never truncated** — a truncated
  reasoning blob would be wrong context presented as real;
- no stable session identity (no `metadata.user_id`/`session_id`) → the
  store is disabled for that request entirely; there is no "most recent
  request" fallback, ever;
- final answer (no tool calls in the response) clears the session's
  entries; a new human turn clears the previous epoch;
- memory-only: a restart loses all state (accepted — ephemeral);
- one `Arc<str>` per assistant turn shared across its tool-call group.

### Isolation

Sessions are isolated by the HMAC fingerprint (issue #9's extraction).
Two Claude Code agents sharing one proxy, key, and model have different
fingerprints → different keys → zero cross-restore. Restores match exact
tool-call ids; a miss is silent, never a mis-attach.

## Alternatives considered

- *Preserve all reasoning forever*: context snowball, the original
  issue-#6 problem.
- *Strip all reasoning including the current epoch*: loses tool-loop
  continuity; the model re-reasons on every round-trip.
- *Persist the shadow store*: contradicts "reasoning is ephemeral";
  durability would wrongly make reasoning a first-class durable state.
- *Attach reasoning by position instead of tool-call id*: brittle under
  parallel tool calls and reordering; id-matching is exact.

## Consequences

- In-epoch reasoning continuity without ever exposing reasoning to the
  client; cross-epoch amplification stays at zero (tests enforce both).
- The store adds one `Mutex<HashMap>` guarded lookup per request; the
  lock is held for map operations only, never across awaits or network.
- Restored reasoning changes prompt bytes between turns of the same
  epoch; prefix-hash telemetry shows this — accepted trade (continuity
  within a turn beats prefix stability across it; the *prefix* sections
  — system, tools — remain stable).
- Claude Code still sees no thinking blocks (exposure stays
  `requested_only`); nothing to store, replay, or re-count.
