# ADR 0009: Session affinity via dynamic X-Task-ID

Date: 2026-09-11
Status: Accepted
Scope: `src/server.rs`, `src/upstream.rs`, `src/cache.rs`, `src/config.rs`, `src/anthropic.rs`
Related: ADR 0001 (key stickiness), ADR 0005 (prefix stability), ADR 0006 (shadow store)

## Context

Session identity was coupled to the `glm53.telemetry.prefix_hash`
observability flag:

```rust
let session_fp = telemetry.prefix_hash
    .then(|| extract_session_fingerprint(...))
    .flatten();
```

With telemetry disabled, the reasoning shadow store silently lost its
session key and no upstream session affinity existed at all — even though
Claude Code always carries a stable identity in
`metadata.user_id`/`metadata.session_id`. Cline's official client also
sends `X-Task-ID: <sessionId>` per conversation; without an equivalent,
key failover looked to upstream scheduling like unrelated traffic. The
identity extraction additionally re-parsed the full (megabyte-scale)
request body a second time just to read `metadata`.

## Decision

1. **One session identity.** Extracted unconditionally — never behind a
   telemetry flag — from `metadata.user_id` (preferred) or
   `metadata.session_id`, HMAC-SHA256'd with the server API key, truncated
   to 16 hex characters. Raw ids never reach logs or the upstream. No
   metadata → no identity; affinity is never guessed from connection, IP,
   key, request id, or recent requests.

2. **Shared fingerprint, independent switches.**

   ```text
   session identity (HMAC fingerprint)
   ├─ reasoning shadow        gated by glm53.reasoning.shadow_current_turn
   ├─ prefix/session telemetry gated by glm53.telemetry.prefix_hash
   │                          (now purely observational)
   └─ upstream X-Task-ID      always, when an identity exists
   ```

3. **Failover affinity.** `send_chat(..., task_id: Option<&str>)` passes
   the identity through the key loop; a `LogicalRequest` groups everything
   that must stay byte-identical across attempts (body, stream, request id,
   model, task id). Per attempt only Authorization rotates:

   ```text
   body1 == body2
   task_id1 == task_id2
   authorization1 != authorization2
   ```

   Only an effective HTTP 429 enters the failover loop (unchanged).
   The dynamic `x-task-id` is inserted after the statically configured
   headers, so it always wins.

4. **Reserved headers.** `x-task-id` and `x-request-id` are rejected in
   `upstream.headers` at startup: a static value could override (or spoof)
   the dynamic identity. Authorization remains sourced only from the
   selected key; arbitrary client headers are never forwarded (unchanged).

5. **Parse once.** The Anthropic handler parses the body once; identity
   extraction and `convert_request_value` share the same parsed value.

## Typed wire messages (protocol conversion, narrow cut)

Inspired by deepseek-recipe's typed adapter layer, the Anthropic→OpenAI
message conversion now builds a small typed IR (`WireMessage`: System /
User / Assistant / Tool) before serializing, instead of scattered `Map`
mutation. Two concrete wins:

- the assistant `tool_calls` ↔ `tool` `tool_call_id` chain has one
  checkable contract: a `tool_result` must reference an id declared by an
  earlier assistant message (cumulative set, in conversation order) —
  parallel results are matched by ID, never by position; an orphan result
  is an explicit `invalid_request_error` instead of a dangling upstream
  id that fails later with a worse error;
- field order in `into_wire` reproduces the historical wire bytes exactly
  (prefix stability depends on it); all pre-existing conversion tests pass
  byte-identically.

## Consequences

- `prefix_hash: false` no longer degrades shadow continuity or affinity
  (regression-tested both).
- A configured `x-request-id`/`x-task-id` in `upstream.headers` now fails
  startup validation (behavior change; neither appears in the shipped
  example or production configuration).
- Session affinity end-to-end (does the Cline route actually sticky-route
  on `X-Task-ID`?) still needs one controlled live verification; the proxy
  side is covered by mocks only.
