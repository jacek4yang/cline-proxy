# ADR 0009: Session affinity via credential-scoped X-Task-ID

Date: 2026-09-11
Status: Accepted (revised during stabilization close-out)
Scope: `src/server.rs`, `src/upstream.rs`, `src/cache.rs`, `src/config.rs`, `src/anthropic.rs`
Related: ADR 0001 (key stickiness), ADR 0005 (prefix stability), ADR 0006 (shadow store)

> Revision note: the first revision of this ADR sent the internal session
> fingerprint unchanged as `X-Task-ID` across credential failover. The
> final design scopes the upstream-visible task id to the selected
> credential (see "Upstream task identity" below); the internal session
> identity semantics are unchanged.

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

2. **Two identities: internal session identity vs upstream task identity.**

   ```text
   Claude metadata (user_id/session_id)
         ↓
   internal session_fp = HMAC-SHA256(server_secret, raw_id)[..16 hex]
         │  cross-credential stable; NEVER sent upstream directly
         ├─ reasoning shadow        gated by glm53.reasoning.shadow_current_turn
         ├─ prefix/session telemetry gated by glm53.telemetry.prefix_hash
         │                          (purely observational)
         │
         └─ upstream task identity, per selected credential:
            X-Task-ID = HMAC-SHA256(
                server_secret,
                "cline-proxy/x-task-id/v2\0" || session_fp || "\0" || actual_cline_api_key
            )[..16 hex]
   ```

   **Internal session identity** — privacy-safe HMAC fingerprint, stable
   across Cline credentials; keys the reasoning shadow store and local
   telemetry; restart-stable while the server secret and session id are
   unchanged.

   **Upstream task identity** — derived from (internal session identity,
   actual selected Cline credential) via a domain-separated HMAC. Stable
   within one credential, different across credentials. Sent as
   `X-Task-ID`. The derivation binds the actual credential secret, not
   the key's config name, so a rotated credential changes the id; raw
   session ids and raw API keys are never embedded (this is a keyed PRF
   under the server secret, not an unsalted hash of the key).

   Rationale: the proxy needs cross-key continuity internally (shadow
   restore, telemetry), but the upstream does not need an explicit
   cross-key task identifier. Credential-scoping removes an unnecessary
   direct linkage signal — the proxy no longer hands Cline the same task
   id under different API keys — while preserving stable task identity
   within each credential. **This is privacy minimization, not an
   unlinkability guarantee**: Cline may still correlate requests through
   request body, timing, IP, model, prompt prefix, tool definitions, or
   account ownership.

3. **Failover semantics.** `send_chat(..., session_fp: Option<&str>)`
   passes the internal identity through the key loop; `LogicalRequest`
   groups everything that must stay byte-identical across attempts (body,
   stream, request id, model). The X-Task-ID is derived inside
   `send_once` AFTER the `SelectedKey` exists, so per attempt:

   ```text
   body1 == body2
   authorization1 != authorization2
   x_task_id1 != x_task_id2
   x_task_id1 == derive(session_fp, key A)
   x_task_id2 == derive(session_fp, key B)
   ```

   Only an effective HTTP 429 enters the failover loop (unchanged).
   The dynamic `x-task-id` is inserted after the statically configured
   headers, so it always wins. The derivation is stateless (one HMAC per
   attempt — no cache, no LRU, no mutex) and therefore deterministic and
   restart-stable.

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
  checkable contract with Anthropic tool-use adjacency semantics: a
  `tool_result` must live in the message immediately following its
  tool_use (the pending window holds only the latest assistant turn's
  declared ids; any user/system message closes it), tool_result blocks
  must come first in their user message (ordinary content after them is
  kept; content before them is an explicit `invalid_request_error`), and
  parallel results are matched by ID in any order — never by position.
  An orphan or stale result is an explicit `invalid_request_error`
  instead of a dangling upstream id that fails later with a worse error;
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
