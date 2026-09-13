# cline-proxy OpenAI Responses API frontend (Grok Build)

## Goal

Repository:

```
https://github.com/jacek4yang/cline-proxy
```

Primary reference implementation:

```
https://github.com/jacek4yang/codebuddy-proxy
```

Implement production-quality OpenAI Responses API (`POST /v1/responses`)
compatibility so **Grok Build** (`api_backend = "responses"`) can run against
this gateway, using Cline as the upstream OpenAI-compatible transport.

The target flow is:

```text
Grok Build
  -> POST /v1/responses
  -> Responses request normalization (input items, tools, tool_choice,
     reasoning effort, max_output_tokens, prompt_cache_key)
  -> OpenAI Chat Completions body (same shape the Anthropic frontend emits)
  -> GLM policy (optimize_request) + reasoning shadow + prefix telemetry
  -> Cline upstream (same pool / 429 failover / route invariants)
  -> upstream chat SSE
  -> Responses SSE stream converter (typed frames, sequence numbers)
  -> Grok Build receives a valid Responses stream / response object
```

This task is the Responses frontend only.

Do not redesign the Anthropic frontend, WebSearch server-tool loop, reasoning
policy, cache behavior, key pool, 429 classification, or routing.

## Required workflow

Read `AGENTS.md`, relevant docs, current source, tests, and the latest `main`
before editing.

Port the proven architecture from `codebuddy-proxy`:

- `src/responses/request.rs` — Responses → chat conversion;
- `src/responses/stream.rs` — chat SSE → Responses SSE converter;
- `src/responses/types.rs` — wire types, usage mapping, IDs;
- `src/server/responses_pump.rs` — streaming pump + non-stream aggregation;
- the `/v1/responses` handler in the orchestrator (session extraction from
  `prompt_cache_key` / `x-grok-conv-id`, error envelope, observability).

Adapt them to `cline-proxy` instead of copying CodeBuddy-specific policy:

- session identity is fingerprinted with `cache::session_fingerprint`
  (domain-tagged so Responses fingerprints never collide with Anthropic
  ones); the raw key is never logged or forwarded;
- the GLM policy step (`optimize::optimize_request` with
  `Origin::OpenAi`) runs on the converted chat body — explicit reasoning
  effort, bounded output, historical-thinking strip, safe compaction,
  canonical tool JSON, prefix telemetry all apply unchanged;
- reasoning shadow restore/store follows the same epoch rules as the OpenAI
  chat path (`request_starts_new_epoch`);
- streaming uses the same `StreamWatch` timeouts, ping keepalive, no-replay
  discipline, and bounded observability as the other frontends;
- non-stream uses ONE upstream streaming generation aggregated locally
  (issue #14 strategy) — never a second generation for shape conversion;
- errors use the OpenAI error envelope (the Anthropic envelope would fail
  Grok Build's error parser).

## Protocol scope (from the verified reference)

Request conversion:

- `input` string → one user message; item array converts in order;
- `message` items: roles system/developer/user/assistant; content parts
  `input_text`/`output_text`/`refusal` joined in order; images rejected;
- `function_call` history → assistant `tool_calls` with the SAME `call_id`;
  consecutive calls of one assistant turn merged into one message;
- `function_call_output` → `role=tool` with `tool_call_id = call_id`;
- historical `reasoning` items dropped (never replayed onto the wire);
- `item_reference` rejected (stateless proxy);
- flat function tools → nested OpenAI function tools, schemas byte-identical;
- hosted backend tools (`web_search`, `x_search`, …) dropped, never
  forwarded, never a hard failure;
- `max_output_tokens` → `max_tokens` (never dropped);
- `temperature`, `top_p` copied; `tool_choice` string modes and
  `{type: "function", name}` mapped; `prompt_cache_key` forwarded verbatim;
- `reasoning.effort` maps none/minimal/low→low, medium/high/xhigh/max→high;
  `reasoning.summary` present → reasoning exposure requested (the
  `requested_only` gate).

Stream events (exact typed shapes, monotonic `sequence_number`):

- `response.created`, `response.in_progress`;
- `response.output_item.added` (reasoning) + summary part/text deltas +
  done frames — only when exposure was requested;
- `response.output_item.added` (message), `response.content_part.added`,
  `response.output_text.delta`, `response.output_text.done`,
  `response.content_part.done`, `response.output_item.done`;
- `response.output_item.added` (function_call) ALWAYS precedes its
  `response.function_call_arguments.delta` / `.done` frames;
- terminal `response.completed` | `response.incomplete` (length) |
  `response.failed` (midstream error, never a replay);
- keepalive is an SSE comment (`: ping`), never a synthetic event frame.

Terminal response object reconstructs the full output (Grok Build replays it
as next-turn conversation state) with `usage` where Responses
`input_tokens` is the FULL prompt count and cached tokens are a subset.

## Server wiring

- Route `POST /v1/responses` behind the same auth middleware.
- `response_log_middleware` protocol label: `responses`.
- Observability: one summary per request; `protocol: "responses"`.
- Model resolution via `models.default` + `models.aliases`, same as chat.

## Tests

Add focused coverage (deterministic, offline, mocked loopback upstream):

- request conversion (string input, structured items, tools, hosted-tool
  drop, reasoning effort mapping, tool history round-trip, unknown items);
- stream lifecycle (text, tools item-added-before-deltas, reasoning
  requested-only, usage mapping, incomplete, response.failed);
- server integration: auth, model alias, stream and non-stream through the
  real handler with a mocked upstream, shadow restore/store interaction,
  429 behavior unchanged, no prompt content in logs.

## Safety invariants

- No prompt/reasoning/tool content in logs.
- Once a downstream stream is committed, never replay on another key.
- Only effective HTTP 429 participates in key failover.
- No unbounded buffers; same hot-path rules as the other frontends.
- Raw session keys (`prompt_cache_key`, conv ids) are fingerprinted before
  any use and never logged.

## Quality gates

Run:

```powershell
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo test --release --workspace --all-targets --all-features
```

No ignored failures.

## Workflow

issue -> `feat/responses-api` branch -> implementation -> tests -> full
gates -> self-review -> PR -> CI -> merge when green.

Do not create a GitHub Release or tag. Do not modify or deploy to
`D:\Workspace\cline-proxy-bin` in this task.

Live validation against the real Cline credential is authorized only with
isolated bind port / state / logs, tiny output budgets, and no secret
printing (per AGENTS.md rules).

## Definition of done

- Grok Build can point `OPENAI_BASE_URL` at the gateway and run a session
  (streaming and non-streaming) with function tools;
- responses are valid typed Responses events / response objects;
- all tests, gates, CI pass;
- README and `docs/DEVELOPMENT_STATE.md` updated;
- PR self-reviewed and merged.
