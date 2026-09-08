# GLM-5.3-Flash Reference (Official Provenance)

This document records the canonical, official sources used for exact token
counting and chat-template semantics. Evidence tags:
**[official]** = upstream repo/HF, **[observed]** = our own runs,
**[report]** = third-party (not a guarantee), **[bench]** = our benchmark.

## Model identity

- Identifier: `z-ai/glm-5.3-flash` **[official]**
- Hugging Face: https://huggingface.co/zai-org/GLM-5.3-Flash
- Revision used (pinned): `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`
- Retrieved: 2026-09-08 · License: MIT **[official]**
- Local metadata cache: `tools/glm_reference/assets/model_api.json`

## Official assets used (tokenizer only — no model weights)

| File | Size | Purpose |
|---|---|---|
| `tokenizer.json` | 20,217,442 B | BPE vocab 154,820 + 36 added special tokens |
| `tokenizer_config.json` | 761 B | tokenizer class metadata |
| `chat_template.jinja` | 10,950 B | official chat template |
| `config.json` | 69,416 B | model config (context window metadata) |
| `generation_config.json` | 194 B | generation defaults |
| `processor_config.json` | 909 B | multimodal processor config |

## Chat template semantics (from official `chat_template.jinja`) [official]

- Prefix `[gMASK]<sop>`; roles `<|system|>`, `<|user|>`, `<|assistant|>`,
  `<|observation|>`; tool calls `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>`;
  tool results `<|observation|><tool_response>…</tool_response>`.
- `reasoning_effort` template variable: only `low` and `high` are honored;
  **anything else (including unset and `medium`) renders as `max`** via
  `<|system|>Reasoning Effort: …` preamble.
- `clear_thinking` template variable: **defaults to `false`**. When `true`,
  assistant `reasoning_content` is stripped (replaced by `<think></think>`)
  for all assistant turns *before the last user message*; the final
  assistant turn's reasoning is always kept.
- Tool schemas are serialized with `tojson(ensure_ascii=False)`; the
  template **drops `strict` and `defer_loading` keys** from tool objects.
- Tool-call `arguments` must be a JSON **object** (the template iterates
  `.items()`); string values are passed through raw, other types are
  JSON-encoded.
- Tool results are re-sorted to match the preceding assistant `tool_calls`
  order when all IDs are unique and matchable.
- Images render as `<|begin_of_image|><|image|><|end_of_image|>` (base64
  payload never enters the text token stream); video/audio analogous.
- Assistant message text is `.strip()`ed; `<think>…</think>` inline in
  content is parsed as reasoning when `reasoning_content` is absent.
- `add_generation_prompt` appends `<|assistant|><think>`.

## Reasoning mapping policy (cline-proxy)

The official template coerces any effort outside {low, high} to `max`, but
the GLM API itself accepts `low`, `high`, and `max`. Every request leaving
cline-proxy carries an **explicit** `reasoning_effort` — never unset —
because "send nothing" silently selects `max` **[official template
behavior]**, the direct cause of multi-minute reasoning runaways observed in
production (issue #6). Single source of truth:
`src/glm53/reasoning.rs::resolve_reasoning_policy`.

Precedence (fixed, tested): explicit `output_config.effort` > explicit
`thinking` > proxy default.

| Request control | GLM `reasoning_effort` |
|---|---|
| nothing (Claude Code default) | **`high`** (config `default_effort`) |
| `thinking: {type: "disabled"}` | `low` |
| `thinking: {type: "adaptive"}` | `high` (config `adaptive_effort`) |
| `thinking: {type: "enabled", budget_tokens: N}`, N < 8_192 | `low` |
| `thinking: {type: "enabled", budget_tokens: N}`, N ≥ 8_192 | `high` |
| `output_config.effort: max` (explicit only) | `max` |
| OpenAI-protocol `reasoning_effort` | `minimal`/`low`→`low`; `medium`/`high`→`high`; `max`→`max` |

The 8,192 threshold is the midpoint between the two supported tiers and is
configurable only in code (not config) to keep behavior reviewable.
`max` is **rejected** as a configured default (`glm53.reasoning.*`) — it
remains available only through an explicit client request.

## Historical thinking: zero accumulation by default

Claude Code stores assistant `thinking` blocks and replays them on every
subsequent turn. Left unchecked, historical reasoning snowballs a coding
session from ~140 KB requests to multi-MB requests even though none of that
reasoning is needed again. cline-proxy therefore (config
`glm53.reasoning`):

- **Strips historical reasoning from the upstream wire**
  (`strip_historical_thinking`, default on): `reasoning_content` is removed
  from all assistant messages *before the last user/tool-result turn*.
  Text, `tool_calls`, call ids, and ordering are never modified — the
  assistant.tool_calls ↔ tool.tool_call_id chain is preserved and verified
  by tests. This is the proxy-local, reliable equivalent of the template's
  official `clear_thinking` variable (which cline-proxy cannot assume Cline
  forwards).
- **Gates thinking exposure** (`expose_thinking: "requested_only"`, the
  default): upstream `reasoning_content` reaches Claude Code as Anthropic
  `thinking` blocks only when the request explicitly carried
  `thinking: {type: enabled|adaptive}`. Unexposed reasoning is still
  counted in telemetry (the model did produce it), but the client can never
  store it, replay it, and pay for it again. Modes: `requested_only` |
  `always` (legacy) | `never`.

The durable state Claude Code needs across turns — code edits, tool
outputs, file state, plans, conclusions — lives in text and tool blocks and
is never touched.

## Output cap

`glm53.limits.max_output_tokens` (default **16,384**, range 1024–131072)
bounds generation: `effective_max_tokens = min(client_max_tokens, cap)`.
8K is tight for complex coding turns; 32K+ invites runaway generation
(observed: 8.6-minute single turns producing 7.5 MB of SSE). When the
client sends no bound, the configured cap becomes the bound.

## Safe context compaction

`glm53.context.safe_compaction` (default on) applies only lossless,
structure-level normalization:

- single-text-block `content` arrays become plain strings (identical to the
  GLM template's `visible_text` semantics);
- empty text blocks (`{"type":"text","text":""}`) are dropped;
- Anthropic-only `metadata` is not forwarded (no OpenAI meaning).

Never: truncating tool results, editing tool descriptions/schemas, deleting
history, or any semantic compression. Claude Code remains responsible for
its own context compaction; the proxy does not compete with it.

## Request token telemetry

Per request (sizes/counts only — never content), logged as
`request optimization`: byte breakdown (`system_bytes`, `messages_bytes`,
`tools_bytes`, `other_bytes`, before/after sizes),
`historical_reasoning_bytes_removed`, `empty_blocks_removed`,
`normalized_text_blocks`, `reasoning_effort`, `effective_max_tokens`. Per
stream (log `Anthropic stream closed`): `reasoning_bytes`, `text_bytes`,
`tool_call_bytes`, event counts, `first_reasoning_ms`, `first_text_ms`,
`first_tool_call_ms`, plus upstream-reported `prompt_tokens`,
`completion_tokens`, `cached_tokens`, and `reasoning_tokens` when the
upstream supplies them (byte counters are never converted to tokens).

With `glm53.telemetry.exact_input_tokens` (default on), the embedded
official tokenizer computes the exact token count of the *optimized*
request in a **background task** (never on the TTFT path) and logs
`input_tokens`, `tokens_removed_historical_reasoning`, and the derived
`saved_percent`. `/v1/messages/count_tokens` counts the optimized request
the same way (header `x-cline-proxy-token-count: exact_glm53_optimized`)
so Claude Code's context budgeting matches real upstream usage; the
official-oracle count (thinking kept) remains available via
`count_input_tokens` when the strip policy is disabled.

## Known behavior notes

- Tools in the request increase GLM TTFT by seconds — third-party reports
  exist **[report]**; cline-proxy measures and logs TTFT per request
  **[bench]** but takes no correctness-risking countermeasures.
- Prompt-cache behavior across Cline credentials is **unverified**; strict
  key stickiness preserves locality without asserting cache semantics.
  Upstream `cached_tokens` are logged per stream when present so cache
  behavior can be evaluated from real data **[observed]**.
- Request-body gzip (`Content-Encoding: gzip`) is **not** implemented: it
  would reduce network upload bytes only, not model input tokens, and Cline
  endpoint compatibility is unverified. Priority stays below historical
  reasoning cleanup.

## Platform parity note

The repository stores `chat_template.jinja` with LF endings
(`.gitattributes`); the Rust pipeline additionally normalizes CRLF at
include time, so a Windows checkout with `core.autocrlf=true` cannot
silently change prompt bytes or token counts **[observed]** (CRLF once
skewed tool fixtures by +5 tokens).

## Oracle workflow

`tools/glm_reference/generate_fixtures.py` (dev-only, Python +
transformers) renders the official template and tokenizes with the official
tokenizer to produce `tests/fixtures/glm53/*.json` golden vectors. The
production binary embeds the same `tokenizer.json` (gzip) and the same
`chat_template.jinja`, rendered in-process with `minijinja` — no Python, no
network at runtime. Parity requirement: Rust token count and token IDs ==
official reference, exactly.
