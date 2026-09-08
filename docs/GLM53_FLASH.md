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
the GLM API itself accepts `low`, `high`, and `max`. cline-proxy maps
Anthropic reasoning controls to GLM efforts as follows — documented, tested,
and mirrored exactly in token counting. Single source of truth:
`src/glm53/reasoning.rs`.

- `thinking: {type: "disabled"}` → `reasoning_effort: "low"` (the closest
  supported tier to "minimal thinking"; sending nothing would render the
  official default `max`, which is the opposite of what the caller asked).
- `thinking: {type: "adaptive"}` → `reasoning_effort: "high"`.
- `thinking: {type: "enabled", budget_tokens: N}` → `N < 8_192` → `"low"`,
  otherwise `"high"`. The previous `medium` tier was removed: GLM-5.3-Flash
  has no `medium`, and the template silently turned it into `max`
  **[official template behavior]** — a real semantic bug this release fixes.
- `output_config.effort`: `low`→`low`; `medium`/`high`/`xhigh`→`high`;
  `max`→`max`. Values outside the Anthropic set are rejected upstream.

The 8,192 threshold is the midpoint between the two supported tiers and is
configurable only in code (not config) to keep behavior reviewable.

## Known behavior notes

- Tools in the request increase GLM TTFT by seconds — third-party reports
  exist **[report]**; cline-proxy measures and logs `ttft_ms` per request
  **[bench]** but takes no correctness-risking countermeasures.
- Prompt-cache behavior across Cline credentials is **unverified**; strict
  key stickiness preserves locality without asserting cache semantics.

## Oracle workflow

`tools/glm_reference/generate_fixtures.py` (dev-only, Python +
transformers) renders the official template and tokenizes with the official
tokenizer to produce `tests/fixtures/glm53/*.json` golden vectors. The
production binary embeds the same `tokenizer.json` (gzip) and the same
`chat_template.jinja`, rendered in-process with `minijinja` — no Python, no
network at runtime. Parity requirement: Rust token count and token IDs ==
official reference, exactly.
