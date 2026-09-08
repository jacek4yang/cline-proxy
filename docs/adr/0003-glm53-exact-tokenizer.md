# ADR 0003: Exact GLM-5.3-Flash tokenizer, embedded official assets

Date: 2026-09-08 · Status: Accepted

## Context

`/v1/messages/count_tokens` previously returned `serialized_bytes / 4`. Claude
Code uses count_tokens to decide when to compact context; an approximation on
~1.5 MB real requests is wrong by thousands of tokens in both directions.

GLM-5.3-Flash publishes its tokenizer and chat template officially
(zai-org/GLM-5.3-Flash @ `eb9eb20`, MIT), making exact counting achievable
in-process.

## Decision

1. Embed the official `tokenizer.json` **gzip-compressed** (20.2 MB → 3.0 MB)
   and the official `chat_template.jinja` (11 KB, unmodified) in the binary.
   The gateway stays a single self-contained file; no Python, no network, no
   runtime asset files.
2. Rendering uses `minijinja` configured to match transformers exactly:
   `trim_blocks=True`, `lstrip_blocks=True`, `preserve_order` maps,
   `pycompat` Python-method emulation (`dict.items()`, `str.strip()`,
   `str.split()`), and a Python-compatible `tojson` filter
   (`, `/`: ` separators, `ensure_ascii=False` semantics). The official
   template file itself is never modified.
3. Parity is enforced by golden fixtures (`tests/fixtures/glm53/`) generated
   by `tools/glm_reference/generate_fixtures.py` with transformers + the
   official tokenizer. Rust output must match the reference **exactly**
   (byte-identical rendering, identical token counts). 15 scenarios cover
   EN/ZH/mixed/code/JSON/emoji, system, multi-turn, tools, parallel tool
   calls, thinking history, and a Claude-Code-like aggregate.
4. Binary size cost: +~3 MB embedded asset + ~5 MB for tokenizers/minijinja
   — accepted for a core correctness feature. The tokenizer parses once,
   lazily, on the first count.
5. Counting follows the real prompt path: Anthropic → GLM message shape →
   official template (`add_generation_prompt=true`, `clear_thinking=false`
   which is the official default) → tokenizer. URL/base64 *documents* are
   rejected instead of undercounted; images count as their template
   placeholder only (vision-token expansion is server-side and not text-
   countable) — both documented in docs/GLM53_FLASH.md.

## Consequences

- `serde_json` gained `preserve_order` (keys keep client order — also good
  for prefix-cache stability) and the binary grows ~8 MB total.
- `tokenizers` (HF crate) with `fancy-regex` keeps the build pure Rust.
- Fixture regeneration is a dev-only Python step; CI never needs Python.
- If the upstream tokenizer/template revision changes, bump the pinned
  revision, regenerate fixtures, and re-run the parity tests.
