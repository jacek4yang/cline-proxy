#!/usr/bin/env python3
"""Golden-fixture generator for exact GLM-5.3-Flash token counting.

DEV-ONLY ORACLE — production cline-proxy never invokes Python. This script
uses the OFFICIAL zai-org/GLM-5.3-Flash tokenizer assets (pinned revision,
see docs/GLM53_FLASH.md) via transformers to render the official chat
template and produce exact token IDs for each fixture scenario.

Each fixture records:
- anthropic_request: the Anthropic Messages payload a client would send
- glm_messages / glm_tools / reasoning_effort: the intermediate GLM
  representation cline-proxy's Rust conversion must produce (hand-written
  here, so conversion parity is tested independently of rendering)
- rendered_prompt: official template output (string)
- token_ids / token_count: official tokenization of that exact string

Usage:
    python tools/glm_reference/generate_fixtures.py
Output:
    tests/fixtures/glm53/<name>.json
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "tools" / "glm_reference" / "assets"
OUT = REPO / "tests" / "fixtures" / "glm53"


def read_file(path: str) -> str:
    return (Path(__file__).parent / "data" / path).read_text(encoding="utf-8")


RUST_SAMPLE = """fn main() {
    let points = vec![(0, 0), (1, 2)];
    for (x, y) in &points {
        println!("({x}, {y})");
    }
}"""


def big_tool(name: str, description: str) -> dict:
    """A Claude-Code-shaped tool with a realistic, sizable schema."""
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the target file, e.g. /home/user/project/src/main.rs",
                },
                "content": {
                    "type": "string",
                    "description": "The full content to write. Must be complete, valid UTF-8 text.",
                },
                "options": {
                    "type": "object",
                    "properties": {
                        "create_dirs": {"type": "boolean", "description": "Create missing parent directories"},
                        "overwrite": {"type": "boolean", "description": "Overwrite existing files"},
                        "mode": {"type": "string", "enum": ["text", "binary"], "description": "Write mode"},
                    },
                },
            },
            "required": ["file_path", "content"],
            "additionalProperties": False,
        },
    }


def fixture(name, anthropic_request, glm_messages, glm_tools=None, reasoning_effort=None):
    return {
        "name": name,
        "anthropic_request": anthropic_request,
        "glm_messages": glm_messages,
        "glm_tools": glm_tools or [],
        "reasoning_effort": reasoning_effort,
        "clear_thinking": False,
        "add_generation_prompt": True,
    }


def build_fixtures():
    fixtures = []

    # 1. Plain English string content.
    fixtures.append(
        fixture(
            "en_simple",
            {
                "model": "claude-sonnet-4-6",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            },
            [{"role": "user", "content": "Reply with exactly: OK"}],
        )
    )

    # 2. Plain Chinese.
    fixtures.append(
        fixture(
            "zh_simple",
            {
                "model": "z-ai/glm-5.3-flash",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
            },
            [{"role": "user", "content": "用一句话介绍你自己。"}],
        )
    )

    # 3. Mixed Chinese/English technical content.
    fixtures.append(
        fixture(
            "mixed_zh_en",
            {
                "model": "z-ai/glm-5.3-flash",
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "user",
                        "content": "请解释 Rust 的 lifetime（生命周期）机制，并用一个 borrow checker 会拒绝的例子说明。",
                    }
                ],
            },
            [
                {
                    "role": "user",
                    "content": "请解释 Rust 的 lifetime（生命周期）机制，并用一个 borrow checker 会拒绝的例子说明。",
                }
            ],
        )
    )

    # 4. Rust code payload.
    fixtures.append(
        fixture(
            "rust_code",
            {
                "model": "z-ai/glm-5.3-flash",
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Review this Rust snippet for correctness:\n"},
                            {"type": "text", "text": RUST_SAMPLE},
                        ],
                    }
                ],
            },
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Review this Rust snippet for correctness:\n"},
                        {"type": "text", "text": RUST_SAMPLE},
                    ]
                }
            ],
        )
    )

    # 5. JSON payload inside text.
    fixtures.append(
        fixture(
            "json_payload",
            {
                "model": "z-ai/glm-5.3-flash",
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "user",
                        "content": 'Summarize this JSON: {"name":"cline-proxy","keys":9,"models":["z-ai/glm-5.3-flash"],"limits":{"max_request_bytes":33554432}}',
                    }
                ],
            },
            [
                {
                    "role": "user",
                    "content": 'Summarize this JSON: {"name":"cline-proxy","keys":9,"models":["z-ai/glm-5.3-flash"],"limits":{"max_request_bytes":33554432}}',
                }
            ],
        )
    )

    # 6. Unicode + emoji.
    fixtures.append(
        fixture(
            "emoji_unicode",
            {
                "model": "z-ai/glm-5.3-flash",
                "max_tokens": 128,
                "messages": [
                    {
                        "role": "user",
                        "content": "Flags 🇨🇳🇺🇸 math ∑∫√∞ accents café naïve — Ünïcödé ✓ 汉字 Arabic العربية",
                    }
                ],
            },
            [
                {
                    "role": "user",
                    "content": "Flags 🇨🇳🇺🇸 math ∑∫√∞ accents café naïve — Ünïcödé ✓ 汉字 Arabic العربية",
                }
            ],
        )
    )

    # 7. System prompt (string) + user.
    fixtures.append(
        fixture(
            "system_prompt",
            {
                "model": "z-ai/glm-5.3-flash",
                "max_tokens": 256,
                "system": "You are a meticulous Rust engineer. Answer with code first, prose second.",
                "messages": [{"role": "user", "content": "Write a deterministic token counter."}],
            },
            [
                {
                    "role": "system",
                    "content": "You are a meticulous Rust engineer. Answer with code first, prose second.",
                },
                {"role": "user", "content": "Write a deterministic token counter."},
            ],
        )
    )

    # 8. Multi-turn conversation with assistant history.
    fixtures.append(
        fixture(
            "multi_turn",
            {
                "model": "z-ai/glm-5.3-flash",
                "max_tokens": 256,
                "messages": [
                    {"role": "user", "content": "What is a BPE tokenizer?"},
                    {
                        "role": "assistant",
                        "content": "A byte-pair-encoding tokenizer merges frequent byte pairs into tokens iteratively.",
                    },
                    {"role": "user", "content": "How does that differ from Unigram?"},
                    {
                        "role": "assistant",
                        "content": "Unigram starts from a large inventory and prunes tokens by likelihood; BPE grows merges bottom-up.",
                    },
                    {"role": "user", "content": "Which one does GLM use?"},
                ],
            },
            [
                {"role": "user", "content": "What is a BPE tokenizer?"},
                {
                    "role": "assistant",
                    "content": "A byte-pair-encoding tokenizer merges frequent byte pairs into tokens iteratively.",
                },
                {"role": "user", "content": "How does that differ from Unigram?"},
                {
                    "role": "assistant",
                    "content": "Unigram starts from a large inventory and prunes tokens by likelihood; BPE grows merges bottom-up.",
                },
                {"role": "user", "content": "Which one does GLM use?"},
            ],
        )
    )

    # 9. Two small tools.
    read_write = [
        {
            "name": "Read",
            "description": "Read a file from disk.",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
        {
            "name": "Write",
            "description": "Write a file to disk.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
                "required": ["path", "text"],
            },
        },
    ]
    fixtures.append(
        fixture(
            "tools_small",
            {
                "model": "z-ai/glm-5.3-flash",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "List the tools you have, briefly."}],
                "tools": read_write,
            },
            [{"role": "user", "content": "List the tools you have, briefly."}],
            glm_tools=read_write,
        )
    )

    # 10. Claude-Code-like large tool schemas (strict flag must not affect count).
    heavy_tools = [
        big_tool("Write", "Write the full contents of a file to disk."),
        big_tool("Edit", "Apply a precise string replacement inside a file."),
        big_tool("Bash", "Run a shell command and capture combined output."),
        big_tool("Grep", "Search file contents with a regular expression."),
        big_tool("Glob", "Find files by glob pattern."),
    ]
    for tool in heavy_tools:
        tool["strict"] = True  # template drops this key; count must not change
    fixtures.append(
        fixture(
            "tools_large_strict_dropped",
            {
                "model": "z-ai/glm-5.3-flash",
                "max_tokens": 1024,
                "system": [
                    {"type": "text", "text": "You are Claude Code, an agentic coding assistant."}
                ],
                "messages": [{"role": "user", "content": "Refactor src/main.rs and run the tests."}],
                "tools": heavy_tools,
            },
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are Claude Code, an agentic coding assistant."}],
                },
                {"role": "user", "content": "Refactor src/main.rs and run the tests."},
            ],
            glm_tools=heavy_tools,
        )
    )

    # 11. Single tool-call loop: assistant tool_use + tool result + follow-up.
    fixtures.append(
        fixture(
            "tool_loop",
            {
                "model": "z-ai/glm-5.3-flash",
                "max_tokens": 512,
                "messages": [
                    {"role": "user", "content": "What is in /tmp/notes.txt?"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "I'll read that file."},
                            {
                                "type": "tool_use",
                                "id": "toolu_01",
                                "name": "Read",
                                "input": {"path": "/tmp/notes.txt"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_01", "content": "buy milk\nwrite tests"}
                        ],
                    },
                ],
                "tools": read_write,
            },
            [
                {"role": "user", "content": "What is in /tmp/notes.txt?"},
                {
                    "role": "assistant",
                    "content": "I'll read that file.",
                    "tool_calls": [{"function": {"name": "Read", "arguments": {"path": "/tmp/notes.txt"}}}],
                },
                {"role": "tool", "tool_call_id": "toolu_01", "content": "buy milk\nwrite tests"},
            ],
            glm_tools=read_write,
        )
    )

    # 12. Parallel tool calls + results (proper Anthropic tool_use blocks).
    fixtures.append(
        fixture(
            "parallel_tool_calls",
            {
                "model": "z-ai/glm-5.3-flash",
                "max_tokens": 512,
                "messages": [
                    {"role": "user", "content": "Compare a.txt and b.txt"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_a",
                                "name": "Read",
                                "input": {"path": "/tmp/a.txt"},
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_b",
                                "name": "Read",
                                "input": {"path": "/tmp/b.txt"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_a", "content": "alpha contents"},
                            {"type": "tool_result", "tool_use_id": "toolu_b", "content": "beta contents"},
                        ],
                    },
                ],
                "tools": read_write,
            },
            [
                {"role": "user", "content": "Compare a.txt and b.txt"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "Read", "arguments": {"path": "/tmp/a.txt"}}},
                        {"function": {"name": "Read", "arguments": {"path": "/tmp/b.txt"}}},
                    ],
                },
                {"role": "tool", "tool_call_id": "toolu_a", "content": "alpha contents"},
                {"role": "tool", "tool_call_id": "toolu_b", "content": "beta contents"},
            ],
            glm_tools=read_write,
        )
    )

    # 13. Assistant history containing reasoning (clear_thinking=false keeps it).
    fixtures.append(
        fixture(
            "thinking_history_kept",
            {
                "model": "z-ai/glm-5.3-flash",
                "max_tokens": 8192,
                "thinking": {"type": "enabled", "budget_tokens": 4096},
                "messages": [
                    {"role": "user", "content": "Is 97 prime?"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Check divisibility up to sqrt(97)≈9.8: not 2,3,5,7. Yes, prime.", "signature": "sig"},
                            {"type": "text", "text": "97 is prime."},
                        ],
                    },
                    {"role": "user", "content": "And 91?"},
                ],
            },
            [
                {"role": "user", "content": "Is 97 prime?"},
                {
                    "role": "assistant",
                    "content": "97 is prime.",
                    "reasoning_content": "Check divisibility up to sqrt(97)≈9.8: not 2,3,5,7. Yes, prime.",
                },
                {"role": "user", "content": "And 91?"},
            ],
            reasoning_effort="low",
        )
    )

    # 14. Adaptive thinking maps to high.
    fixtures.append(
        fixture(
            "adaptive_thinking_high",
            {
                "model": "z-ai/glm-5.3-flash",
                "max_tokens": 512,
                "thinking": {"type": "adaptive"},
                "messages": [{"role": "user", "content": "Plan a release checklist."}],
            },
            [{"role": "user", "content": "Plan a release checklist."}],
            reasoning_effort="high",
        )
    )

    # 15. Claude-Code-like aggregate payload (system + heavy tools + tool loop).
    fixtures.append(
        fixture(
            "claude_code_like",
            {
                "model": "claude-sonnet-4-6",
                "max_tokens": 8192,
                "system": [
                    {
                        "type": "text",
                        "text": (
                            "You are Claude Code, Anthropic's official CLI for Claude. "
                            "You are an interactive CLI tool that helps users with software "
                            "engineering tasks. Use the instructions below and the tools "
                            "available to you to assist the user. Answer concisely."
                        ),
                    }
                ],
                "messages": [
                    {"role": "user", "content": "Run the test suite and fix any failure in src/pool.rs."},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "Bash",
                                "input": {"command": "cargo test 2>&1 | tail -20", "options": {"overwrite": False}},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "test result: FAILED. 1 failed; 78 passed\nfailures: pool::tests::sticky_rotation",
                            }
                        ],
                    },
                ],
                "tools": heavy_tools,
            },
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are Claude Code, Anthropic's official CLI for Claude. "
                                "You are an interactive CLI tool that helps users with software "
                                "engineering tasks. Use the instructions below and the tools "
                                "available to you to assist the user. Answer concisely."
                            ),
                        }
                    ],
                },
                {"role": "user", "content": "Run the test suite and fix any failure in src/pool.rs."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "Bash",
                                "arguments": {"command": "cargo test 2>&1 | tail -20", "options": {"overwrite": False}},
                            }
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "toolu_1",
                    "content": "test result: FAILED. 1 failed; 78 passed\nfailures: pool::tests::sticky_rotation",
                },
            ],
            glm_tools=heavy_tools,
        )
    )

    return fixtures


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(str(ASSETS))
    OUT.mkdir(parents=True, exist_ok=True)
    for f in build_fixtures():
        kwargs = dict(
            conversation=f["glm_messages"],
            tools=f["glm_tools"] or None,
            clear_thinking=f["clear_thinking"],
            add_generation_prompt=f["add_generation_prompt"],
            tokenize=True,
            return_dict=False,
        )
        if f["reasoning_effort"] is not None:
            kwargs["reasoning_effort"] = f["reasoning_effort"]
        token_ids = tokenizer.apply_chat_template(**kwargs)
        rendered = tokenizer.apply_chat_template(
            **{**kwargs, "tokenize": False}
        )
        record = {
            "name": f["name"],
            "anthropic_request": f["anthropic_request"],
            "glm_messages": f["glm_messages"],
            "glm_tools": f["glm_tools"],
            "reasoning_effort": f["reasoning_effort"],
            "clear_thinking": f["clear_thinking"],
            "add_generation_prompt": f["add_generation_prompt"],
            "rendered_prompt": rendered,
            "token_ids": token_ids,
            "token_count": len(token_ids),
        }
        out_path = OUT / f"{f['name']}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{f['name']}: {len(token_ids)} tokens -> {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
