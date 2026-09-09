//! Latency baseline for exact GLM-5.3-Flash token counting.
//!
//! `harness = false`: plain binary, no criterion dependency. Run with
//! `cargo bench` (release profile). Payloads scale from 1 KB to ~1.4 MB,
//! matching real Claude Code request sizes.

use std::time::Instant;

use serde_json::{json, Value};

fn repeat_text(seed: &str, target_bytes: usize) -> String {
    let mut text = String::with_capacity(target_bytes);
    while text.len() < target_bytes {
        text.push_str(seed);
    }
    text
}

fn claude_code_like(request_bytes: usize) -> Value {
    // Realistic shape: system prompt + several tool schemas + tool loop +
    // long accumulated context.
    let tool = json!({
        "name": "Bash",
        "description": repeat_text(
            "Run a shell command and capture combined output. ", 8),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": repeat_text("The command to execute. ", 8)},
                "timeout": {"type": "number", "description": repeat_text("Optional timeout in seconds. ", 8)},
            },
            "required": ["command"],
        },
    });
    json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 8192,
        "system": [{"type": "text", "text": repeat_text(
            "You are Claude Code, an agentic coding assistant. Follow the user's instructions. ",
            request_bytes / 4,
        )}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": repeat_text(
                "Please analyze this repository section. ", request_bytes / 2,
            )}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {
                    "command": repeat_text("cargo test --quiet ", request_bytes / 16),
                }},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": repeat_text(
                    "test result: ok. 100 passed; 0 failed\n", request_bytes / 8,
                )},
            ]},
        ],
        "tools": [tool],
    })
}

fn measure(name: &str, request: &Value) {
    let rendered_bytes = serde_json::to_vec(request).unwrap().len();
    // Warm-up (loads the tokenizer once).
    let first = cline_proxy::glm53::count::count_input_tokens(request).unwrap();
    let start = Instant::now();
    let iterations = 20;
    for _ in 0..iterations {
        cline_proxy::glm53::count::count_input_tokens(request).unwrap();
    }
    let per_call = start.elapsed() / iterations;
    println!(
        "{name:<14} payload={rendered_bytes:>9} B  tokens={first:>7}  count={per_call:>10.1?}/call"
    );
}

fn main() {
    // Runs against the library crate (src/lib.rs) so the bench exercises
    // exactly the production counting pipeline.
    for size in [1_024, 16_384, 65_536, 262_144, 1_400_000] {
        measure(&format!("{size} B"), &claude_code_like(size));
    }
}
