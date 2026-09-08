//! Latency baseline for the GLM-5.3-Flash request policy pipeline
//! (issue #6): convert → optimize → exact token count → serialize.
//!
//! `harness = false`: plain binary, no criterion dependency. Run with
//! `cargo bench --bench request_optimization` (release profile). Payloads
//! scale 1 KB / ~100 KB / ~1.3 MB, matching real Claude Code request sizes.
//! Budget: local overhead must stay milliseconds-scale and far below
//! upstream latency; the exact token count may take hundreds of ms at
//! 1.3 MB but runs in a background task in production (never on TTFT).

use std::time::Instant;

use serde_json::{json, Value};

use cline_proxy::anthropic::convert_request;
use cline_proxy::optimize::{optimize_request, Origin};

fn repeat_text(seed: &str, target_bytes: usize) -> String {
    let mut text = String::with_capacity(target_bytes);
    while text.len() < target_bytes {
        text.push_str(seed);
    }
    text
}

/// Claude Code-like request: system prompt, several tool schemas, a
/// multi-turn tool loop with historical thinking blocks and large tool
/// results.
fn claude_code_like(request_bytes: usize, turns: usize) -> Value {
    let tools: Vec<Value> = ["Read", "Edit", "Bash", "Grep", "Glob"]
        .iter()
        .map(|name| {
            json!({
                "name": name,
                "description": repeat_text(
                    &format!("Use {name} during coding tasks. "), 4),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string",
                            "description": repeat_text("Absolute path. ", 8)},
                        "command": {"type": "string",
                            "description": repeat_text("The command to run. ", 8)},
                    },
                    "required": ["file_path"],
                },
            })
        })
        .collect();
    let per_turn = request_bytes / turns.max(1);
    let mut messages = vec![json!({"role":"user", "content":[{"type":"text",
        "text": repeat_text("Analyze this repository section. ", per_turn / 2)}]})];
    for turn in 0..turns.max(1) {
        messages.push(json!({"role":"assistant", "content":[
            {"type":"thinking", "thinking": repeat_text(
                "Reasoning about the edit and its blast radius. ", per_turn / 8),
             "signature":"sig"},
            {"type":"tool_use", "id": format!("toolu_{turn}"), "name":"Bash",
             "input":{"command": repeat_text("cargo test --quiet ", per_turn / 16)}}
        ]}));
        messages.push(json!({"role":"user", "content":[
            {"type":"tool_result", "tool_use_id": format!("toolu_{turn}"),
             "content": repeat_text("test result: ok. 100 passed\n", per_turn / 4)}
        ]}));
    }
    json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 8192,
        "system": [{"type": "text", "text": repeat_text(
            "You are Claude Code, an agentic coding assistant. ",
            request_bytes / 4,
        )}],
        "messages": messages,
        "tools": tools,
    })
}

fn bench(name: &str, request_bytes: usize, turns: usize, token_count: bool) {
    let payload = claude_code_like(request_bytes, turns);
    let raw = serde_json::to_vec(&payload).unwrap();
    let rounds = if request_bytes > 500_000 { 3 } else { 10 };

    // Warm the template/tokenizer singletons so we measure steady state.
    let _ = cline_proxy::glm53::count::count_input_tokens(&{
        let mut warm = payload.clone();
        warm["messages"] = json!([{"role":"user","content":"warm"}]);
        warm
    });

    let mut convert_total = 0u128;
    let mut optimize_total = 0u128;
    let mut serialize_total = 0u128;
    let mut count_total = 0u128;
    let mut after_bytes = 0usize;
    let mut removed_bytes = 0u64;
    let mut tokens = 0u32;
    for _ in 0..rounds {
        let t0 = Instant::now();
        let mut converted = convert_request(&raw).unwrap();
        let t1 = Instant::now();
        let optimization = optimize_request(
            &mut converted.body,
            &cline_proxy::config::Glm53Config::default(),
            Origin::Anthropic {
                thinking: converted.thinking.as_ref(),
                output_effort: converted.output_effort.as_deref(),
            },
        )
        .unwrap();
        let t2 = Instant::now();
        let body_bytes = serde_json::to_vec(&converted.body).unwrap();
        let t3 = Instant::now();
        if token_count {
            let mut counted: Value = serde_json::from_slice(&raw).unwrap();
            cline_proxy::optimize::strip_anthropic_thinking(&mut counted);
            counted["output_config"] = json!({"effort":"high"});
            tokens = cline_proxy::glm53::count::count_input_tokens(&counted).unwrap();
        }
        let t4 = Instant::now();
        convert_total += (t1 - t0).as_micros();
        optimize_total += (t2 - t1).as_micros();
        serialize_total += (t3 - t2).as_micros();
        count_total += (t4 - t3).as_micros();
        after_bytes = body_bytes.len();
        removed_bytes = optimization.historical_reasoning_bytes_removed;
    }
    let rounds_f = rounds as f64;
    println!(
        "{name:<14} raw={:>8}B after={:>8}B removed_reasoning={:>7}B tokens={:>7} | \
         convert={:>7.1}ms optimize={:>7.1}ms serialize={:>7.1}ms count={:>7.1}ms",
        raw.len(),
        after_bytes,
        removed_bytes,
        tokens,
        convert_total as f64 / rounds_f / 1000.0,
        optimize_total as f64 / rounds_f / 1000.0,
        serialize_total as f64 / rounds_f / 1000.0,
        count_total as f64 / rounds_f / 1000.0,
    );
}

fn main() {
    bench("small (1KB)", 1_000, 2, true);
    bench("medium (100KB)", 100_000, 6, true);
    bench("large (1.3MB)", 1_300_000, 24, true);
}
