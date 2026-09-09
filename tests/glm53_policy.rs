//! Claude Code-shaped integration scenarios for the GLM-5.3-Flash request
//! policy (issue #6).
//!
//! Each scenario simulates a multi-turn agent loop at the request level:
//! every turn builds the Anthropic Messages request Claude Code would send
//! (history includes assistant `thinking` blocks with signatures, tool_use,
//! and tool_result), converts and optimizes it exactly like
//! `POST /v1/messages` does, and asserts the invariants that matter for
//! real coding sessions:
//!
//! - the upstream wire never carries historical `reasoning_content`;
//! - the assistant.tool_calls ↔ tool.tool_call_id chain survives stripping;
//! - historical thinking contributes ~0 tokens to later turns (no
//!   snowballing on long sessions);
//! - `reasoning_effort` is always explicit and bounded output is capped.

use serde_json::{json, Value};

use cline_proxy::anthropic::convert_request;
use cline_proxy::config::Glm53Config;
use cline_proxy::optimize::{optimize_request, Origin};

fn glm_config() -> Glm53Config {
    Glm53Config::default()
}

/// One assistant turn as Claude Code would send it back in history.
fn assistant_turn(thinking: &str, text: &str, tool_calls: Vec<(u64, &str, &str, Value)>) -> Value {
    let mut content = vec![json!({"type":"thinking", "thinking":thinking,
               "signature":"sig-opaque-from-previous-turn"})];
    if !text.is_empty() {
        content.push(json!({"type":"text", "text":text}));
    }
    for (index, id, name, input) in tool_calls {
        let _ = index;
        content.push(json!({"type":"tool_use", "id":id, "name":name, "input":input}));
    }
    json!({"role":"assistant", "content":content})
}

fn tool_result_turn(id: &str, content: &str) -> Value {
    json!({"role":"user", "content":[
        {"type":"tool_result", "tool_use_id":id, "content":content}
    ]})
}

fn user_turn(text: &str) -> Value {
    json!({"role":"user", "content":[{"type":"text", "text":text}]})
}

fn tools() -> Value {
    json!([
        {"name":"Read", "description":"Read a file from disk.",
         "input_schema":{"type":"object","properties":{"file_path":{"type":"string"}},
                         "required":["file_path"]}},
        {"name":"Edit", "description":"Replace text in a file.",
         "input_schema":{"type":"object","properties":{
             "file_path":{"type":"string"},"old_text":{"type":"string"},
             "new_text":{"type":"string"}},
             "required":["file_path","old_text","new_text"]}},
        {"name":"Bash", "description":"Run a shell command.",
         "input_schema":{"type":"object","properties":{"command":{"type":"string"}},
                         "required":["command"]}}
    ])
}

fn build_request(history: &[Value], thinking_control: Option<Value>) -> Vec<u8> {
    let mut request = json!({
        "model":"claude-sonnet-4-6",
        "max_tokens":32_000,
        "system":[{"type":"text","text":"You are Claude Code, an agentic coding assistant."}],
        "messages":history,
        "tools":tools()
    });
    if let Some(thinking) = thinking_control {
        request["thinking"] = thinking;
    }
    serde_json::to_vec(&request).unwrap()
}

/// Convert + optimize exactly as the /v1/messages handler does, including
/// alias resolution (claude-* ids resolve to the GLM upstream model before
/// the policy step runs).
fn pipeline(bytes: &[u8]) -> (Value, cline_proxy::optimize::RequestOptimization) {
    let mut converted = convert_request(bytes).unwrap();
    if converted.model == "claude-sonnet-4-6" {
        cline_proxy::anthropic::apply_model(&mut converted, "z-ai/glm-5.3-flash".to_owned());
    }
    let optimization = optimize_request(
        &mut converted.body,
        &glm_config(),
        Origin::Anthropic {
            thinking: converted.thinking.as_ref(),
            output_effort: converted.output_effort.as_deref(),
        },
    )
    .unwrap();
    converted.expose_thinking = optimization.expose_thinking;
    (converted.body, optimization)
}

fn assert_tool_chain_intact(body: &Value) {
    let messages = body["messages"].as_array().unwrap();
    let mut pending_call_ids = std::collections::BTreeSet::new();
    for message in messages {
        let role = message["role"].as_str().unwrap();
        if role == "assistant" {
            if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    let id = call["id"].as_str().unwrap();
                    assert!(!id.is_empty());
                    assert!(!call["function"]["name"].as_str().unwrap().is_empty());
                    pending_call_ids.insert(id.to_owned());
                }
            }
        }
        if role == "tool" {
            let id = message["tool_call_id"].as_str().unwrap();
            assert!(
                pending_call_ids.remove(id),
                "tool result {id} must follow its tool_call"
            );
        }
    }
    assert!(
        pending_call_ids.is_empty(),
        "every tool_call must have its tool result"
    );
}

/// A large, realistic reasoning blob (the kind Claude Code stores from an
/// exposed thinking block and replays on every subsequent turn).
fn big_thinking(turn: usize) -> String {
    format!(
        "Turn {turn} analysis: I need to inspect src/config.rs, trace the \
         reasoning_effort resolution path, check the template coercion, and \
         design a bounded default. Let me enumerate the call sites, the \
         fixture expectations, and the streaming exposure gate. "
    )
    .repeat(40)
}

fn tool_result_body(turn: usize) -> String {
    format!(
        "warning: unused variable `effort` at src/config.rs:{line}\n\
         note: {} potential issues found in workspace\n\
         test result: ok. {} passed\n",
        turn * 3,
        turn * 12,
        line = 100 + turn
    )
    .repeat(10)
}

/// Case A: read file → edit file → run tests, THEN a second human turn
/// starts a new reasoning epoch. Epoch-1 thinking must be stripped; the
/// current epoch keeps its reasoning continuity (issue #10).
#[test]
fn case_a_new_human_turn_strips_previous_epoch_reasoning() {
    let history = vec![
        user_turn("Fix the off-by-one in src/counter.rs, then run the tests."),
        assistant_turn(
            &big_thinking(1),
            "",
            vec![(0, "toolu_a1", "Read", json!({"file_path":"src/counter.rs"}))],
        ),
        tool_result_turn("toolu_a1", "fn counter(n: u32) -> u32 { n + 1 }"),
        // New human turn = new reasoning epoch.
        user_turn("Now make the counter thread-safe."),
        assistant_turn(
            &big_thinking(2),
            "",
            vec![(
                0,
                "toolu_a2",
                "Edit",
                json!({"file_path":"src/counter.rs",
                "old_text":"n + 1", "new_text":"n + 2"}),
            )],
        ),
        tool_result_turn("toolu_a2", "edited successfully"),
        assistant_turn(
            &big_thinking(3),
            "",
            vec![(0, "toolu_a3", "Bash", json!({"command":"cargo test"}))],
        ),
        tool_result_turn("toolu_a3", "test result: ok. 3 passed"),
    ];
    let (body, optimization) = pipeline(&build_request(&history, None));
    let messages = body["messages"].as_array().unwrap();
    // Epoch-1 assistant reasoning (before the second human turn) stripped;
    // current-epoch reasoning preserved for tool-loop continuity.
    let stripped = messages
        .iter()
        .take(3)
        .filter(|m| m.get("reasoning_content").is_some())
        .count();
    assert_eq!(stripped, 0, "epoch-1 reasoning must be gone");
    let kept = messages
        .iter()
        .skip(3)
        .filter(|m| m.get("reasoning_content").is_some())
        .count();
    assert_eq!(kept, 2, "current-epoch reasoning is preserved");
    assert_eq!(optimization.reasoning_effort, "high");
    assert_eq!(optimization.effective_max_tokens, Some(16_384));
    assert!(optimization.historical_reasoning_bytes_removed > 3_000);
    assert_tool_chain_intact(&body);
}

/// Case B: compile-error debugging loop with cargo check diagnostics, all
/// within one reasoning epoch (single human turn). In-epoch reasoning is
/// preserved (continuity), and diagnostics are never truncated.
#[test]
fn case_b_debug_loop_preserves_diagnostics_and_effort_precedence() {
    let history = vec![
        user_turn("cargo check fails after the refactor; fix it."),
        assistant_turn(
            &big_thinking(1),
            "",
            vec![(0, "toolu_b1", "Bash", json!({"command":"cargo check"}))],
        ),
        tool_result_turn("toolu_b1", &tool_result_body(1)),
        assistant_turn(
            &big_thinking(2),
            "",
            vec![(0, "toolu_b2", "Read", json!({"file_path":"src/lib.rs"}))],
        ),
        tool_result_turn("toolu_b2", "pub mod optimize; pub mod glm53;"),
        assistant_turn(
            &big_thinking(3),
            "",
            vec![(
                0,
                "toolu_b3",
                "Edit",
                json!({"file_path":"src/lib.rs",
                "old_text":"pub mod optimize;", "new_text":"pub mod optimize;\npub mod new_mod;"}),
            )],
        ),
        tool_result_turn("toolu_b3", "edited successfully"),
        assistant_turn(
            &big_thinking(4),
            "",
            vec![(0, "toolu_b4", "Bash", json!({"command":"cargo test"}))],
        ),
        tool_result_turn("toolu_b4", &tool_result_body(4)),
    ];
    // Explicit large thinking budget: stays high but must be exposed.
    let (body, optimization) = pipeline(&build_request(
        &history,
        Some(json!({"type":"enabled","budget_tokens":20_000})),
    ));
    assert_eq!(optimization.reasoning_effort, "high");
    assert!(optimization.expose_thinking);
    // One reasoning epoch, no second human turn: nothing is stripped —
    // the wire keeps in-epoch reasoning continuity (issue #10).
    assert_eq!(optimization.historical_reasoning_bytes_removed, 0);
    assert_eq!(
        body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m.get("reasoning_content").is_some())
            .count(),
        4
    );
    // Diagnostics are never truncated: tool results keep their full text.
    let diagnostics = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["tool_call_id"] == "toolu_b1")
        .unwrap();
    assert!(diagnostics["content"]
        .as_str()
        .unwrap()
        .starts_with("warning: unused variable"));
    assert_tool_chain_intact(&body);
}

/// Case C: a long session of 24 human turns (each followed by a tool
/// loop). Reasoning from *previous* epochs must contribute zero tokens:
/// the count of a session whose history carries big reasoning blobs in
/// every epoch must equal the count of a session carrying the blobs ONLY
/// in the current epoch — and doubling current-epoch blob size must be the
/// only growth. Sub-linear snowballing is the acceptance bar.
#[test]
fn case_c_long_session_previous_epoch_reasoning_contribution_is_zero() {
    // count_at(turn, historical_scale, current_scale): all 1..=turn
    // human+tool-loop steps; steps 1..=turn-1 carry `historical_scale`
    // blobs, the final step carries `current_scale` blobs.
    let count_at = |turn: usize, historical_scale: usize, current_scale: usize| -> u64 {
        let mut history: Vec<Value> = Vec::new();
        for step in 1..=turn {
            let scale = if step == turn {
                current_scale
            } else {
                historical_scale
            };
            let thinking = if scale == 0 {
                String::new()
            } else {
                big_thinking(step).repeat(scale)
            };
            history.push(user_turn(&format!("Step {step} of the refactoring plan.")));
            history.push(assistant_turn(
                &thinking,
                "",
                vec![(
                    0,
                    "toolu_c",
                    "Bash",
                    json!({"command":format!("cargo test step_{step}")}),
                )],
            ));
            history.push(tool_result_turn("toolu_c", &tool_result_body(step)));
        }
        cline_proxy::glm53::count::count_input_tokens(&{
            let mut request: Value = json!({
                "model": "z-ai/glm-5.3-flash",
                "max_tokens": 32_000,
                "messages": history,
            });
            cline_proxy::optimize::strip_anthropic_thinking(&mut request);
            request["output_config"] = json!({"effort":"high"});
            request
        })
        .unwrap() as u64
    };
    for turn in [4usize, 12, 24] {
        // Historical epochs: scale-0 vs scale-2 blobs must count the same
        // after the strip — previous-epoch reasoning contributes 0 tokens.
        // Both carry the SAME current-epoch blob (scale 1).
        let no_history = count_at(turn, 0, 1);
        let heavy_history = count_at(turn, 2, 1);
        assert_eq!(
            no_history, heavy_history,
            "turn {turn}: previous-epoch reasoning must contribute zero tokens"
        );
        // The current epoch's reasoning is preserved (one blob): growing it
        // grows the count, proving the equality above is not vacuous.
        let bigger_current = count_at(turn, 0, 3);
        assert!(
            bigger_current > no_history,
            "turn {turn}: current-epoch reasoning is retained"
        );
    }
}

/// Disabling the policy restores passthrough behavior (escape hatch).
#[test]
fn policy_disabled_preserves_historical_reasoning() {
    let mut config = glm_config();
    config.reasoning.strip_historical_thinking = false;
    config.context.safe_compaction = false;
    let history = vec![
        user_turn("hello"),
        assistant_turn(&big_thinking(1), "working", vec![]),
        user_turn("continue"),
    ];
    let bytes = build_request(&history, None);
    let mut converted = convert_request(&bytes).unwrap();
    let optimization = optimize_request(
        &mut converted.body,
        &config,
        Origin::Anthropic {
            thinking: converted.thinking.as_ref(),
            output_effort: converted.output_effort.as_deref(),
        },
    )
    .unwrap();
    assert_eq!(optimization.historical_reasoning_bytes_removed, 0);
    assert_eq!(
        converted.body["messages"][2]["reasoning_content"]
            .as_str()
            .unwrap()
            .len(),
        big_thinking(1).len()
    );
}
