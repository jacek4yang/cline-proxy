//! Upstream request optimization (issue #6).
//!
//! Applied to the OpenAI Chat Completions body right before it is sent to
//! Cline. Four concerns, all individually configurable and logged:
//!
//! 1. **Reasoning policy** — every upstream request carries an explicit
//!    `reasoning_effort` resolved by
//!    `crate::glm53::reasoning::resolve_reasoning_policy` (single source of
//!    truth). Unset must never reach GLM: the official chat template coerces
//!    any effort outside {low, high} to `max`.
//! 2. **Output cap** — `effective_max_tokens = min(client, configured)`.
//! 3. **Historical thinking strip** — `reasoning_content` is removed from
//!    assistant messages before the last user/tool-result turn. Text,
//!    `tool_calls`, call ids, and ordering are never modified, so the
//!    assistant.tool_calls ↔ tool.tool_call_id chain is preserved. This does
//!    for GLM what its official `clear_thinking` does, but locally and
//!    reliably (we do not depend on Cline forwarding the flag).
//! 4. **Safe compaction** — lossless structural normalization only:
//!    single-text-block content arrays become plain strings, empty text
//!    blocks are dropped, and Anthropic-only `metadata` is not forwarded.
//!    Tool results are never truncated and tool schemas are never edited.
//!
//! Only sizes and counts are recorded — never prompt content.

use serde_json::{Map, Value};

use crate::config::Glm53Config;
use crate::glm53::reasoning::resolve_reasoning_policy;

/// Upstream model family, resolved from the *upstream* model id. GLM policy
/// (reasoning effort, output cap, historical-reasoning strip, compaction) is
/// GLM-specific wire semantics; every other model receives the wire request
/// essentially untouched (issue #6 review: the policy must be model-scoped,
/// never a global rewrite of all OpenAI-compatible bodies).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ModelFamily {
    /// `z-ai/glm-*` ids resolved through `models.aliases`/`models.default`.
    #[default]
    Glm53,
    /// Any other model id: passthrough-compatible handling only.
    GenericOpenAi,
}

impl ModelFamily {
    pub fn from_upstream_model(model: &str) -> Self {
        let id = model.rsplit('/').next().unwrap_or(model);
        // Segment match: "glm" alone (glm-5.3-flash) or glm followed by a
        // version digit (glm53, glm4.7). Bare substring matching would
        // misclassify ids that merely mention glm in another segment.
        if id.split(['-', '_', '.', '+']).any(|segment| {
            let lower = segment.to_ascii_lowercase();
            let rest = lower.strip_prefix("glm").unwrap_or_else(|| {
                if lower == "glm" {
                    ""
                } else {
                    "\u{0}not-a-match"
                }
            });
            rest.is_empty() || rest.starts_with(|character: char| character.is_ascii_digit())
        }) {
            Self::Glm53
        } else {
            Self::GenericOpenAi
        }
    }
}

/// Size/count telemetry for one optimized request. All fields are logged;
/// none contain request content.
#[derive(Debug, Default, Clone)]
pub struct RequestOptimization {
    pub model_family: ModelFamily,
    /// GLM effort placed on the wire (`""` when the family does not use it).
    pub reasoning_effort: &'static str,
    pub expose_thinking: bool,
    pub client_max_tokens: Option<u64>,
    pub effective_max_tokens: Option<u64>,
    pub historical_reasoning_bytes_removed: u64,
    pub empty_blocks_removed: u64,
    pub normalized_text_blocks: u64,
    pub before_bytes: usize,
    pub after_bytes: usize,
    pub system_bytes: usize,
    pub messages_bytes: usize,
    pub tools_bytes: usize,
}

impl RequestOptimization {
    pub fn other_bytes(&self) -> usize {
        self.after_bytes
            .saturating_sub(self.system_bytes)
            .saturating_sub(self.messages_bytes)
            .saturating_sub(self.tools_bytes)
    }
}

/// Request origin: Anthropic clients carry reasoning controls as
/// `thinking`/`output_config.effort` (parsed by `convert_request`, never
/// forwarded on the wire); OpenAI-protocol clients express effort directly
/// via the `reasoning_effort` body field.
#[derive(Debug, Clone, Copy)]
pub enum Origin<'a> {
    Anthropic {
        thinking: Option<&'a Value>,
        output_effort: Option<&'a str>,
    },
    OpenAi,
}

/// Apply the upstream-model request policy to an OpenAI Chat Completions
/// body. GLM-5.3-Flash gets the full GLM policy; every other model gets a
/// compatibility passthrough (unknown models must fail safe toward
/// compatibility, not silently receive GLM semantics).
pub fn optimize_request(
    body: &mut Value,
    glm: &Glm53Config,
    origin: Origin<'_>,
) -> Result<RequestOptimization, String> {
    let Some(object) = body.as_object_mut() else {
        return Err("request body must be a JSON object".into());
    };
    let upstream_model = object
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let family = ModelFamily::from_upstream_model(&upstream_model);
    match family {
        ModelFamily::Glm53 => optimize_glm(object, glm, origin),
        ModelFamily::GenericOpenAi => optimize_generic(object),
    }
}

/// Generic OpenAI-compatible passthrough: no GLM fields are injected, no
/// GLM limits are applied. Byte telemetry still records what was seen.
fn optimize_generic(object: &mut Map<String, Value>) -> Result<RequestOptimization, String> {
    let mut optimization = RequestOptimization {
        model_family: ModelFamily::GenericOpenAi,
        reasoning_effort: "",
        before_bytes: serialized_len(&Value::Object(object.clone())),
        ..RequestOptimization::default()
    };
    optimization.after_bytes = optimization.before_bytes;
    compute_breakdown(object, &mut optimization);
    Ok(optimization)
}

/// GLM-5.3-Flash policy (the historical behavior of this module).
fn optimize_glm(
    object: &mut Map<String, Value>,
    glm: &Glm53Config,
    origin: Origin<'_>,
) -> Result<RequestOptimization, String> {
    let mut optimization = RequestOptimization {
        model_family: ModelFamily::Glm53,
        before_bytes: serialized_len(&Value::Object(object.clone())),
        ..RequestOptimization::default()
    };

    // --- 1. Reasoning policy (always explicit; never unset) ---
    let max_tokens = object
        .get("max_tokens")
        .and_then(Value::as_u64)
        .or_else(|| object.get("max_completion_tokens").and_then(Value::as_u64));
    let (thinking, output_effort) = match origin {
        Origin::Anthropic {
            thinking,
            output_effort,
        } => (thinking, output_effort),
        // OpenAI-protocol clients express effort directly.
        Origin::OpenAi => (None, object.get("reasoning_effort").and_then(Value::as_str)),
    };
    let policy = resolve_reasoning_policy(
        thinking,
        output_effort.map(openai_effort_to_glm),
        max_tokens.unwrap_or(u64::MAX),
        glm.reasoning.default_effort,
        glm.reasoning.adaptive_effort,
        glm.reasoning.expose_thinking,
    )?;
    optimization.reasoning_effort = policy.effort.as_str();
    optimization.expose_thinking = policy.expose_thinking;
    object.insert(
        "reasoning_effort".into(),
        Value::String(policy.effort.as_str().into()),
    );

    // --- 2. Output cap ---
    let configured_cap = u64::from(glm.limits.max_output_tokens);
    let effective = max_tokens.map(|client| client.min(configured_cap));
    optimization.client_max_tokens = max_tokens;
    optimization.effective_max_tokens = effective;
    match effective {
        Some(effective) => {
            if object.remove("max_completion_tokens").is_some() && max_tokens.is_none() {
                // max_completion_tokens was the only carrier; keep using it.
                object.insert("max_completion_tokens".into(), Value::from(effective));
            } else {
                object.insert("max_tokens".into(), Value::from(effective));
            }
        }
        None => {
            // No client bound at all: the configured cap becomes the bound.
            optimization.effective_max_tokens = Some(configured_cap);
            object.insert("max_tokens".into(), Value::from(configured_cap));
        }
    }

    // --- 3 + 4. Message-level strip and compaction ---
    if glm.reasoning.strip_historical_thinking || glm.context.safe_compaction {
        if let Some(messages) = object.get_mut("messages").and_then(Value::as_array_mut) {
            optimize_messages(messages, glm, &mut optimization);
        }
    }

    if matches!(origin, Origin::Anthropic { .. }) && glm.context.safe_compaction {
        // Anthropic-only request metadata has no OpenAI meaning and is
        // otherwise forwarded verbatim.
        object.remove("metadata");
    }

    optimization.after_bytes = serialized_len(&Value::Object(object.clone()));
    compute_breakdown(object, &mut optimization);
    Ok(optimization)
}

/// OpenAI-protocol effort vocabulary → GLM effort. `minimal` behaves like
/// `low`; unknown values fail closed (the request is rejected) instead of
/// being silently coerced to `max` by the upstream template.
fn openai_effort_to_glm(effort: &str) -> &str {
    match effort {
        "minimal" | "low" => "low",
        "medium" | "high" | "xhigh" => "high",
        other => other,
    }
}

fn optimize_messages(
    messages: &mut [Value],
    glm: &Glm53Config,
    optimization: &mut RequestOptimization,
) {
    let last_action_index = messages.iter().rposition(|message| {
        matches!(
            message.get("role").and_then(Value::as_str),
            Some("user") | Some("tool")
        )
    });
    for (index, message) in messages.iter_mut().enumerate() {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        if glm.reasoning.strip_historical_thinking
            && object.get("role").and_then(Value::as_str) == Some("assistant")
            && last_action_index.is_some_and(|last| index < last)
        {
            if let Some(removed) = object.remove("reasoning_content") {
                optimization.historical_reasoning_bytes_removed = optimization
                    .historical_reasoning_bytes_removed
                    .saturating_add(serialized_len(&removed) as u64);
            }
        }
        if glm.context.safe_compaction {
            compact_content(object, optimization);
        }
    }
}

/// Lossless content normalization for one message.
fn compact_content(object: &mut Map<String, Value>, optimization: &mut RequestOptimization) {
    let Some(Value::Array(blocks)) = object.get_mut("content") else {
        return;
    };
    blocks.retain(|block| {
        let empty = block.get("type").and_then(Value::as_str) == Some("text")
            && block
                .get("text")
                .and_then(Value::as_str)
                .is_some_and(str::is_empty);
        if empty {
            optimization.empty_blocks_removed = optimization.empty_blocks_removed.saturating_add(1);
        }
        !empty
    });
    if blocks.len() == 1
        && blocks[0].get("type").and_then(Value::as_str) == Some("text")
        && blocks[0].get("text").and_then(Value::as_str).is_some()
    {
        let text = blocks[0]["text"].take();
        optimization.normalized_text_blocks = optimization.normalized_text_blocks.saturating_add(1);
        object.insert("content".into(), text);
    } else if blocks.is_empty() {
        object.insert("content".into(), Value::String(String::new()));
    }
}

fn compute_breakdown(object: &Map<String, Value>, optimization: &mut RequestOptimization) {
    if let Some(messages) = object.get("messages").and_then(Value::as_array) {
        for message in messages {
            match message.get("role").and_then(Value::as_str) {
                Some("system") => {
                    optimization.system_bytes = optimization
                        .system_bytes
                        .saturating_add(serialized_len(message));
                }
                _ => {
                    optimization.messages_bytes = optimization
                        .messages_bytes
                        .saturating_add(serialized_len(message));
                }
            }
        }
    }
    if let Some(tools) = object.get("tools") {
        optimization.tools_bytes = optimization
            .tools_bytes
            .saturating_add(serialized_len(tools));
    }
}

fn serialized_len(value: &Value) -> usize {
    serde_json::to_vec(value)
        .map(|bytes| bytes.len())
        .unwrap_or(0)
}

/// Strip historical `thinking` blocks from an *Anthropic* request in place
/// (assistant blocks before the last user or tool_result message). Used for
/// exact token accounting of the optimized request; the wire-level strip
/// happens on the converted OpenAI body. Returns removed bytes.
pub fn strip_anthropic_thinking(request: &mut Value) -> usize {
    let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut) else {
        return 0;
    };
    let last_action_index = messages.iter().rposition(|message| {
        let is_user = message.get("role").and_then(Value::as_str) == Some("user");
        let has_tool_result = message
            .get("content")
            .and_then(Value::as_array)
            .is_some_and(|blocks| {
                blocks
                    .iter()
                    .any(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"))
            });
        is_user || has_tool_result
    });
    let mut removed = 0usize;
    let Some(last_action_index) = last_action_index else {
        return 0;
    };
    for message in messages.iter_mut().take(last_action_index) {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        let before = blocks.len();
        blocks.retain(|block| block.get("type").and_then(Value::as_str) != Some("thinking"));
        removed = removed.saturating_add(before.saturating_sub(blocks.len()));
    }
    removed
}

/// Exact token count of the removed historical reasoning chunks (the
/// difference between counting with and without them). Cheap: only the
/// removed strings are tokenized, never a second full pass.
pub fn removed_reasoning_tokens(request: &Value) -> Result<u64, String> {
    let mut chunks = Vec::new();
    collect_anthropic_thinking(request, &mut chunks);
    let mut total = 0u64;
    for chunk in chunks {
        let ids =
            crate::glm53::tokenizer::encode_no_special(&chunk).map_err(|error| error.message)?;
        total = total.saturating_add(ids.len() as u64);
    }
    Ok(total)
}

fn collect_anthropic_thinking(request: &Value, chunks: &mut Vec<String>) {
    let Some(messages) = request.get("messages").and_then(Value::as_array) else {
        return;
    };
    let last_action_index = messages
        .iter()
        .rposition(|message| {
            let is_user = message.get("role").and_then(Value::as_str) == Some("user");
            let has_tool_result = message
                .get("content")
                .and_then(Value::as_array)
                .is_some_and(|blocks| {
                    blocks.iter().any(|block| {
                        block.get("type").and_then(Value::as_str) == Some("tool_result")
                    })
                });
            is_user || has_tool_result
        })
        .unwrap_or(0);
    for message in messages.iter().take(last_action_index) {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(blocks) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some("thinking") {
                if let Some(text) = block.get("thinking").and_then(Value::as_str) {
                    chunks.push(text.to_owned());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn glm_config() -> Glm53Config {
        Glm53Config::default()
    }

    fn config_with(max_output: u32) -> Glm53Config {
        let mut config = glm_config();
        config.limits.max_output_tokens = max_output;
        config
    }

    #[test]
    fn anthropic_request_without_thinking_gets_explicit_high_and_cap() {
        let mut body = json!({
            "model": "z-ai/glm-5.3-flash",
            "max_tokens": 64_000,
            "messages": [{"role": "user", "content": "hello"}]
        });
        let optimization = optimize_request(
            &mut body,
            &config_with(16_384),
            Origin::Anthropic {
                thinking: None,
                output_effort: None,
            },
        )
        .unwrap();
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["max_tokens"], 16_384);
        assert_eq!(optimization.client_max_tokens, Some(64_000));
        assert_eq!(optimization.effective_max_tokens, Some(16_384));
        assert!(!optimization.expose_thinking);
    }

    #[test]
    fn output_cap_never_raises_the_client_bound() {
        for (client, expected) in [(4_096u64, 4_096u64), (16_384, 16_384), (64_000, 16_384)] {
            let mut body = json!({
                "model": "z-ai/glm-5.3-flash",
                "max_tokens": client,
                "messages": [{"role": "user", "content": "hi"}]
            });
            optimize_request(
                &mut body,
                &glm_config(),
                Origin::Anthropic {
                    thinking: None,
                    output_effort: None,
                },
            )
            .unwrap();
            assert_eq!(body["max_tokens"], expected, "client {client}");
        }
    }

    #[test]
    fn missing_client_max_tokens_gets_the_configured_cap() {
        let mut body = json!({
            "model": "z-ai/glm-5.3-flash",
            "messages": [{"role": "user", "content": "hi"}]
        });
        optimize_request(
            &mut body,
            &glm_config(),
            Origin::Anthropic {
                thinking: None,
                output_effort: None,
            },
        )
        .unwrap();
        assert_eq!(body["max_tokens"], 16_384);
    }

    #[test]
    fn historical_reasoning_is_stripped_but_tool_chain_is_intact() {
        let mut body = json!({
            "model": "z-ai/glm-5.3-flash",
            "max_tokens": 1_000,
            "messages": [
                {"role": "user", "content": "fix the bug"},
                {"role": "assistant", "content": "looking", "reasoning_content": "LONG OLD THINKING",
                 "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "Read", "arguments": "{\"path\":\"a\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "file body"},
                {"role": "assistant", "content": "again", "reasoning_content": "MORE THINKING",
                 "tool_calls": [{"id": "call_2", "type": "function",
                     "function": {"name": "Edit", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_2", "content": "done"}
            ]
        });
        let optimization = optimize_request(
            &mut body,
            &glm_config(),
            Origin::Anthropic {
                thinking: None,
                output_effort: None,
            },
        )
        .unwrap();
        let messages = body["messages"].as_array().unwrap();
        assert!(messages[1].get("reasoning_content").is_none());
        assert!(messages[3].get("reasoning_content").is_none());
        // Text, tool_calls, ids, and order survive untouched.
        assert_eq!(messages[1]["content"], "looking");
        assert_eq!(messages[1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(messages[3]["tool_calls"][0]["id"], "call_2");
        assert_eq!(messages[2]["tool_call_id"], "call_1");
        assert_eq!(messages[4]["tool_call_id"], "call_2");
        assert_eq!(
            optimization.historical_reasoning_bytes_removed as usize,
            "\"LONG OLD THINKING\"".len() + "\"MORE THINKING\"".len()
        );
    }

    #[test]
    fn reasoning_after_the_last_user_turn_is_kept() {
        let mut body = json!({
            "model": "z-ai/glm-5.3-flash",
            "max_tokens": 1_000,
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "reasoning_content": "kept prefill reasoning"}
            ]
        });
        optimize_request(
            &mut body,
            &glm_config(),
            Origin::Anthropic {
                thinking: None,
                output_effort: None,
            },
        )
        .unwrap();
        assert_eq!(
            body["messages"][1]["reasoning_content"],
            "kept prefill reasoning"
        );
    }

    #[test]
    fn safe_compaction_normalizes_and_drops_empty_blocks() {
        let mut body = json!({
            "model": "z-ai/glm-5.3-flash",
            "max_tokens": 1_000,
            "metadata": {"user_id": "anthropic-only"},
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "sys"}]},
                {"role": "user", "content": [
                    {"type": "text", "text": ""},
                    {"type": "text", "text": "real"}
                ]},
                {"role": "assistant", "content": []},
                {"role": "user", "content": "plain"}
            ]
        });
        let optimization = optimize_request(
            &mut body,
            &glm_config(),
            Origin::Anthropic {
                thinking: None,
                output_effort: None,
            },
        )
        .unwrap();
        assert_eq!(body["messages"][0]["content"], "sys");
        assert_eq!(body["messages"][1]["content"], "real");
        assert_eq!(body["messages"][2]["content"], "");
        assert_eq!(optimization.empty_blocks_removed, 1);
        assert_eq!(optimization.normalized_text_blocks, 2);
        assert!(body.get("metadata").is_none());
    }

    #[test]
    fn openai_passthrough_gets_default_effort_and_cap_without_thinking_controls() {
        let mut body = json!({
            "model": "z-ai/glm-5.3-flash",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let optimization = optimize_request(&mut body, &glm_config(), Origin::OpenAi).unwrap();
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["max_tokens"], 16_384);
        assert_eq!(optimization.reasoning_effort, "high");

        let mut explicit = json!({
            "model": "z-ai/glm-5.3-flash",
            "reasoning_effort": "medium",
            "max_tokens": 2_000,
            "messages": [{"role": "user", "content": "hi"}]
        });
        optimize_request(&mut explicit, &glm_config(), Origin::OpenAi).unwrap();
        assert_eq!(explicit["reasoning_effort"], "high");
        assert_eq!(explicit["max_tokens"], 2_000);

        let mut maximal = json!({
            "model": "z-ai/glm-5.3-flash",
            "reasoning_effort": "max",
            "max_tokens": 2_000,
            "messages": [{"role": "user", "content": "hi"}]
        });
        optimize_request(&mut maximal, &glm_config(), Origin::OpenAi).unwrap();
        assert_eq!(maximal["reasoning_effort"], "max");
    }

    #[test]
    fn breakdown_reflects_sections() {
        let mut body = json!({
            "model": "z-ai/glm-5.3-flash",
            "max_tokens": 1_000,
            "messages": [
                {"role": "system", "content": "system text"},
                {"role": "user", "content": "hello"}
            ],
            "tools": [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}]
        });
        let optimization = optimize_request(
            &mut body,
            &glm_config(),
            Origin::Anthropic {
                thinking: None,
                output_effort: None,
            },
        )
        .unwrap();
        assert!(optimization.system_bytes > 0);
        assert!(optimization.messages_bytes > 0);
        assert!(optimization.tools_bytes > 0);
        assert_eq!(
            optimization.system_bytes + optimization.messages_bytes + optimization.tools_bytes,
            optimization.after_bytes - optimization.other_bytes()
        );
    }

    #[test]
    fn compaction_disabled_preserves_structure() {
        let mut config = glm_config();
        config.context.safe_compaction = false;
        config.reasoning.strip_historical_thinking = false;
        let mut body = json!({
            "model": "z-ai/glm-5.3-flash",
            "max_tokens": 1_000,
            "metadata": {"user_id": "u"},
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "kept"}],
                 "reasoning_content": "kept too"},
                {"role": "user", "content": [{"type": "text", "text": ""}]}
            ]
        });
        optimize_request(
            &mut body,
            &config,
            Origin::Anthropic {
                thinking: None,
                output_effort: None,
            },
        )
        .unwrap();
        assert_eq!(body["messages"][0]["reasoning_content"], "kept too");
        assert_eq!(body["messages"][1]["content"][0]["text"], "");
        assert_eq!(body["metadata"]["user_id"], "u");
    }

    #[test]
    fn model_family_detection_is_conservative() {
        use ModelFamily::{GenericOpenAi, Glm53};
        for model in [
            "z-ai/glm-5.3-flash",
            "glm-5.3-flash",
            "GLM53",
            "glm4.7",
            "openai/glm-4.6",
            "zai/glm-4.5-air",
            "glm", // a full "glm" segment matches
        ] {
            assert_eq!(ModelFamily::from_upstream_model(model), Glm53, "{model}");
        }
        for model in ["claude-sonnet-4-6", "deepseek-chat", "gpt-5", "qwen3-coder"] {
            assert_eq!(
                ModelFamily::from_upstream_model(model),
                GenericOpenAi,
                "{model}"
            );
        }
        // Substrings inside unrelated tokens must not match.
        for model in ["aglm-4", "gpt-glmish", "kaggle"] {
            assert_eq!(
                ModelFamily::from_upstream_model(model),
                GenericOpenAi,
                "{model}"
            );
        }
    }

    /// Unknown models fail safe toward compatibility: no GLM effort, no GLM
    /// output cap, no message rewriting, no metadata removal.
    #[test]
    fn generic_models_receive_no_glm_semantics() {
        let mut body = json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 1_000,
            "metadata": {"user_id": "u"},
            "reasoning_effort": "medium",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "sys"}]},
                {"role": "user", "content": [{"type": "text", "text": ""}, {"type": "text", "text": "go"}]},
                {"role": "assistant", "content": "old", "reasoning_content": "OLD THINKING",
                 "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "Read", "arguments": "{\"path\":\"a\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "body"}
            ]
        });
        let before = body.clone();
        let optimization = optimize_request(&mut body, &glm_config(), Origin::OpenAi).unwrap();
        assert_eq!(optimization.model_family, ModelFamily::GenericOpenAi);
        assert_eq!(optimization.reasoning_effort, "");
        assert_eq!(optimization.effective_max_tokens, None);
        assert_eq!(optimization.historical_reasoning_bytes_removed, 0);
        assert_eq!(optimization.empty_blocks_removed, 0);
        assert_eq!(optimization.normalized_text_blocks, 0);
        // The body is byte-identical to the input (aside from nothing at all).
        assert_eq!(body, before);
        assert_eq!(body["reasoning_effort"], "medium");
        assert_eq!(body["max_tokens"], 1_000);
        assert!(body.get("metadata").is_some());
    }

    #[test]
    fn strip_anthropic_thinking_removes_only_historical_blocks() {
        let mut request = json!({
            "model": "z-ai/glm-5.3-flash",
            "max_tokens": 1_000,
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "old reasoning"},
                    {"type": "text", "text": "step"}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "current reasoning"},
                    {"type": "text", "text": "final"}
                ]}
            ]
        });
        strip_anthropic_thinking(&mut request);
        let messages = request["messages"].as_array().unwrap();
        assert_eq!(messages[1]["content"].as_array().unwrap().len(), 1);
        assert_eq!(messages[1]["content"][0]["type"], "text");
        assert_eq!(messages[3]["content"][0]["thinking"], "current reasoning");
        assert_eq!(messages[3]["content"][1]["text"], "final");
    }
}
