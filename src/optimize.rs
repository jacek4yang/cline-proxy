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
        before_bytes: serialized_object_len(object),
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
        before_bytes: serialized_object_len(object),
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

    optimization.after_bytes = serialized_object_len(object);
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
    // Reasoning-epoch boundary (issue #10): a `tool`-role message or a
    // pure-tool-result user message continues the current epoch; the strip
    // removes reasoning only from epochs *before* the newest human turn.
    // OpenAI protocol has no block-array user content in this code path, so
    // the boundary is the last plain `user` message (a `tool` role alone is
    // never a human turn).
    let epoch_boundary = messages
        .iter()
        .rposition(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        .unwrap_or(messages.len());
    for (index, message) in messages.iter_mut().enumerate() {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        if glm.reasoning.strip_historical_thinking
            && object.get("role").and_then(Value::as_str) == Some("assistant")
            && index < epoch_boundary
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

/// Serialized length of a request-level object without cloning the map
/// into a temporary `Value` (a 1 MB request would otherwise be copied for
/// every size probe).
fn serialized_object_len(object: &Map<String, Value>) -> usize {
    serde_json::to_vec(object)
        .map(|bytes| bytes.len())
        .unwrap_or(0)
}

/// Strip historical `thinking` blocks from an *Anthropic* request in place.
/// The reasoning-epoch boundary is the newest **human** user message — a
/// `user` message that carries ordinary content, NOT a pure `tool_result`
/// carrier (a tool result continues the current assistant reasoning epoch,
/// it does not start a new one). Assistant thinking before that boundary is
/// removed; current-epoch thinking is preserved for in-epoch continuity
/// (issue #10). Used for exact token accounting of the optimized request;
/// the wire-level strip happens on the converted OpenAI body. Returns
/// removed bytes.
pub fn strip_anthropic_thinking(request: &mut Value) -> usize {
    let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut) else {
        return 0;
    };
    let epoch_boundary = reasoning_epoch_boundary(messages);
    let mut removed = 0usize;
    for message in messages.iter_mut().take(epoch_boundary) {
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

/// Index of the newest human user message (the reasoning-epoch boundary):
/// a `user` message that contains any content other than tool_result
/// blocks. Everything at and after that index belongs to the current
/// epoch. When no human turn exists, the epoch covers the whole history
/// (returns `messages.len()`).
fn reasoning_epoch_boundary(messages: &[Value]) -> usize {
    messages
        .iter()
        .rposition(|message| {
            if message.get("role").and_then(Value::as_str) != Some("user") {
                return false;
            }
            let has_human_content = match message.get("content") {
                // A plain string is by definition human content.
                Some(Value::String(text)) => !text.is_empty(),
                Some(Value::Array(blocks)) => blocks
                    .iter()
                    .any(|block| block.get("type").and_then(Value::as_str) != Some("tool_result")),
                _ => false,
            };
            has_human_content
        })
        .unwrap_or(messages.len())
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
    let epoch_boundary = reasoning_epoch_boundary(messages);
    for message in messages.iter().take(epoch_boundary) {
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
        // Two epochs: epoch 1 = user "fix the bug" + its tool loop; epoch 2
        // starts at the second human turn. Only epoch-1 reasoning is
        // historical.
        let mut body = json!({
            "model": "z-ai/glm-5.3-flash",
            "max_tokens": 1_000,
            "messages": [
                {"role": "user", "content": "fix the bug"},
                {"role": "assistant", "content": "looking", "reasoning_content": "LONG OLD THINKING",
                 "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "Read", "arguments": "{\"path\":\"a\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "file body"},
                {"role": "user", "content": "thanks, now make it faster"},
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
        // Epoch-1 reasoning stripped; epoch-2 (current) reasoning kept.
        assert!(messages[1].get("reasoning_content").is_none());
        assert!(messages[4].get("reasoning_content").is_some());
        // Text, tool_calls, ids, and order survive untouched.
        assert_eq!(messages[1]["content"], "looking");
        assert_eq!(messages[1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(messages[4]["tool_calls"][0]["id"], "call_2");
        assert_eq!(messages[2]["tool_call_id"], "call_1");
        assert_eq!(messages[5]["tool_call_id"], "call_2");
        assert_eq!(
            optimization.historical_reasoning_bytes_removed as usize,
            "\"LONG OLD THINKING\"".len()
        );
    }

    /// Within one epoch, tool-loop reasoning continuity is preserved on the
    /// wire (issue #10): tool results do not erase the reasoning of the
    /// assistant turns they belong to.
    #[test]
    fn same_epoch_tool_loop_reasoning_is_preserved() {
        let mut body = json!({
            "model": "z-ai/glm-5.3-flash",
            "max_tokens": 1_000,
            "messages": [
                {"role": "user", "content": "fix the bug"},
                {"role": "assistant", "content": "looking", "reasoning_content": "EPOCH CURRENT A",
                 "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "Read", "arguments": "{\"path\":\"a\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "file body"},
                {"role": "assistant", "content": "again", "reasoning_content": "EPOCH CURRENT B",
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
        assert_eq!(optimization.historical_reasoning_bytes_removed, 0);
        assert_eq!(messages[1]["reasoning_content"], "EPOCH CURRENT A");
        assert_eq!(messages[3]["reasoning_content"], "EPOCH CURRENT B");
        assert_tool_chain_intact_helper(&body);
    }

    fn assert_tool_chain_intact_helper(body: &Value) {
        let messages = body["messages"].as_array().unwrap();
        let mut pending = std::collections::BTreeSet::new();
        for message in messages {
            if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    pending.insert(call["id"].as_str().unwrap().to_owned());
                }
            }
            if message.get("role").and_then(Value::as_str) == Some("tool") {
                assert!(pending.remove(message["tool_call_id"].as_str().unwrap()));
            }
        }
        assert!(pending.is_empty());
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
    fn strip_anthropic_thinking_follows_human_turn_epoch_boundary() {
        // Same epoch: user task -> assistant thinking -> tool_result ->
        // assistant thinking. A tool_result does NOT start a new epoch, so
        // NOTHING is stripped until a new human turn arrives.
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
        assert_eq!(strip_anthropic_thinking(&mut request), 0);
        let messages = request["messages"].as_array().unwrap();
        assert_eq!(
            messages[1]["content"][0]["thinking"], "old reasoning",
            "same-epoch reasoning is preserved"
        );
        assert_eq!(messages[3]["content"][0]["thinking"], "current reasoning");

        // After a NEW human turn, all earlier-epoch thinking is stripped.
        request["messages"].as_array_mut().unwrap().push(json!(
            {"role": "user", "content": "now do something else"}
        ));
        let removed = strip_anthropic_thinking(&mut request);
        assert_eq!(removed, 2, "both epoch-1 thinking blocks are historical");
        let messages = request["messages"].as_array().unwrap();
        // Only the text blocks survive; order preserved.
        assert_eq!(messages[1]["content"].as_array().unwrap().len(), 1);
        assert_eq!(messages[1]["content"][0]["type"], "text");
        assert_eq!(messages[1]["content"][0]["text"], "step");
        assert_eq!(messages[3]["content"].as_array().unwrap().len(), 1);
        assert_eq!(messages[3]["content"][0]["type"], "text");
        assert_eq!(messages[3]["content"][0]["text"], "final");
    }

    #[test]
    fn pure_tool_result_user_turn_never_starts_an_epoch() {
        // Long tool loop within one epoch: strip boundary must skip past
        // tool_result-only user messages even when they contain text too?
        // No — a user message mixing tool_result AND human text starts a
        // new epoch (it contains ordinary content).
        let mut request = json!({
            "model": "z-ai/glm-5.3-flash",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "t1"}, {"type": "text", "text": "a"}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "r"},
                    {"type": "text", "text": "actually also do X"}
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "t2"}, {"type": "text", "text": "b"}
                ]}
            ]
        });
        // The mixed user message at index 2 IS a human turn (has text).
        assert_eq!(strip_anthropic_thinking(&mut request), 1);
        let messages = request["messages"].as_array().unwrap();
        assert_eq!(messages[1]["content"][0].get("thinking"), None);
        assert_eq!(messages[3]["content"][0]["thinking"], "t2");
    }
}
