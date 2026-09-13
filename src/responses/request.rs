//! Responses API request → OpenAI Chat Completions conversion.
//!
//! Compatibility target: Grok Build (`api_backend = "responses"`), verified
//! against the Responses wire schema. The request is normalized (never
//! forwarded blindly) into the same OpenAI Chat Completions body the
//! Anthropic frontend produces, so all shared policy (GLM reasoning epochs,
//! prefix stability, canonical tool arguments) applies unchanged.
//!
//! Conversion rules:
//! - `input` string → one user message; array items convert in order
//!   (ordering is replayed conversation state — never reordered);
//! - `function_call` history → assistant `tool_calls` with the SAME
//!   `call_id`; consecutive calls of one assistant turn are merged into one
//!   message (matching how the model emits them); `function_call_output` →
//!   `role=tool` with `tool_call_id = call_id`;
//! - `reasoning` input items are dropped (historical reasoning is stripped
//!   by policy for every frontend; text/tool semantics are unaffected);
//! - flat Responses function tools → nested OpenAI function tools, schemas
//!   byte-identical;
//! - `max_output_tokens` → `max_tokens` (never dropped);
//! - `prompt_cache_key` → forwarded verbatim to the upstream;
//! - unknown optional fields (`store`, `metadata`, `previous_response_id`,
//!   `truncation`, `stream_options`, …) are ignored safely;
//! - hosted backend tools (`web_search`, `x_search`, …) are dropped, not
//!   forwarded: this proxy cannot execute them and never fakes results, and
//!   the model must not be offered a tool whose call could never be answered
//!   (Grok Build sends a default `web_search` declaration routinely; dropping
//!   it lets the request proceed with the client function tools).

use serde_json::{json, Map, Value};

use super::types::ProtocolError;

/// A converted Responses request, ready for the shared generation path.
#[derive(Debug, Clone)]
pub struct Converted {
    /// OpenAI Chat Completions body (without `stream` — forced upstream).
    pub chat_body: Value,
    /// Whether the client requested reasoning exposure via
    /// `reasoning.summary` (Responses equivalent of Anthropic thinking).
    pub thinking_requested: bool,
    /// The client-requested model id (informational only).
    pub client_model: Option<String>,
    /// The raw `prompt_cache_key` (fingerprinted by the caller before any
    /// use as a session identity; forwarded verbatim to the upstream).
    pub prompt_cache_key: Option<String>,
}

/// Convert a Responses request into the normalized chat body.
pub fn convert_request(request: &Value, default_model: &str) -> Result<Converted, ProtocolError> {
    let object = request
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("request body must be a JSON object"))?;

    // --- model & sampling ---
    let client_model = object
        .get("model")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let mut chat = Map::new();
    chat.insert("model".into(), Value::String(default_model.to_owned()));

    // max_output_tokens → max_tokens. Optional in Responses (unlike Anthropic)
    // but never silently dropped when present.
    if let Some(max) = object.get("max_output_tokens") {
        let max = max.as_u64().filter(|v| *v > 0).ok_or_else(|| {
            ProtocolError::invalid("max_output_tokens must be a positive integer")
        })?;
        chat.insert("max_tokens".into(), json!(max));
    }
    if let Some(temperature) = object.get("temperature").and_then(Value::as_f64) {
        chat.insert("temperature".into(), json!(temperature));
    }
    if let Some(top_p) = object.get("top_p").and_then(Value::as_f64) {
        chat.insert("top_p".into(), json!(top_p));
    }
    // prompt_cache_key: sticky-routing + upstream prompt-cache hint. Forwarded
    // verbatim; the caller fingerprints it before using it as a session
    // identity.
    let prompt_cache_key = object
        .get("prompt_cache_key")
        .and_then(Value::as_str)
        .filter(|key| !key.is_empty())
        .map(ToOwned::to_owned);
    if let Some(key) = &prompt_cache_key {
        chat.insert("prompt_cache_key".into(), json!(key));
    }

    // --- reasoning: map effort to the existing upstream mechanism ---
    // Grok Build sends `reasoning: {effort, summary}` (summary always
    // "concise" in practice). There is no second reasoning system here:
    // - effort low/medium/high/xhigh/max → `reasoning_effort` on the wire
    //   (the same field the GLM policy resolves;
    //   `crate::optimize::openai_effort_to_glm` performs the final GLM
    //   mapping). `minimal`/`none` mean "do not emphasize reasoning" and map
    //   to `low` rather than being dropped, so an explicit request never
    //   silently upgrades to the upstream default.
    // - summary present → reasoning may be EXPOSED to the client as a
    //   Responses reasoning item (requested-only exposure, matching the
    //   Anthropic frontend's `thinking: enabled` gate).
    let mut thinking_requested = false;
    if let Some(reasoning) = object.get("reasoning") {
        let reasoning = reasoning
            .as_object()
            .ok_or_else(|| ProtocolError::invalid("reasoning must be an object"))?;
        if let Some(effort) = reasoning.get("effort").and_then(Value::as_str) {
            let wire = match effort {
                "none" | "minimal" | "low" => "low",
                "medium" | "high" | "xhigh" | "max" => "high",
                other => {
                    return Err(ProtocolError::invalid(format!(
                        "reasoning.effort {other:?} is not supported (expected one of none, minimal, low, medium, high, xhigh, max)"
                    )))
                }
            };
            chat.insert("reasoning_effort".into(), json!(wire));
        }
        thinking_requested = reasoning
            .get("summary")
            .and_then(Value::as_str)
            .is_some_and(|summary| summary != "none")
            || reasoning.get("effort").and_then(Value::as_str).is_none();
    }

    // --- messages ---
    chat.insert("messages".into(), Value::Array(convert_input(object)?));

    // --- tools ---
    if let Some(tools) = object.get("tools") {
        let tools = tools
            .as_array()
            .ok_or_else(|| ProtocolError::invalid("tools must be an array"))?;
        if !tools.is_empty() {
            let converted = convert_tools(tools)?;
            // Every declaration may have been a dropped hosted tool; an
            // empty tools array is omitted rather than forwarded.
            if !converted.is_empty() {
                chat.insert("tools".into(), Value::Array(converted));
            }
        }
    }
    if let Some(choice) = object.get("tool_choice") {
        if let Some(converted) = convert_tool_choice(choice)? {
            chat.insert("tool_choice".into(), converted);
        }
    }

    Ok(Converted {
        chat_body: Value::Object(chat),
        thinking_requested,
        client_model,
        prompt_cache_key,
    })
}

// ---------------------------------------------------------------------------
// input items → chat messages
// ---------------------------------------------------------------------------

/// Convert the Responses `input` field. Supports the string convenience form
/// and the full item array (both `EasyInputMessage` and typed items). Order
/// is preserved exactly — this is replayed conversation state.
fn convert_input(object: &Map<String, Value>) -> Result<Vec<Value>, ProtocolError> {
    match object.get("input") {
        Some(Value::String(text)) => Ok(vec![json!({"role": "user", "content": text})]),
        Some(Value::Array(items)) => convert_input_items(items),
        Some(_) => Err(ProtocolError::invalid(
            "input must be a string or an array of items",
        )),
        None => Err(ProtocolError::invalid("input is required")),
    }
}

fn convert_input_items(items: &[Value]) -> Result<Vec<Value>, ProtocolError> {
    let mut messages: Vec<Value> = Vec::with_capacity(items.len());
    // Consecutive function_call items belong to one assistant turn; merged
    // into a single message with multiple tool_calls (never merged across a
    // message/output boundary, never given new ids).
    let mut pending_calls: Vec<Value> = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let obj = item
            .as_object()
            .ok_or_else(|| ProtocolError::invalid(format!("input[{index}] must be an object")))?;
        let kind = obj.get("type").and_then(Value::as_str).unwrap_or("message");
        match kind {
            "message" => convert_message_item(obj, index, &mut messages)?,
            "function_call" => {
                let call = convert_function_call(obj, index)?;
                pending_calls.push(call);
            }
            "function_call_output" => {
                flush_calls(&mut messages, &mut pending_calls);
                messages.push(convert_function_call_output(obj, index)?);
            }
            "reasoning" => {
                // Historical reasoning is deliberately not replayed onto the
                // chat wire: policy strips historical reasoning for every
                // frontend (epoch boundary), and `encrypted_content` is
                // provider-opaque. Text/tool-call semantics are unaffected;
                // user/assistant/tool ordering is preserved.
                flush_calls(&mut messages, &mut pending_calls);
            }
            "item_reference" => {
                return Err(ProtocolError::invalid(
                    "input item type \"item_reference\" is not supported (this proxy is stateless; replay full items)",
                ))
            }
            other => {
                return Err(ProtocolError::invalid(format!(
                    "input[{index}]: unsupported item type {other:?}"
                )))
            }
        }
    }
    flush_calls(&mut messages, &mut pending_calls);
    if messages.is_empty() {
        return Err(ProtocolError::invalid("input must not be empty"));
    }
    Ok(messages)
}

fn flush_calls(messages: &mut Vec<Value>, pending: &mut Vec<Value>) {
    if !pending.is_empty() {
        let mut message = Map::new();
        message.insert("role".into(), json!("assistant"));
        message.insert("content".into(), Value::Null);
        message.insert("tool_calls".into(), Value::Array(std::mem::take(pending)));
        messages.push(Value::Object(message));
    }
}

/// One `type: "message"` item: role + string or content-part array.
fn convert_message_item(
    obj: &Map<String, Value>,
    index: usize,
    messages: &mut Vec<Value>,
) -> Result<(), ProtocolError> {
    let role = obj.get("role").and_then(Value::as_str).ok_or_else(|| {
        ProtocolError::invalid(format!("input[{index}]: message requires a role"))
    })?;
    let content = obj.get("content").cloned().unwrap_or(Value::Null);
    let text = match content {
        Value::String(text) => text,
        Value::Array(parts) => {
            let mut text_parts = Vec::new();
            for (part_index, part) in parts.iter().enumerate() {
                let part = part.as_object().ok_or_else(|| {
                    ProtocolError::invalid(format!(
                        "input[{index}].content[{part_index}] must be an object"
                    ))
                })?;
                match part.get("type").and_then(Value::as_str) {
                    // input_text (input shapes) and output_text (assistant
                    // replay) both carry plain text.
                    Some("input_text") | Some("output_text") => {
                        text_parts.push(
                            part.get("text")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                        );
                    }
                    Some("input_image") => {
                        return Err(ProtocolError::invalid(format!(
                            "input[{index}]: image inputs are not supported by this model"
                        )))
                    }
                    Some("refusal") => text_parts.push(
                        part.get("refusal")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    ),
                    Some(other) => {
                        return Err(ProtocolError::invalid(format!(
                        "input[{index}].content[{part_index}]: unsupported content type {other:?}"
                    )))
                    }
                    None => {
                        return Err(ProtocolError::invalid(format!(
                            "input[{index}].content[{part_index}] requires a type"
                        )))
                    }
                }
            }
            text_parts.join("\n")
        }
        Value::Null => String::new(),
        _ => {
            return Err(ProtocolError::invalid(format!(
                "input[{index}].content must be a string or an array"
            )))
        }
    };
    match role {
        "system" | "developer" => messages.push(json!({"role": "system", "content": text})),
        "user" => messages.push(json!({"role": "user", "content": text})),
        "assistant" => {
            if text.is_empty() {
                // Assistant replay with no text: keep the turn valid without
                // inventing content.
                messages.push(json!({"role": "assistant", "content": Value::Null}));
            } else {
                messages.push(json!({"role": "assistant", "content": text}));
            }
        }
        other => {
            return Err(ProtocolError::invalid(format!(
                "input[{index}]: unsupported message role {other:?}"
            )))
        }
    }
    Ok(())
}

/// `type: "function_call"` → one OpenAI tool call. `call_id` is preserved
/// byte-identical end-to-end (never regenerated).
fn convert_function_call(obj: &Map<String, Value>, index: usize) -> Result<Value, ProtocolError> {
    let call_id = obj
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            ProtocolError::invalid(format!("input[{index}]: function_call requires call_id"))
        })?;
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            ProtocolError::invalid(format!("input[{index}]: function_call requires name"))
        })?;
    // Arguments are a JSON-encoded string on the wire. Malformed strings are
    // forwarded as-is (the model authored them; rewriting could corrupt).
    let arguments = obj
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("{}")
        .to_owned();
    Ok(json!({
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments}
    }))
}

/// `type: "function_call_output"` → `role=tool` with the same `call_id`.
fn convert_function_call_output(
    obj: &Map<String, Value>,
    index: usize,
) -> Result<Value, ProtocolError> {
    let call_id = obj
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            ProtocolError::invalid(format!(
                "input[{index}]: function_call_output requires call_id"
            ))
        })?;
    // Output may be a plain string or a content-part list; text parts are
    // joined in order. Tool results are never truncated or rewritten.
    let content = match obj.get("output") {
        None | Some(Value::Null) => Value::String(String::new()),
        Some(Value::String(text)) => Value::String(text.clone()),
        Some(Value::Array(parts)) => {
            let mut text_parts = Vec::new();
            for (part_index, part) in parts.iter().enumerate() {
                let part = part.as_object().ok_or_else(|| {
                    ProtocolError::invalid(format!(
                        "input[{index}].output[{part_index}] must be an object"
                    ))
                })?;
                match part.get("type").and_then(Value::as_str) {
                    Some("input_text") | Some("output_text") => text_parts.push(
                        part.get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    ),
                    Some("input_image") => {
                        return Err(ProtocolError::invalid(format!(
                            "input[{index}]: image tool outputs are not supported by this model"
                        )))
                    }
                    Some(other) => {
                        return Err(ProtocolError::invalid(format!(
                        "input[{index}].output[{part_index}]: unsupported content type {other:?}"
                    )))
                    }
                    None => {
                        return Err(ProtocolError::invalid(format!(
                            "input[{index}].output[{part_index}] requires a type"
                        )))
                    }
                }
            }
            Value::String(text_parts.join("\n"))
        }
        Some(_) => {
            return Err(ProtocolError::invalid(format!(
                "input[{index}]: function_call_output.output must be a string or an array"
            )))
        }
    };
    Ok(json!({"role": "tool", "tool_call_id": call_id, "content": content}))
}

// ---------------------------------------------------------------------------
// tools & tool_choice
// ---------------------------------------------------------------------------

/// Flat Responses function tool → nested OpenAI function tool. The `parameters`
/// schema passes through byte-identical (never truncated or rewritten).
fn convert_tools(tools: &[Value]) -> Result<Vec<Value>, ProtocolError> {
    let mut out = Vec::with_capacity(tools.len());
    for (index, tool) in tools.iter().enumerate() {
        let obj = tool
            .as_object()
            .ok_or_else(|| ProtocolError::invalid(format!("tools[{index}] must be an object")))?;
        match obj.get("type").and_then(Value::as_str) {
            // Already OpenAI-shaped (defensive; Grok never sends this).
            Some("function") if obj.contains_key("function") => out.push(tool.clone()),
            Some("function") | None => {
                let name = obj
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| {
                        ProtocolError::invalid(format!("tools[{index}].name is required"))
                    })?;
                let mut function = Map::new();
                function.insert("name".into(), json!(name));
                if let Some(description) = obj.get("description").and_then(Value::as_str) {
                    function.insert("description".into(), json!(description));
                }
                if let Some(schema) = obj.get("parameters") {
                    if !schema.is_object() {
                        return Err(ProtocolError::invalid(format!(
                            "tools[{index}].parameters must be an object"
                        )));
                    }
                    function.insert("parameters".into(), schema.clone());
                }
                if let Some(strict) = obj.get("strict") {
                    if strict.is_boolean() {
                        function.insert("strict".into(), strict.clone());
                    }
                }
                out.push(json!({"type": "function", "function": function}));
            }
            // Backend-hosted tools (web_search, x_search, code_interpreter,
            // MCP, …): this proxy cannot execute them and never fakes
            // results. Drop the declaration instead of failing the request —
            // Grok Build sends a default `web_search` entry routinely, and a
            // hard 400 would break every session with backend search left at
            // its default. The model simply never sees a tool it could call
            // but whose result could never be produced; client function
            // tools continue to pass through.
            Some(other) => {
                tracing::debug!(
                    tool_type = other,
                    tool_index = index,
                    "dropped backend-hosted tool declaration (not executable by this proxy)"
                );
            }
        }
    }
    Ok(out)
}

/// Responses `tool_choice` → OpenAI `tool_choice`. Grok sends the string
/// modes or `{type: "function", name}`; unsupported shapes are rejected.
fn convert_tool_choice(choice: &Value) -> Result<Option<Value>, ProtocolError> {
    match choice {
        Value::String(mode) => match mode.as_str() {
            "auto" | "none" | "required" => Ok(Some(json!(mode))),
            other => Err(ProtocolError::invalid(format!(
                "unsupported tool_choice mode {other:?}"
            ))),
        },
        Value::Object(obj) => match obj.get("type").and_then(Value::as_str) {
            Some("function") => {
                let name = obj
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| {
                        ProtocolError::invalid("tool_choice.type=function requires name")
                    })?;
                Ok(Some(
                    json!({"type": "function", "function": {"name": name}}),
                ))
            }
            Some("auto") | Some("none") | Some("required") | Some("allowed_tools") => {
                Ok(Some(json!("auto")))
            }
            Some(other) => Err(ProtocolError::invalid(format!(
                "unsupported tool_choice type {other:?}"
            ))),
            None => Err(ProtocolError::invalid("tool_choice requires a type")),
        },
        Value::Null => Ok(None),
        _ => Err(ProtocolError::invalid(
            "tool_choice must be a string or object",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_input_and_field_mapping() {
        let request = json!({
            "model": "grok-4",
            "input": "hello",
            "max_output_tokens": 512,
            "temperature": 0.2,
            "reasoning": {"effort": "high", "summary": "concise"},
            "prompt_cache_key": "conv-123",
            "store": false,
            "metadata": {"x": "y"},
            "previous_response_id": "resp_old"
        });
        let converted = convert_request(&request, "z-ai/glm-5.3-flash").unwrap();
        let body = converted.chat_body;
        assert_eq!(body["model"], "z-ai/glm-5.3-flash");
        assert_eq!(body["max_tokens"], 512);
        assert_eq!(body["temperature"], 0.2);
        assert_eq!(body["prompt_cache_key"], "conv-123");
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hello");
        assert!(converted.thinking_requested);
        // Unknown optional fields are dropped, not forwarded.
        assert!(body.get("store").is_none());
        assert!(body.get("metadata").is_none());
        assert!(body.get("previous_response_id").is_none());
    }

    #[test]
    fn structured_items_and_developer_role() {
        let request = json!({
            "input": [
                {"type": "message", "role": "developer",
                 "content": [{"type": "input_text", "text": "Be terse."}]},
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "part one"},
                    {"type": "input_text", "text": "part two"}
                ]}
            ]
        });
        let converted = convert_request(&request, "m").unwrap();
        let messages = converted.chat_body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "Be terse.");
        assert_eq!(messages[1]["content"], "part one\npart two");
    }

    #[test]
    fn tool_history_roundtrip_preserves_call_ids() {
        let request = json!({
            "input": [
                {"type": "message", "role": "user", "content": "fix it"},
                {"type": "function_call", "call_id": "call_AAA", "name": "Read",
                 "arguments": "{\"path\": \"a.rs\"}"},
                {"type": "function_call", "call_id": "call_BBB", "name": "Edit",
                 "arguments": "{\"path\": \"a.rs\", \"text\": \"x\"}"},
                {"type": "function_call_output", "call_id": "call_AAA", "output": "old body"},
                {"type": "function_call_output", "call_id": "call_BBB", "output": "ok"},
                {"type": "message", "role": "user", "content": "done?"}
            ]
        });
        let converted = convert_request(&request, "m").unwrap();
        let messages = converted.chat_body["messages"].as_array().unwrap();
        let roles: Vec<&str> = messages
            .iter()
            .map(|m| m["role"].as_str().unwrap_or_default())
            .collect();
        assert_eq!(roles, vec!["user", "assistant", "tool", "tool", "user"]);
        let calls = messages[1]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["id"], "call_AAA");
        assert_eq!(calls[1]["id"], "call_BBB");
        assert_eq!(messages[2]["tool_call_id"], "call_AAA");
        assert_eq!(messages[2]["content"], "old body");
        assert_eq!(messages[3]["tool_call_id"], "call_BBB");
    }

    #[test]
    fn reasoning_items_dropped_without_breaking_chain() {
        let request = json!({
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "reasoning", "id": "rs_x",
                 "summary": [{"type": "summary_text", "text": "hmm"}]},
                {"type": "function_call", "call_id": "call_C", "name": "Read",
                 "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_C", "output": "ok"}
            ]
        });
        let converted = convert_request(&request, "m").unwrap();
        let messages = converted.chat_body["messages"].as_array().unwrap();
        let roles: Vec<&str> = messages
            .iter()
            .map(|m| m["role"].as_str().unwrap_or_default())
            .collect();
        assert_eq!(roles, vec!["user", "assistant", "tool"]);
    }

    #[test]
    fn hosted_tools_dropped_and_flat_tools_nested() {
        let request = json!({
            "input": "hi",
            "tools": [
                {"type": "web_search"},
                {"type": "function", "name": "Read",
                 "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}
            ]
        });
        let converted = convert_request(&request, "m").unwrap();
        let tools = converted.chat_body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "Read");
        assert_eq!(
            tools[0]["function"]["parameters"]["properties"]["path"]["type"],
            "string"
        );
    }

    #[test]
    fn rejects_bad_effort_item_reference_and_empty_input() {
        let request = json!({"input": "hi", "reasoning": {"effort": "bogus"}});
        assert!(convert_request(&request, "m").is_err());
        let request = json!({"input": [{"type": "item_reference", "id": "x"}]});
        assert!(convert_request(&request, "m").is_err());
        let request = json!({"input": []});
        assert!(convert_request(&request, "m").is_err());
    }
}
