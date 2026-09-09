//! Anthropic Messages v1 compatibility over OpenAI Chat Completions.
//!
//! The conversion and SSE lifecycle are adapted from mogick-proxy's mature
//! provider-independent adapter. Provider-specific compaction, OAuth, and
//! strict-output repair behavior are intentionally not carried over.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use axum::body::Body;
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Map, Value};

const MAX_UPSTREAM_JSON_BYTES: usize = 16 * 1024 * 1024;
const MAX_UPSTREAM_SSE_EVENT_BYTES: usize = 8 * 1024 * 1024;
#[cfg(not(test))]
const STREAM_PING_INTERVAL: Duration = Duration::from_secs(10);
#[cfg(test)]
const STREAM_PING_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Debug, Clone)]
pub struct ProtocolError {
    pub error_type: &'static str,
    pub message: String,
}

impl ProtocolError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            error_type: "invalid_request_error",
            message: message.into(),
        }
    }

    pub fn upstream(message: impl Into<String>) -> Self {
        Self {
            error_type: "api_error",
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ProtocolError {}

#[derive(Debug)]
pub struct ConvertedRequest {
    pub body: Value,
    pub model: String,
    pub stream: bool,
    /// Explicit Anthropic reasoning controls, preserved for
    /// `crate::optimize::optimize_request` (the single resolution point).
    /// Never forwarded to the OpenAI wire.
    pub thinking: Option<Value>,
    pub output_effort: Option<String>,
    /// Whether upstream reasoning may be surfaced as Anthropic thinking
    /// blocks. Set by the GLM policy step; false until then.
    pub expose_thinking: bool,
}

pub fn convert_request(bytes: &[u8]) -> Result<ConvertedRequest, ProtocolError> {
    let input: Value = serde_json::from_slice(bytes)
        .map_err(|error| ProtocolError::invalid(format!("invalid JSON: {error}")))?;
    let object = input
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("request body must be a JSON object"))?;
    let model = required_string(object, "model")?.to_owned();
    let max_tokens = object
        .get("max_tokens")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| ProtocolError::invalid("max_tokens must be a positive integer"))?;
    let stream = optional_bool(object, "stream")?.unwrap_or(false);
    let thinking = object.get("thinking").cloned();
    let output_effort = object
        .get("output_config")
        .and_then(|config| config.get("effort"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);

    let mut output = Map::new();
    output.insert("model".into(), Value::String(model.clone()));
    output.insert("max_tokens".into(), Value::from(max_tokens));
    output.insert("stream".into(), Value::Bool(stream));

    let mut messages = Vec::new();
    if let Some(system) = object.get("system") {
        messages.push(json!({"role":"system", "content":convert_system(system)?}));
    }
    let input_messages = object
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| ProtocolError::invalid("messages must be an array"))?;
    for (index, message) in input_messages.iter().enumerate() {
        convert_message(message, index, &mut messages)?;
    }
    output.insert("messages".into(), Value::Array(messages));

    copy_number(object, &mut output, "temperature")?;
    copy_number(object, &mut output, "top_p")?;
    copy_positive_integer(object, &mut output, "top_k")?;
    if let Some(stop) = object.get("stop_sequences") {
        let sequences = stop
            .as_array()
            .ok_or_else(|| ProtocolError::invalid("stop_sequences must be an array"))?;
        if sequences.iter().any(|value| value.as_str().is_none()) {
            return Err(ProtocolError::invalid(
                "stop_sequences entries must be strings",
            ));
        }
        output.insert("stop".into(), stop.clone());
    }
    if let Some(metadata) = object.get("metadata") {
        if !metadata.is_object() {
            return Err(ProtocolError::invalid("metadata must be an object"));
        }
        output.insert("metadata".into(), metadata.clone());
    }
    if let Some(tools) = object.get("tools") {
        output.insert("tools".into(), convert_tools(tools)?);
    }
    if let Some(choice) = object.get("tool_choice") {
        let (choice, parallel) = convert_tool_choice(choice)?;
        output.insert("tool_choice".into(), choice);
        if let Some(parallel) = parallel {
            output.insert("parallel_tool_calls".into(), Value::Bool(parallel));
        }
    }
    // `thinking` and `output_config.effort` are resolved by
    // `crate::optimize::optimize_request` against the GLM policy config, so
    // there is exactly one mapping and every request leaves with an explicit
    // `reasoning_effort` (unset would coerce to `max` upstream).
    if let Some(config) = object.get("output_config") {
        if !config.is_object() {
            return Err(ProtocolError::invalid("output_config must be an object"));
        }
        if let Some(format) = config.get("format") {
            output.insert("response_format".into(), convert_output_format(format)?);
        }
    }
    if let Some(format) = object.get("output_format") {
        output.insert("response_format".into(), convert_output_format(format)?);
    }
    if stream {
        output.insert("stream_options".into(), json!({"include_usage":true}));
    }
    Ok(ConvertedRequest {
        body: Value::Object(output),
        model,
        stream,
        thinking,
        output_effort,
        expose_thinking: false,
    })
}

pub fn apply_model(converted: &mut ConvertedRequest, model: String) {
    converted.model = model.clone();
    converted.body["model"] = Value::String(model);
}

fn convert_system(value: &Value) -> Result<Value, ProtocolError> {
    if value.is_string() {
        return Ok(value.clone());
    }
    let blocks = value
        .as_array()
        .ok_or_else(|| ProtocolError::invalid("system must be a string or content block array"))?;
    blocks
        .iter()
        .enumerate()
        .map(|(index, block)| {
            let object = block.as_object().ok_or_else(|| {
                ProtocolError::invalid(format!("system[{index}] must be an object"))
            })?;
            if required_string(object, "type")? != "text" {
                return Err(ProtocolError::invalid("system only supports text blocks"));
            }
            Ok(json!({"type":"text", "text":required_string(object, "text")?}))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Value::Array)
}

/// Normalize the converted OpenAI body's system message(s) for prefix
/// stability: strip a *leading* `x-anthropic-billing-header:` line (its
/// dynamic attribution metadata would otherwise change the system prefix
/// every turn), then join consecutive system messages into one so the
/// message order is always `system... user...` regardless of how the
/// client split its system content (issue #8). Returns removed bytes.
pub fn normalize_system_messages(body: &mut Value) -> u64 {
    let Some(object) = body.as_object_mut() else {
        return 0;
    };
    let Some(messages) = object.get_mut("messages").and_then(Value::as_array_mut) else {
        return 0;
    };
    // Strip a leading billing header from every system message's text.
    let mut removed_bytes = 0u64;
    for message in messages.iter_mut() {
        if message.get("role").and_then(Value::as_str) != Some("system") {
            continue;
        }
        let strip = |text: &str| -> String {
            let cut = crate::cache::strip_leading_anthropic_billing_header(text);
            if cut == 0 {
                return text.to_owned();
            }
            text[cut..].to_owned()
        };
        match message.get_mut("content") {
            Some(Value::String(text)) => {
                let stripped = strip(text);
                removed_bytes =
                    removed_bytes.saturating_add(text.len().saturating_sub(stripped.len()) as u64);
                message["content"] = Value::String(stripped);
            }
            Some(Value::Array(blocks)) => {
                for block in blocks.iter_mut() {
                    if block.get("type").and_then(Value::as_str) == Some("text") {
                        if let Some(Value::String(text)) = block.get_mut("text") {
                            let stripped = strip(text);
                            removed_bytes = removed_bytes
                                .saturating_add(text.len().saturating_sub(stripped.len()) as u64);
                            *text = stripped;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    removed_bytes
}

fn convert_message(
    value: &Value,
    message_index: usize,
    output: &mut Vec<Value>,
) -> Result<(), ProtocolError> {
    let object = value.as_object().ok_or_else(|| {
        ProtocolError::invalid(format!("messages[{message_index}] must be an object"))
    })?;
    let role = required_string(object, "role")?;
    if !matches!(role, "user" | "assistant" | "system") {
        return Err(ProtocolError::invalid(format!(
            "messages[{message_index}].role must be user, assistant, or system"
        )));
    }
    let content = object
        .get("content")
        .ok_or_else(|| ProtocolError::invalid("message content is required"))?;
    if role == "system" {
        output.push(json!({"role":"system", "content":convert_system(content)?}));
    } else if content.is_string() {
        output.push(json!({"role":role, "content":content}));
    } else {
        let blocks = content
            .as_array()
            .ok_or_else(|| ProtocolError::invalid("message content must be a string or array"))?;
        if role == "assistant" {
            convert_assistant_blocks(blocks, output)?;
        } else {
            convert_user_blocks(blocks, output)?;
        }
    }
    Ok(())
}

fn convert_assistant_blocks(
    blocks: &[Value],
    output: &mut Vec<Value>,
) -> Result<(), ProtocolError> {
    let mut content = Vec::new();
    let mut tool_calls = Vec::new();
    let mut reasoning = String::new();
    for block in blocks {
        let object = block
            .as_object()
            .ok_or_else(|| ProtocolError::invalid("assistant content block must be an object"))?;
        match required_string(object, "type")? {
            "text" => content.push(convert_text_block(object)?),
            "thinking" => {
                if let Some(text) = object.get("thinking").and_then(Value::as_str) {
                    reasoning.push_str(text);
                }
            }
            "redacted_thinking" => {}
            "tool_use" => {
                let arguments = serde_json::to_string(
                    object.get("input").unwrap_or(&Value::Object(Map::new())),
                )
                .map_err(|error| ProtocolError::invalid(error.to_string()))?;
                tool_calls.push(json!({
                    "id":required_string(object, "id")?,
                    "type":"function",
                    "function":{
                        "name":required_string(object, "name")?,
                        "arguments":arguments
                    }
                }));
            }
            kind => {
                return Err(ProtocolError::invalid(format!(
                    "unsupported assistant content block type {kind:?}"
                )))
            }
        }
    }
    let mut message = Map::new();
    message.insert("role".into(), Value::String("assistant".into()));
    message.insert("content".into(), Value::Array(content));
    if !tool_calls.is_empty() {
        message.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    if !reasoning.is_empty() {
        message.insert("reasoning_content".into(), Value::String(reasoning));
    }
    output.push(Value::Object(message));
    Ok(())
}

fn convert_user_blocks(blocks: &[Value], output: &mut Vec<Value>) -> Result<(), ProtocolError> {
    let mut ordinary = Vec::new();
    for block in blocks {
        let object = block
            .as_object()
            .ok_or_else(|| ProtocolError::invalid("user content block must be an object"))?;
        match required_string(object, "type")? {
            "text" => ordinary.push(convert_text_block(object)?),
            "image" => ordinary.push(convert_image_block(object)?),
            "document" => ordinary.push(convert_document_block(object)?),
            "tool_result" => {
                flush_user_content(&mut ordinary, output);
                output.push(convert_tool_result(object)?);
            }
            kind => {
                return Err(ProtocolError::invalid(format!(
                    "unsupported user content block type {kind:?}"
                )))
            }
        }
    }
    flush_user_content(&mut ordinary, output);
    if blocks.is_empty() {
        output.push(json!({"role":"user", "content":[]}));
    }
    Ok(())
}

fn flush_user_content(content: &mut Vec<Value>, output: &mut Vec<Value>) {
    if !content.is_empty() {
        output.push(json!({"role":"user", "content":std::mem::take(content)}));
    }
}

fn convert_text_block(object: &Map<String, Value>) -> Result<Value, ProtocolError> {
    Ok(json!({"type":"text", "text":required_string(object, "text")?}))
}

fn convert_image_block(object: &Map<String, Value>) -> Result<Value, ProtocolError> {
    let source = object
        .get("source")
        .and_then(Value::as_object)
        .ok_or_else(|| ProtocolError::invalid("image source must be an object"))?;
    let url = match required_string(source, "type")? {
        "base64" => format!(
            "data:{};base64,{}",
            required_string(source, "media_type")?,
            required_string(source, "data")?
        ),
        "url" => required_string(source, "url")?.to_owned(),
        kind => {
            return Err(ProtocolError::invalid(format!(
                "unsupported image source {kind:?}"
            )))
        }
    };
    Ok(json!({"type":"image_url", "image_url":{"url":url}}))
}

fn convert_document_block(object: &Map<String, Value>) -> Result<Value, ProtocolError> {
    let source = object
        .get("source")
        .and_then(Value::as_object)
        .ok_or_else(|| ProtocolError::invalid("document source must be an object"))?;
    match required_string(source, "type")? {
        "text" => Ok(json!({"type":"text", "text":required_string(source, "data")?})),
        "url" => Ok(json!({"type":"file", "file":{"file_url":required_string(source, "url")?}})),
        "base64" => Ok(json!({"type":"file", "file":{
            "filename":object.get("title").and_then(Value::as_str).unwrap_or("document"),
            "file_data":format!("data:{};base64,{}",
                required_string(source, "media_type")?, required_string(source, "data")?)
        }})),
        kind => Err(ProtocolError::invalid(format!(
            "unsupported document source {kind:?}"
        ))),
    }
}

fn convert_tool_result(object: &Map<String, Value>) -> Result<Value, ProtocolError> {
    let content = object
        .get("content")
        .cloned()
        .unwrap_or_else(|| Value::String(String::new()));
    let content = if let Some(blocks) = content.as_array() {
        Value::Array(
            blocks
                .iter()
                .map(|block| {
                    let object = block.as_object().ok_or_else(|| {
                        ProtocolError::invalid("tool_result content block must be an object")
                    })?;
                    match required_string(object, "type")? {
                        "text" => convert_text_block(object),
                        "image" => convert_image_block(object),
                        "document" => convert_document_block(object),
                        kind => Err(ProtocolError::invalid(format!(
                            "unsupported tool_result content type {kind:?}"
                        ))),
                    }
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else if content.is_string() {
        content
    } else {
        return Err(ProtocolError::invalid(
            "tool_result content must be a string or array",
        ));
    };
    let mut result = json!({
        "role":"tool",
        "tool_call_id":required_string(object, "tool_use_id")?,
        "content":content
    });
    if let Some(is_error) = object.get("is_error") {
        if !is_error.is_boolean() {
            return Err(ProtocolError::invalid(
                "tool_result is_error must be boolean",
            ));
        }
        result["is_error"] = is_error.clone();
    }
    Ok(result)
}

fn convert_tools(value: &Value) -> Result<Value, ProtocolError> {
    let tools = value
        .as_array()
        .ok_or_else(|| ProtocolError::invalid("tools must be an array"))?;
    tools
        .iter()
        .map(|tool| {
            let object = tool
                .as_object()
                .ok_or_else(|| ProtocolError::invalid("tool must be an object"))?;
            let schema = object
                .get("input_schema")
                .filter(|value| value.is_object())
                .ok_or_else(|| ProtocolError::invalid("tool input_schema must be an object"))?;
            let mut function = json!({
                "name":required_string(object, "name")?,
                "parameters":schema
            });
            if let Some(description) = object.get("description") {
                if !description.is_string() {
                    return Err(ProtocolError::invalid("tool description must be a string"));
                }
                function["description"] = description.clone();
            }
            if let Some(strict) = object.get("strict") {
                if !strict.is_boolean() {
                    return Err(ProtocolError::invalid("tool strict must be a boolean"));
                }
                function["strict"] = strict.clone();
            }
            Ok(json!({"type":"function", "function":function}))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Value::Array)
}

fn convert_tool_choice(value: &Value) -> Result<(Value, Option<bool>), ProtocolError> {
    let object = value
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("tool_choice must be an object"))?;
    let choice = match required_string(object, "type")? {
        "auto" => Value::String("auto".into()),
        "any" => Value::String("required".into()),
        "none" => Value::String("none".into()),
        "tool" => json!({"type":"function", "function":{"name":required_string(object, "name")?}}),
        kind => {
            return Err(ProtocolError::invalid(format!(
                "unsupported tool_choice type {kind:?}"
            )))
        }
    };
    let parallel = object
        .get("disable_parallel_tool_use")
        .map(|value| {
            value
                .as_bool()
                .map(|disabled| !disabled)
                .ok_or_else(|| ProtocolError::invalid("disable_parallel_tool_use must be boolean"))
        })
        .transpose()?;
    Ok((choice, parallel))
}

fn convert_output_format(value: &Value) -> Result<Value, ProtocolError> {
    let object = value
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("output format must be an object"))?;
    match required_string(object, "type")? {
        "json_object" => Ok(json!({"type":"json_object"})),
        "json_schema" => {
            let schema = object
                .get("schema")
                .filter(|value| value.is_object())
                .ok_or_else(|| ProtocolError::invalid("output schema must be an object"))?;
            Ok(json!({"type":"json_schema", "json_schema":{
                "name":object.get("name").and_then(Value::as_str).unwrap_or("response"),
                "schema":schema,
                "strict":object.get("strict").and_then(Value::as_bool).unwrap_or(true)
            }}))
        }
        kind => Err(ProtocolError::invalid(format!(
            "unsupported output format type {kind:?}"
        ))),
    }
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    name: &str,
) -> Result<&'a str, ProtocolError> {
    object
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| ProtocolError::invalid(format!("{name} must be a non-empty string")))
}

fn optional_bool(object: &Map<String, Value>, name: &str) -> Result<Option<bool>, ProtocolError> {
    object
        .get(name)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| ProtocolError::invalid(format!("{name} must be a boolean")))
        })
        .transpose()
}

fn copy_number(
    input: &Map<String, Value>,
    output: &mut Map<String, Value>,
    name: &str,
) -> Result<(), ProtocolError> {
    if let Some(value) = input.get(name) {
        if !value.is_number() {
            return Err(ProtocolError::invalid(format!("{name} must be a number")));
        }
        output.insert(name.into(), value.clone());
    }
    Ok(())
}

fn copy_positive_integer(
    input: &Map<String, Value>,
    output: &mut Map<String, Value>,
    name: &str,
) -> Result<(), ProtocolError> {
    if let Some(value) = input.get(name) {
        if value.as_u64().filter(|value| *value > 0).is_none() {
            return Err(ProtocolError::invalid(format!(
                "{name} must be a positive integer"
            )));
        }
        output.insert(name.into(), value.clone());
    }
    Ok(())
}

pub fn convert_response(
    upstream: &Value,
    request_id: &str,
    fallback_model: &str,
    expose_thinking: bool,
) -> Result<Value, ProtocolError> {
    let choice = upstream
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| ProtocolError::upstream("upstream response did not contain a choice"))?;
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| ProtocolError::upstream("upstream choice did not contain a message"))?;
    let mut content = Vec::new();
    // Anti-amplification gate: reasoning reaches the client only when the
    // request explicitly asked for thinking. Unexposed reasoning still cost
    // this turn's output tokens, but it can never be stored, replayed, and
    // re-counted by Claude Code on later turns.
    let reasoning = if expose_thinking {
        reasoning_text(message)
    } else {
        String::new()
    };
    if !reasoning.is_empty() {
        content.push(json!({
            "type":"thinking",
            "thinking":reasoning,
            "signature":thinking_signature(&reasoning, request_id)
        }));
    }
    append_message_text(message.get("content"), &mut content);
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for call in calls {
            let function = call
                .get("function")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    ProtocolError::upstream("upstream returned a malformed tool call")
                })?;
            let arguments = function
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("");
            let input = if arguments.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(arguments).map_err(|_| {
                    ProtocolError::upstream("upstream returned malformed tool arguments")
                })?
            };
            content.push(json!({
                "type":"tool_use",
                "id":call.get("id").and_then(Value::as_str).unwrap_or("toolu_unknown"),
                "name":function.get("name").and_then(Value::as_str).unwrap_or("unknown"),
                "input":input
            }));
        }
    }
    Ok(json!({
        "id":upstream.get("id").and_then(Value::as_str).unwrap_or(request_id),
        "type":"message",
        "role":"assistant",
        "content":content,
        "model":upstream.get("model").and_then(Value::as_str).unwrap_or(fallback_model),
        "stop_reason":map_stop_reason(choice.get("finish_reason").and_then(Value::as_str)),
        "stop_sequence":choice.get("stop_sequence").cloned().unwrap_or(Value::Null),
        "usage":convert_usage(upstream.get("usage"))
    }))
}

fn append_message_text(content: Option<&Value>, output: &mut Vec<Value>) {
    match content {
        Some(Value::String(text)) if !text.is_empty() => {
            output.push(json!({"type":"text", "text":text}));
        }
        Some(Value::Array(blocks)) => {
            for block in blocks {
                if let Some(text) = block
                    .get("text")
                    .or_else(|| block.get("content"))
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                {
                    output.push(json!({"type":"text", "text":text}));
                }
            }
        }
        _ => {}
    }
}

fn reasoning_text(object: &Map<String, Value>) -> String {
    for name in ["reasoning_content", "reasoning"] {
        if let Some(text) = object.get(name).and_then(Value::as_str) {
            return text.to_owned();
        }
    }
    let Some(details) = object.get("reasoning_details") else {
        return String::new();
    };
    match details {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                item.as_str().or_else(|| {
                    item.get("text")
                        .or_else(|| item.get("content"))
                        .and_then(Value::as_str)
                })
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn thinking_signature(reasoning: &str, request_id: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    reasoning.hash(&mut hasher);
    request_id.hash(&mut hasher);
    format!("cline-proxy-v1-{:016x}", hasher.finish())
}

fn convert_usage(usage: Option<&Value>) -> Value {
    let output = usage
        .and_then(|usage| usage.get("completion_tokens"))
        .and_then(Value::as_u64)
        .or_else(|| {
            usage
                .and_then(|usage| usage.get("output_tokens"))
                .and_then(Value::as_u64)
        })
        .unwrap_or(0);
    let cached = usage
        .and_then(|usage| usage.get("prompt_tokens_details"))
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .or_else(|| {
            usage
                .and_then(|usage| usage.get("cache_read_input_tokens"))
                .and_then(Value::as_u64)
        })
        .unwrap_or(0);
    let total_input = usage
        .and_then(|usage| usage.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .or_else(|| {
            usage
                .and_then(|usage| usage.get("input_tokens"))
                .and_then(Value::as_u64)
        })
        .unwrap_or(0);
    let cache_creation = usage
        .and_then(|usage| usage.get("cache_creation_input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    json!({
        "input_tokens":total_input.saturating_sub(cached).saturating_sub(cache_creation),
        "output_tokens":output,
        "cache_creation_input_tokens":cache_creation,
        "cache_read_input_tokens":cached
    })
}

fn map_stop_reason(reason: Option<&str>) -> Value {
    match reason {
        Some("length" | "max_tokens") => Value::String("max_tokens".into()),
        Some("tool_calls" | "function_call") => Value::String("tool_use".into()),
        Some("content_filter" | "refusal") => Value::String("refusal".into()),
        Some("pause_turn") => Value::String("pause_turn".into()),
        Some(_) => Value::String("end_turn".into()),
        None => Value::Null,
    }
}

pub fn error_envelope(error_type: &str, message: impl Into<String>, request_id: &str) -> Value {
    json!({
        "type":"error",
        "error":{"type":error_type, "message":message.into()},
        "request_id":request_id
    })
}

pub struct StreamShadowContext {
    pub store: std::sync::Arc<crate::reasoning_shadow::ReasoningShadowStore>,
    pub session_fingerprint: String,
}

/// Options for [`stream_body`] beyond the response itself.
pub struct StreamOptions {
    pub request_id: String,
    pub fallback_model: String,
    pub key_name: String,
    pub request_started: Instant,
    pub progress_secs: u64,
    pub expose_thinking: bool,
    pub shadow: Option<StreamShadowContext>,
}

pub fn stream_body(response: reqwest::Response, options: StreamOptions) -> Body {
    let StreamOptions {
        request_id,
        fallback_model,
        key_name,
        request_started,
        progress_secs,
        expose_thinking,
        shadow,
    } = options;
    let output = async_stream::stream! {
        let mut upstream = response.bytes_stream();
        let idle = tokio::time::sleep(STREAM_PING_INTERVAL);
        tokio::pin!(idle);
        let progress_interval = Duration::from_secs(progress_secs.max(1));
        let progress = tokio::time::sleep(progress_interval);
        tokio::pin!(progress);
        let mut decoder = SseDecoder::default();
        let mut state = StreamState::new(request_id, fallback_model, expose_thinking, request_started);
        let mut telemetry = StreamTelemetry::new(
            state.request_id.clone(),
            key_name,
            request_started,
        );
        loop {
            tokio::select! {
                biased;
                chunk = upstream.next() => match chunk {
                    Some(Ok(chunk)) => {
                        telemetry.upstream_chunks = telemetry.upstream_chunks.saturating_add(1);
                        telemetry.upstream_bytes = telemetry.upstream_bytes.saturating_add(chunk.len() as u64);
                        let events = match decoder.push(&chunk) {
                            Ok(events) => events,
                            Err(()) => {
                                let frame = sse_frame("error", error_envelope(
                                    "api_error", "upstream SSE event exceeded the gateway limit", &state.request_id,
                                ));
                                telemetry.commit(frame.len());
                                yield Ok::<Bytes, std::io::Error>(frame);
                                telemetry.absorb(&state);
                                telemetry.finish("decode_error");
                                return;
                            }
                        };
                        for event in events {
                            telemetry.upstream_events = telemetry.upstream_events.saturating_add(1);
                            telemetry.first_event();
                            for frame in state.handle(&event.data) {
                                idle.as_mut().reset(tokio::time::Instant::now() + STREAM_PING_INTERVAL);
                                telemetry.commit(frame.len());
                                yield Ok::<Bytes, std::io::Error>(frame);
                            }
                            if state.terminal {
                                state.commit_shadow(shadow.as_ref());
                                telemetry.absorb(&state);
                                telemetry.finish(if state.finish_reason.is_some() { "complete" } else { "protocol_error" });
                                return;
                            }
                        }
                    }
                    Some(Err(error)) => {
                        tracing::warn!(
                            request_id = %state.request_id,
                            error_class = if error.is_timeout() { "timeout" } else { "stream_transport" },
                            committed_to_client = telemetry.committed_to_client,
                            "upstream Anthropic stream interrupted; request will not be replayed"
                        );
                        let frame = sse_frame("error", error_envelope(
                            "api_error", "upstream stream was interrupted", &state.request_id,
                        ));
                        telemetry.commit(frame.len());
                        yield Ok(frame);
                        telemetry.absorb(&state);
                        telemetry.finish("upstream_error");
                        return;
                    }
                    None => break,
                },
                _ = &mut idle => {
                    let frame = sse_frame("ping", json!({"type":"ping"}));
                    telemetry.commit(frame.len());
                    yield Ok(frame);
                    idle.as_mut().reset(tokio::time::Instant::now() + STREAM_PING_INTERVAL);
                }
                _ = &mut progress, if progress_secs > 0 => {
                    telemetry.log_progress();
                    progress.as_mut().reset(tokio::time::Instant::now() + progress_interval);
                }
            }
        }
        let trailing = match decoder.finish() {
            Ok(events) => events,
            Err(()) => {
                let frame = sse_frame("error", error_envelope(
                    "api_error", "upstream SSE event exceeded the gateway limit", &state.request_id,
                ));
                telemetry.commit(frame.len());
                yield Ok(frame);
                telemetry.absorb(&state);
                telemetry.finish("decode_error");
                return;
            }
        };
        for event in trailing {
            telemetry.upstream_events = telemetry.upstream_events.saturating_add(1);
            telemetry.first_event();
            for frame in state.handle(&event.data) {
                telemetry.commit(frame.len());
                yield Ok(frame);
            }
        }
        if !state.terminal {
            if state.finish_reason.is_some() {
                for frame in state.finalize() {
                    telemetry.commit(frame.len());
                    yield Ok(frame);
                }
                state.commit_shadow(shadow.as_ref());
                telemetry.absorb(&state);
                telemetry.finish("complete");
            } else {
                let frame = sse_frame("error", error_envelope(
                    "api_error", "upstream stream ended unexpectedly", &state.request_id,
                ));
                telemetry.commit(frame.len());
                yield Ok(frame);
                telemetry.absorb(&state);
                telemetry.finish("unexpected_eof");
            }
        }
    };
    Body::from_stream(output)
}

struct StreamTelemetry {
    request_id: String,
    key_name: String,
    started: Instant,
    upstream_chunks: u64,
    upstream_bytes: u64,
    upstream_events: u64,
    downstream_frames: u64,
    downstream_bytes: u64,
    committed_to_client: bool,
    saw_first_event: bool,
    finished: bool,
    /// Output composition accounting, absorbed from the stream state at
    /// completion. Bytes are SSE delta payload sizes, not tokens.
    reasoning_bytes: u64,
    text_bytes: u64,
    tool_call_bytes: u64,
    reasoning_events: u64,
    text_events: u64,
    tool_call_events: u64,
    first_reasoning_ms: Option<u128>,
    first_text_ms: Option<u128>,
    first_tool_call_ms: Option<u128>,
    usage: Option<Value>,
}

impl StreamTelemetry {
    fn new(request_id: String, key_name: String, started: Instant) -> Self {
        Self {
            request_id,
            key_name,
            started,
            upstream_chunks: 0,
            upstream_bytes: 0,
            upstream_events: 0,
            downstream_frames: 0,
            downstream_bytes: 0,
            committed_to_client: false,
            saw_first_event: false,
            finished: false,
            reasoning_bytes: 0,
            text_bytes: 0,
            tool_call_bytes: 0,
            reasoning_events: 0,
            text_events: 0,
            tool_call_events: 0,
            first_reasoning_ms: None,
            first_text_ms: None,
            first_tool_call_ms: None,
            usage: None,
        }
    }

    /// Copy the output-composition counters the stream state collected.
    fn absorb(&mut self, state: &StreamState) {
        self.reasoning_bytes = state.reasoning_bytes;
        self.text_bytes = state.text_bytes;
        self.tool_call_bytes = state.tool_call_bytes;
        self.reasoning_events = state.reasoning_events;
        self.text_events = state.text_events;
        self.tool_call_events = state.tool_call_events;
        self.first_reasoning_ms = state.first_reasoning;
        self.first_text_ms = state.first_text;
        self.first_tool_call_ms = state.first_tool_call;
        if state.usage.is_object() && !state.usage.as_object().is_some_and(Map::is_empty) {
            self.usage = Some(state.usage.clone());
        }
    }

    fn first_event(&mut self) {
        if self.saw_first_event {
            return;
        }
        self.saw_first_event = true;
        tracing::info!(
            request_id = %self.request_id,
            selected_key_name = %self.key_name,
            time_to_first_event_ms = self.started.elapsed().as_millis(),
            "Anthropic stream received first upstream event"
        );
    }

    fn commit(&mut self, bytes: usize) {
        // This flag documents the replay boundary. No retry code is reachable
        // from the body stream, before or after it becomes true.
        self.committed_to_client = true;
        self.downstream_frames = self.downstream_frames.saturating_add(1);
        self.downstream_bytes = self.downstream_bytes.saturating_add(bytes as u64);
    }

    fn log_progress(&self) {
        tracing::info!(
            request_id = %self.request_id,
            selected_key_name = %self.key_name,
            elapsed_ms = self.started.elapsed().as_millis(),
            upstream_chunks = self.upstream_chunks,
            upstream_bytes = self.upstream_bytes,
            upstream_events = self.upstream_events,
            downstream_frames = self.downstream_frames,
            downstream_bytes = self.downstream_bytes,
            committed_to_client = self.committed_to_client,
            reasoning_bytes = self.reasoning_bytes,
            text_bytes = self.text_bytes,
            tool_call_bytes = self.tool_call_bytes,
            "Anthropic stream still active"
        );
    }

    fn finish(&mut self, outcome: &'static str) {
        if self.finished {
            return;
        }
        self.finished = true;
        let usage = self.usage.as_ref();
        // Only report token figures the upstream actually provided; byte
        // counters above are always real measurements and never converted.
        let prompt_tokens = usage
            .and_then(|usage| usage.get("prompt_tokens"))
            .and_then(Value::as_u64);
        let completion_tokens = usage
            .and_then(|usage| usage.get("completion_tokens"))
            .and_then(Value::as_u64);
        let cached_tokens = usage
            .and_then(|usage| usage.get("prompt_tokens_details"))
            .and_then(|details| details.get("cached_tokens"))
            .and_then(Value::as_u64);
        let reasoning_tokens = usage
            .and_then(|usage| usage.get("completion_tokens_details"))
            .and_then(|details| details.get("reasoning_tokens"))
            .and_then(Value::as_u64);
        // Ratios (issue #8): only computed from figures the upstream
        // actually reported. OpenAI semantics: `prompt_tokens` INCLUDES
        // `cached_tokens` (cached is a subset, reported in
        // prompt_tokens_details), so hit ratio = cached/prompt. If a future
        // upstream reports them as disjoint, this must be revisited.
        let cache_hit_ratio = match (cached_tokens, prompt_tokens) {
            (Some(cached), Some(prompt)) if prompt > 0 => {
                Some((cached.min(prompt) as f64 / prompt as f64 * 1000.0).round() / 10.0)
            }
            _ => None,
        };
        let reasoning_ratio = match (reasoning_tokens, completion_tokens) {
            (Some(reasoning), Some(completion)) if completion > 0 => {
                Some((reasoning.min(completion) as f64 / completion as f64 * 1000.0).round() / 10.0)
            }
            _ => None,
        };
        tracing::info!(
            request_id = %self.request_id,
            selected_key_name = %self.key_name,
            outcome,
            duration_ms = self.started.elapsed().as_millis(),
            upstream_chunks = self.upstream_chunks,
            upstream_events = self.upstream_events,
            downstream_frames = self.downstream_frames,
            committed_to_client = self.committed_to_client,
            reasoning_bytes = self.reasoning_bytes,
            text_bytes = self.text_bytes,
            tool_call_bytes = self.tool_call_bytes,
            reasoning_events = self.reasoning_events,
            text_events = self.text_events,
            tool_call_events = self.tool_call_events,
            first_reasoning_ms = self.first_reasoning_ms.unwrap_or(0),
            first_text_ms = self.first_text_ms.unwrap_or(0),
            first_tool_call_ms = self.first_tool_call_ms.unwrap_or(0),
            prompt_tokens = prompt_tokens.unwrap_or(0),
            completion_tokens = completion_tokens.unwrap_or(0),
            cached_tokens = cached_tokens.unwrap_or(0),
            reasoning_tokens = reasoning_tokens.unwrap_or(0),
            cache_hit_ratio = cache_hit_ratio,
            reasoning_ratio = reasoning_ratio,
            usage_present = usage.is_some(),
            "Anthropic stream closed"
        );
    }
}

impl Drop for StreamTelemetry {
    fn drop(&mut self) {
        if !self.finished {
            self.finish("client_disconnected");
        }
    }
}

#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
    event: Option<String>,
    data_lines: Vec<String>,
    data_bytes: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct SseEvent {
    #[allow(dead_code)]
    event: Option<String>,
    data: String,
}

impl SseDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<SseEvent>, ()> {
        if self.buffer.len().saturating_add(bytes.len()) > MAX_UPSTREAM_SSE_EVENT_BYTES {
            self.clear();
            return Err(());
        }
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some(position) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..=position).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            self.process_line(&String::from_utf8_lossy(&line), &mut events);
            if self.data_bytes > MAX_UPSTREAM_SSE_EVENT_BYTES {
                self.clear();
                return Err(());
            }
        }
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<SseEvent>, ()> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            let line = String::from_utf8_lossy(&std::mem::take(&mut self.buffer)).into_owned();
            self.process_line(&line, &mut events);
        }
        if self.data_bytes > MAX_UPSTREAM_SSE_EVENT_BYTES {
            self.clear();
            return Err(());
        }
        self.dispatch(&mut events);
        Ok(events)
    }

    fn process_line(&mut self, line: &str, events: &mut Vec<SseEvent>) {
        if line.is_empty() {
            self.dispatch(events);
        } else if line.starts_with(':') {
            // Comment/keepalive.
        } else if let Some(value) = line.strip_prefix("data:") {
            let value = value.strip_prefix(' ').unwrap_or(value);
            self.data_bytes = self.data_bytes.saturating_add(value.len());
            self.data_lines.push(value.to_owned());
        } else if let Some(value) = line.strip_prefix("event:") {
            self.event = Some(value.strip_prefix(' ').unwrap_or(value).to_owned());
        }
    }

    fn dispatch(&mut self, events: &mut Vec<SseEvent>) {
        if self.data_lines.is_empty() {
            self.event = None;
            return;
        }
        events.push(SseEvent {
            event: self.event.take(),
            data: self.data_lines.join("\n"),
        });
        self.data_lines.clear();
        self.data_bytes = 0;
    }

    fn clear(&mut self) {
        self.buffer.clear();
        self.event = None;
        self.data_lines.clear();
        self.data_bytes = 0;
    }
}

struct StreamState {
    request_id: String,
    fallback_model: String,
    started: bool,
    terminal: bool,
    /// Anti-amplification gate: when false, upstream reasoning is counted
    /// but never emitted as Anthropic thinking blocks.
    expose_thinking: bool,
    blocks: Vec<BlockKind>,
    text_index: Option<usize>,
    thinking_index: Option<usize>,
    tools: HashMap<u64, ToolStream>,
    finish_reason: Option<String>,
    stop_sequence: Value,
    usage: Value,
    /// Output-composition accounting (SSE delta payload bytes; never
    /// converted to tokens — upstream usage supplies tokens when present).
    reasoning_bytes: u64,
    text_bytes: u64,
    tool_call_bytes: u64,
    reasoning_events: u64,
    text_events: u64,
    tool_call_events: u64,
    first_reasoning: Option<u128>,
    first_text: Option<u128>,
    first_tool_call: Option<u128>,
    request_started: Instant,
    /// Reasoning shadow (issue #10): full reasoning text accumulated when
    /// NOT exposed to the client, for the shadow store. Kept only for the
    /// stream duration; empty when exposed (client already has it) or no
    /// shadow context was provided.
    shadow_reasoning: String,
}

#[derive(Clone, Copy)]
enum BlockKind {
    Text,
    Thinking,
    Tool,
}

#[derive(Default)]
struct ToolStream {
    block_index: Option<usize>,
    id: Option<String>,
    name: Option<String>,
    pending_arguments: String,
}

impl StreamState {
    fn new(
        request_id: String,
        fallback_model: String,
        expose_thinking: bool,
        request_started: Instant,
    ) -> Self {
        Self {
            request_id,
            fallback_model,
            started: false,
            terminal: false,
            expose_thinking,
            blocks: Vec::new(),
            text_index: None,
            thinking_index: None,
            tools: HashMap::new(),
            finish_reason: None,
            stop_sequence: Value::Null,
            usage: json!({}),
            reasoning_bytes: 0,
            text_bytes: 0,
            tool_call_bytes: 0,
            reasoning_events: 0,
            text_events: 0,
            tool_call_events: 0,
            first_reasoning: None,
            first_text: None,
            first_tool_call: None,
            request_started,
            shadow_reasoning: String::new(),
        }
    }

    fn mark_first(kind: &mut Option<u128>, request_started: Instant) {
        if kind.is_none() {
            *kind = Some(request_started.elapsed().as_millis());
        }
    }

    fn handle(&mut self, data: &str) -> Vec<Bytes> {
        if data.trim() == "[DONE]" {
            if self.finish_reason.is_some() {
                return self.finalize();
            }
            self.terminal = true;
            return vec![sse_frame(
                "error",
                error_envelope(
                    "api_error",
                    "upstream stream ended without a stop reason",
                    &self.request_id,
                ),
            )];
        }
        let value: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(_) => {
                self.terminal = true;
                return vec![sse_frame(
                    "error",
                    error_envelope(
                        "api_error",
                        "upstream sent invalid SSE JSON",
                        &self.request_id,
                    ),
                )];
            }
        };
        if value.get("error").is_some() {
            self.terminal = true;
            return vec![sse_frame(
                "error",
                error_envelope("api_error", "upstream stream error", &self.request_id),
            )];
        }
        let mut frames = Vec::new();
        if !self.started {
            self.started = true;
            let id = value
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or(&self.request_id);
            let model = value
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or(&self.fallback_model);
            frames.push(sse_frame(
                "message_start",
                json!({"type":"message_start", "message":{
                    "id":id, "type":"message", "role":"assistant", "content":[],
                    "model":model, "stop_reason":null, "stop_sequence":null,
                    "usage":{"input_tokens":0,"output_tokens":0,
                        "cache_creation_input_tokens":0,"cache_read_input_tokens":0}
                }}),
            ));
        }
        if let Some(usage) = value.get("usage") {
            self.usage = usage.clone();
        }
        if let Some(choice) = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        {
            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                self.finish_reason = Some(reason.to_owned());
            }
            if let Some(sequence) = choice.get("stop_sequence") {
                self.stop_sequence = sequence.clone();
            }
            if let Some(delta) = choice.get("delta").and_then(Value::as_object) {
                if let Some(reasoning) = reasoning_delta(delta).filter(|text| !text.is_empty()) {
                    // Count reasoning even when unexposed: the model still
                    // produced (and billed) it, and progress telemetry should
                    // show a stream that is thinking rather than hung.
                    self.reasoning_bytes =
                        self.reasoning_bytes.saturating_add(reasoning.len() as u64);
                    self.reasoning_events = self.reasoning_events.saturating_add(1);
                    Self::mark_first(&mut self.first_reasoning, self.request_started);
                    if !self.expose_thinking {
                        // Shadow accumulation: the client never sees this
                        // text; the shadow store may hand it to the next
                        // request in the same reasoning epoch.
                        self.shadow_reasoning.push_str(reasoning);
                    }
                    if self.expose_thinking {
                        let index = self.ensure_thinking(&mut frames);
                        frames.push(sse_frame(
                            "content_block_delta",
                            json!({"type":"content_block_delta", "index":index,
                                "delta":{"type":"thinking_delta", "thinking":reasoning}}),
                        ));
                    }
                }
                if let Some(text) = delta
                    .get("content")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                {
                    self.text_bytes = self.text_bytes.saturating_add(text.len() as u64);
                    self.text_events = self.text_events.saturating_add(1);
                    Self::mark_first(&mut self.first_text, self.request_started);
                    let index = self.ensure_text(&mut frames);
                    frames.push(sse_frame(
                        "content_block_delta",
                        json!({"type":"content_block_delta", "index":index,
                            "delta":{"type":"text_delta", "text":text}}),
                    ));
                }
                if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                    for call in calls {
                        self.tool_call_events = self.tool_call_events.saturating_add(1);
                        if let Some(arguments) = call
                            .get("function")
                            .and_then(|function| function.get("arguments"))
                            .and_then(Value::as_str)
                        {
                            self.tool_call_bytes =
                                self.tool_call_bytes.saturating_add(arguments.len() as u64);
                        }
                        if self.first_tool_call.is_none()
                            && call
                                .get("function")
                                .and_then(|function| function.get("name"))
                                .and_then(Value::as_str)
                                .is_some_and(|name| !name.is_empty())
                        {
                            self.first_tool_call = Some(self.request_started.elapsed().as_millis());
                        }
                        self.handle_tool(call, &mut frames);
                    }
                }
            }
        }
        frames
    }

    fn ensure_text(&mut self, frames: &mut Vec<Bytes>) -> usize {
        if let Some(index) = self.text_index {
            return index;
        }
        let index = self.blocks.len();
        self.blocks.push(BlockKind::Text);
        self.text_index = Some(index);
        frames.push(sse_frame(
            "content_block_start",
            json!({"type":"content_block_start", "index":index,
                "content_block":{"type":"text", "text":""}}),
        ));
        index
    }

    fn ensure_thinking(&mut self, frames: &mut Vec<Bytes>) -> usize {
        if let Some(index) = self.thinking_index {
            return index;
        }
        let index = self.blocks.len();
        self.blocks.push(BlockKind::Thinking);
        self.thinking_index = Some(index);
        frames.push(sse_frame(
            "content_block_start",
            json!({"type":"content_block_start", "index":index,
                "content_block":{"type":"thinking", "thinking":""}}),
        ));
        index
    }

    fn handle_tool(&mut self, call: &Value, frames: &mut Vec<Bytes>) {
        let upstream_index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
        let function = call.get("function").and_then(Value::as_object);
        let tool = self.tools.entry(upstream_index).or_default();
        if let Some(id) = call
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        {
            tool.id = Some(id.to_owned());
        }
        if let Some(name) = function
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
        {
            tool.name = Some(name.to_owned());
        }
        let arguments = function
            .and_then(|function| function.get("arguments"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if tool.block_index.is_none() && (tool.name.is_some() || !arguments.is_empty()) {
            let index = self.blocks.len();
            self.blocks.push(BlockKind::Tool);
            tool.block_index = Some(index);
            let id = tool
                .id
                .clone()
                .unwrap_or_else(|| format!("toolu_{upstream_index}"));
            let name = tool.name.clone().unwrap_or_else(|| "unknown".into());
            frames.push(sse_frame(
                "content_block_start",
                json!({"type":"content_block_start", "index":index,
                    "content_block":{"type":"tool_use", "id":id, "name":name, "input":{}}}),
            ));
            if !tool.pending_arguments.is_empty() {
                frames.push(sse_frame(
                    "content_block_delta",
                    json!({"type":"content_block_delta", "index":index,
                        "delta":{"type":"input_json_delta", "partial_json":std::mem::take(&mut tool.pending_arguments)}}),
                ));
            }
        }
        if arguments.is_empty() {
            return;
        }
        if let Some(index) = tool.block_index {
            frames.push(sse_frame(
                "content_block_delta",
                json!({"type":"content_block_delta", "index":index,
                    "delta":{"type":"input_json_delta", "partial_json":arguments}}),
            ));
        } else {
            tool.pending_arguments.push_str(arguments);
        }
    }

    fn finalize(&mut self) -> Vec<Bytes> {
        if self.terminal {
            return Vec::new();
        }
        let mut frames = Vec::new();
        let pending = self
            .tools
            .iter()
            .filter(|(_, tool)| tool.block_index.is_none())
            .map(|(index, _)| *index)
            .collect::<Vec<_>>();
        for upstream_index in pending {
            if let Some(tool) = self.tools.get_mut(&upstream_index) {
                let index = self.blocks.len();
                self.blocks.push(BlockKind::Tool);
                tool.block_index = Some(index);
                let id = tool
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("toolu_{upstream_index}"));
                let name = tool.name.clone().unwrap_or_else(|| "unknown".into());
                frames.push(sse_frame(
                    "content_block_start",
                    json!({"type":"content_block_start", "index":index,
                        "content_block":{"type":"tool_use", "id":id, "name":name, "input":{}}}),
                ));
                if !tool.pending_arguments.is_empty() {
                    frames.push(sse_frame(
                        "content_block_delta",
                        json!({"type":"content_block_delta", "index":index,
                            "delta":{"type":"input_json_delta",
                                "partial_json":std::mem::take(&mut tool.pending_arguments)}}),
                    ));
                }
            }
        }
        self.terminal = true;
        for (index, block) in self.blocks.iter().enumerate() {
            if matches!(block, BlockKind::Thinking) {
                frames.push(sse_frame(
                    "content_block_delta",
                    json!({"type":"content_block_delta", "index":index,
                        "delta":{"type":"signature_delta",
                            "signature":thinking_signature("streamed", &self.request_id)}}),
                ));
            }
            frames.push(sse_frame(
                "content_block_stop",
                json!({"type":"content_block_stop", "index":index}),
            ));
        }
        frames.push(sse_frame(
            "message_delta",
            json!({"type":"message_delta", "delta":{
                "stop_reason":map_stop_reason(self.finish_reason.as_deref()),
                "stop_sequence":self.stop_sequence
            }, "usage":convert_usage(Some(&self.usage))}),
        ));
        frames.push(sse_frame("message_stop", json!({"type":"message_stop"})));
        frames
    }
}

/// Extract reasoning text from an upstream (non-stream) message object.
/// Shared with the server's reasoning shadow store; returns "" when absent.
pub fn reasoning_text_from_message(message: &Map<String, Value>) -> String {
    for name in ["reasoning_content", "reasoning"] {
        if let Some(text) = message.get(name).and_then(Value::as_str) {
            return text.to_owned();
        }
    }
    let Some(details) = message.get("reasoning_details") else {
        return String::new();
    };
    match details {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                item.as_str().or_else(|| {
                    item.get("text")
                        .or_else(|| item.get("content"))
                        .and_then(Value::as_str)
                })
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

impl StreamState {
    /// Reasoning shadow commit (issue #10): called once when the stream
    /// ends. Responses with tool calls store their (unexposed) reasoning;
    /// final answers clear the session's shadow state.
    fn commit_shadow(&mut self, shadow: Option<&StreamShadowContext>) {
        let Some(context) = shadow else {
            return;
        };
        if self.tools.is_empty() {
            context.store.clear_session(&context.session_fingerprint);
            return;
        }
        if self.shadow_reasoning.is_empty() {
            return;
        }
        let ids: Vec<String> = self
            .tools
            .values()
            .filter_map(|tool| tool.id.clone())
            .collect();
        if !ids.is_empty() {
            context
                .store
                .store(&context.session_fingerprint, &ids, &self.shadow_reasoning);
        }
        self.shadow_reasoning.clear();
    }
}

fn reasoning_delta(object: &Map<String, Value>) -> Option<&str> {
    object
        .get("reasoning_content")
        .or_else(|| object.get("reasoning"))
        .and_then(Value::as_str)
        .or_else(|| {
            object
                .get("reasoning_details")
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .and_then(|item| item.get("text").or_else(|| item.get("content")))
                .and_then(Value::as_str)
        })
}

fn sse_frame(event: &str, data: Value) -> Bytes {
    Bytes::from(format!("event: {event}\ndata: {data}\n\n"))
}

pub async fn parse_json_response(response: reqwest::Response) -> Result<Value, ProtocolError> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_UPSTREAM_JSON_BYTES as u64)
    {
        return Err(ProtocolError::upstream(
            "upstream JSON response was too large",
        ));
    }
    let bytes = response
        .bytes()
        .await
        .map_err(|_| ProtocolError::upstream("failed to read upstream response"))?;
    if bytes.len() > MAX_UPSTREAM_JSON_BYTES {
        return Err(ProtocolError::upstream(
            "upstream JSON response was too large",
        ));
    }
    serde_json::from_slice(&bytes)
        .map_err(|_| ProtocolError::upstream("upstream returned invalid JSON"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Glm53Config;
    use crate::optimize::optimize_request;
    use std::time::Instant;

    fn test_glm_config() -> Glm53Config {
        Glm53Config::default()
    }

    /// Convert + apply the GLM policy exactly as the server does, including
    /// alias resolution (claude-* ids map to the GLM upstream model).
    fn convert_and_optimize(
        bytes: &[u8],
    ) -> Result<(ConvertedRequest, crate::optimize::RequestOptimization), ProtocolError> {
        let mut converted = convert_request(bytes)?;
        // Mirror the server: the alias "claude-sonnet-4-6" -> GLM upstream
        // model is applied before the policy step.
        if converted.model == "claude-sonnet-4-6" {
            converted.body["model"] = Value::String("z-ai/glm-5.3-flash".into());
        }
        let optimization = optimize_request(
            &mut converted.body,
            &test_glm_config(),
            crate::optimize::Origin::Anthropic {
                thinking: converted.thinking.as_ref(),
                output_effort: converted.output_effort.as_deref(),
            },
        )
        .map_err(ProtocolError::invalid)?;
        converted.expose_thinking = optimization.expose_thinking;
        Ok((converted, optimization))
    }

    #[test]
    fn request_converts_multimodal_tools_and_tool_result() {
        let (converted, optimization) = convert_and_optimize(
            serde_json::to_vec(&json!({
                "model":"claude-sonnet-4-6", "max_tokens":256,
                "system":[{"type":"text","text":"system","cache_control":{"type":"ephemeral"}}],
                "messages":[
                    {"role":"user","content":[
                        {"type":"text","text":"look"},
                        {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}
                    ]},
                    {"role":"assistant","content":[
                        {"type":"thinking","thinking":"reason","signature":"opaque"},
                        {"type":"tool_use","id":"toolu_1","name":"Read","input":{"file_path":"/tmp/a"}}
                    ]},
                    {"role":"user","content":[
                        {"type":"tool_result","tool_use_id":"toolu_1","content":"ok"}
                    ]}
                ],
                "tools":[{"name":"Read","description":"read file","input_schema":{"type":"object"}}],
                "tool_choice":{"type":"any","disable_parallel_tool_use":false},
                "thinking":{"type":"enabled","budget_tokens":64},
                "future_advisory_field":{"safe_to_ignore":true}
            }))
            .unwrap()
            .as_slice(),
        )
        .unwrap();
        assert_eq!(
            converted.body["messages"][2]["tool_calls"][0]["function"]["name"],
            "Read"
        );
        assert_eq!(converted.body["messages"][3]["role"], "tool");
        assert_eq!(converted.body["parallel_tool_calls"], true);
        // Small thinking budget -> low effort, explicitly on the wire.
        assert_eq!(converted.body["reasoning_effort"], "low");
        assert_eq!(optimization.reasoning_effort, "low");
        // thinking blocks never cross to the OpenAI wire as blocks; the
        // historical reasoning_content is stripped before the last user turn.
        assert!(converted.body["messages"][1]
            .get("reasoning_content")
            .is_none());
        assert!(converted.body.get("thinking").is_none());
        assert!(converted.body.get("metadata").is_none());
    }

    #[test]
    fn request_without_thinking_defaults_to_explicit_high() {
        let (converted, optimization) = convert_and_optimize(
            serde_json::to_vec(&json!({
                "model":"claude-sonnet-4-6", "max_tokens":64_000,
                "messages":[{"role":"user","content":"hello"}]
            }))
            .unwrap()
            .as_slice(),
        )
        .unwrap();
        // The critical unset->max regression: the effort must be explicit.
        assert_eq!(converted.body["reasoning_effort"], "high");
        assert_eq!(converted.body["max_tokens"], 16_384);
        assert_eq!(optimization.client_max_tokens, Some(64_000));
        assert_eq!(optimization.effective_max_tokens, Some(16_384));
        assert!(!converted.expose_thinking);
    }

    #[test]
    fn requested_thinking_is_exposed_and_unrequested_reasoning_is_not() {
        let requested = convert_and_optimize(
            serde_json::to_vec(&json!({
                "model":"z-ai/glm-5.3-flash", "max_tokens":4_096,
                "thinking":{"type":"adaptive"},
                "messages":[{"role":"user","content":"hi"}]
            }))
            .unwrap()
            .as_slice(),
        )
        .unwrap()
        .0;
        assert!(requested.expose_thinking);
        assert_eq!(requested.body["reasoning_effort"], "high");

        let unrequested = convert_and_optimize(
            serde_json::to_vec(&json!({
                "model":"z-ai/glm-5.3-flash", "max_tokens":4_096,
                "messages":[{"role":"user","content":"hi"}]
            }))
            .unwrap()
            .as_slice(),
        )
        .unwrap()
        .0;
        assert!(!unrequested.expose_thinking);

        let disabled = convert_and_optimize(
            serde_json::to_vec(&json!({
                "model":"z-ai/glm-5.3-flash", "max_tokens":4_096,
                "thinking":{"type":"disabled"},
                "messages":[{"role":"user","content":"hi"}]
            }))
            .unwrap()
            .as_slice(),
        )
        .unwrap()
        .0;
        assert!(!disabled.expose_thinking);
        assert_eq!(disabled.body["reasoning_effort"], "low");
    }

    #[test]
    fn invalid_thinking_budgets_are_rejected_by_the_policy_step() {
        let result = convert_and_optimize(
            serde_json::to_vec(&json!({
                "model":"z-ai/glm-5.3-flash", "max_tokens":256,
                "thinking":{"type":"enabled","budget_tokens":4_096},
                "messages":[{"role":"user","content":"hi"}]
            }))
            .unwrap()
            .as_slice(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn nonstream_response_exposure_gates_thinking_blocks() {
        let upstream = json!({
            "id":"chat_1","model":"upstream-model",
            "choices":[{"message":{
                "content":"answer","reasoning":"think","tool_calls":[
                    {"id":"call_1","function":{"name":"Read","arguments":"{\"file_path\":\"/tmp/é\"}"}},
                    {"id":"call_2","function":{"name":"Write","arguments":"{}"}}
                ]
            },"finish_reason":"tool_calls"}],
            "usage":{"prompt_tokens":12,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":5}}
        });
        let exposed = convert_response(&upstream, "req_1", "fallback", true).unwrap();
        assert_eq!(exposed["content"][0]["type"], "thinking");
        assert_eq!(exposed["content"][2]["input"]["file_path"], "/tmp/é");
        assert_eq!(exposed["content"][3]["name"], "Write");
        assert_eq!(exposed["stop_reason"], "tool_use");
        assert_eq!(exposed["usage"]["input_tokens"], 7);
        assert_eq!(exposed["usage"]["output_tokens"], 3);

        let unexposed = convert_response(&upstream, "req_1", "fallback", false).unwrap();
        assert!(unexposed["content"]
            .as_array()
            .unwrap()
            .iter()
            .all(|block| block["type"] != "thinking"));
        assert_eq!(unexposed["content"][0]["type"], "text");
        assert_eq!(unexposed["content"][1]["type"], "tool_use");
    }

    #[test]
    fn malformed_tool_arguments_return_an_error_without_panicking() {
        let result = convert_response(
            &json!({"choices":[{"message":{"tool_calls":[
                {"id":"x","function":{"name":"Read","arguments":"{"}}
            ]},"finish_reason":"tool_calls"}]}),
            "req",
            "model",
            true,
        );
        assert!(result.is_err());
    }

    #[test]
    fn decoder_handles_arbitrary_splits_multiline_comments_and_crlf() {
        let mut decoder = SseDecoder::default();
        assert!(decoder
            .push(b": keepalive\r\nevent: x\r\ndata: {\"a\":")
            .unwrap()
            .is_empty());
        let events = decoder.push(b"1}\r\ndata: tail\r\n\r\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("x"));
        assert_eq!(events[0].data, "{\"a\":1}\ntail");
    }

    #[test]
    fn stream_lifecycle_handles_fragmented_parallel_tool_arguments() {
        let mut state = StreamState::new("req".into(), "model".into(), true, Instant::now());
        let mut frames = Vec::new();
        frames.extend(
            state.handle(r#"{"id":"chat","model":"m","choices":[{"delta":{"reasoning":"h"}}]}"#),
        );
        frames.extend(state.handle(r#"{"choices":[{"delta":{"content":"hi","tool_calls":[{"index":1,"id":"call_b","function":{"name":"B","arguments":"{\"b\":"}},{"index":0,"id":"call_a","function":{"name":"A","arguments":"{\"a\":"}}]}}]}"#));
        frames.extend(state.handle(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}},{"index":1,"function":{"arguments":"2}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":4}}"#));
        frames.extend(state.handle("[DONE]"));
        let text = frames
            .into_iter()
            .map(|frame| String::from_utf8(frame.to_vec()).unwrap())
            .collect::<String>();
        assert!(text.starts_with("event: message_start"));
        assert_eq!(text.matches("\"type\":\"tool_use\"").count(), 2);
        assert!(text.contains("{\\\"a\\\":"));
        assert!(text.contains("1}"));
        assert!(text.contains("thinking_delta"));
        assert!(text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
        assert_eq!(state.reasoning_bytes, 1);
        assert_eq!(state.text_bytes, 2);
        assert!(state.first_tool_call.is_some());
    }

    #[test]
    fn unrequested_reasoning_is_counted_but_never_emitted() {
        let mut state = StreamState::new("req".into(), "model".into(), false, Instant::now());
        let frames = state.handle(
            r#"{"id":"chat","model":"m","choices":[{"delta":{"reasoning":"secret thinking"}}]}"#,
        );
        // message_start is emitted; no thinking block frames are.
        let text = frames
            .iter()
            .map(|frame| String::from_utf8(frame.to_vec()).unwrap())
            .collect::<String>();
        assert!(text.contains("message_start"));
        assert!(!text.contains("thinking"));
        assert_eq!(state.reasoning_bytes, "secret thinking".len() as u64);
        assert!(state.thinking_index.is_none());

        // And a later text delta still produces a correct (index 0) text block.
        let frames = state.handle(r#"{"choices":[{"delta":{"content":"ok"}}]}"#);
        assert_eq!(frames.len(), 2); // content_block_start + text_delta
        let text = frames
            .iter()
            .map(|frame| String::from_utf8(frame.to_vec()).unwrap())
            .collect::<String>();
        assert!(text.contains("\"index\":0"));
        assert!(text.contains("text_delta"));
    }

    #[test]
    fn malformed_and_truncated_streams_are_terminal_errors() {
        let mut malformed = StreamState::new("req".into(), "model".into(), true, Instant::now());
        let text = String::from_utf8(malformed.handle("not json")[0].to_vec()).unwrap();
        assert!(text.starts_with("event: error"));
        assert!(malformed.terminal);

        let mut truncated = StreamState::new("req".into(), "model".into(), true, Instant::now());
        let frames = truncated.handle(r#"{"choices":[{"delta":{"content":"partial"}}]}"#);
        assert!(!frames.is_empty());
        assert!(truncated.finish_reason.is_none());
    }
}
