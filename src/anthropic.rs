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
    if let Some(thinking) = object.get("thinking") {
        if let Some(effort) = convert_thinking(thinking, max_tokens)? {
            output.insert("reasoning_effort".into(), Value::String(effort.into()));
        }
    }
    if let Some(config) = object.get("output_config") {
        convert_output_config(config, &mut output)?;
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

fn convert_thinking(value: &Value, max_tokens: u64) -> Result<Option<&'static str>, ProtocolError> {
    let object = value
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("thinking must be an object"))?;
    match required_string(object, "type")? {
        "disabled" => Ok(Some(
            crate::glm53::reasoning::GlmReasoningEffort::Low.as_str(),
        )),
        "adaptive" => Ok(Some(
            crate::glm53::reasoning::GlmReasoningEffort::High.as_str(),
        )),
        "enabled" => {
            let budget = object
                .get("budget_tokens")
                .and_then(Value::as_u64)
                .filter(|value| *value > 0)
                .ok_or_else(|| ProtocolError::invalid("thinking budget_tokens must be positive"))?;
            if budget >= max_tokens {
                return Err(ProtocolError::invalid(
                    "thinking budget_tokens must be less than max_tokens",
                ));
            }
            // GLM-5.3-Flash has no `medium`; the official template coerced
            // it to `max`. See src/glm53/reasoning.rs and docs/GLM53_FLASH.md.
            let effort = if budget < crate::glm53::reasoning::LOW_BUDGET_LIMIT {
                crate::glm53::reasoning::GlmReasoningEffort::Low
            } else {
                crate::glm53::reasoning::GlmReasoningEffort::High
            };
            Ok(Some(effort.as_str()))
        }
        kind => Err(ProtocolError::invalid(format!(
            "unsupported thinking type {kind:?}"
        ))),
    }
}

fn convert_output_config(
    value: &Value,
    output: &mut Map<String, Value>,
) -> Result<(), ProtocolError> {
    let object = value
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("output_config must be an object"))?;
    if let Some(effort) = object.get("effort") {
        let effort = effort
            .as_str()
            .ok_or_else(|| ProtocolError::invalid("output_config.effort must be a string"))?;
        let effort = match effort {
            "low" => "low",
            "medium" | "high" | "xhigh" => "high",
            "max" => "max",
            _ => return Err(ProtocolError::invalid("unsupported output_config.effort")),
        };
        output.insert("reasoning_effort".into(), Value::String(effort.into()));
    }
    if let Some(format) = object.get("format") {
        output.insert("response_format".into(), convert_output_format(format)?);
    }
    Ok(())
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
    let reasoning = reasoning_text(message);
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

pub fn stream_body(
    response: reqwest::Response,
    request_id: String,
    fallback_model: String,
    key_name: String,
    request_started: Instant,
    progress_secs: u64,
) -> Body {
    let output = async_stream::stream! {
        let mut upstream = response.bytes_stream();
        let idle = tokio::time::sleep(STREAM_PING_INTERVAL);
        tokio::pin!(idle);
        let progress_interval = Duration::from_secs(progress_secs.max(1));
        let progress = tokio::time::sleep(progress_interval);
        tokio::pin!(progress);
        let mut decoder = SseDecoder::default();
        let mut state = StreamState::new(request_id, fallback_model);
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
                telemetry.finish("complete");
            } else {
                let frame = sse_frame("error", error_envelope(
                    "api_error", "upstream stream ended unexpectedly", &state.request_id,
                ));
                telemetry.commit(frame.len());
                yield Ok(frame);
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
            "Anthropic stream still active"
        );
    }

    fn finish(&mut self, outcome: &'static str) {
        if self.finished {
            return;
        }
        self.finished = true;
        tracing::info!(
            request_id = %self.request_id,
            selected_key_name = %self.key_name,
            outcome,
            duration_ms = self.started.elapsed().as_millis(),
            upstream_chunks = self.upstream_chunks,
            upstream_events = self.upstream_events,
            downstream_frames = self.downstream_frames,
            committed_to_client = self.committed_to_client,
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
    blocks: Vec<BlockKind>,
    text_index: Option<usize>,
    thinking_index: Option<usize>,
    tools: HashMap<u64, ToolStream>,
    finish_reason: Option<String>,
    stop_sequence: Value,
    usage: Value,
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
    fn new(request_id: String, fallback_model: String) -> Self {
        Self {
            request_id,
            fallback_model,
            started: false,
            terminal: false,
            blocks: Vec::new(),
            text_index: None,
            thinking_index: None,
            tools: HashMap::new(),
            finish_reason: None,
            stop_sequence: Value::Null,
            usage: json!({}),
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
                    let index = self.ensure_thinking(&mut frames);
                    frames.push(sse_frame(
                        "content_block_delta",
                        json!({"type":"content_block_delta", "index":index,
                            "delta":{"type":"thinking_delta", "thinking":reasoning}}),
                    ));
                }
                if let Some(text) = delta
                    .get("content")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                {
                    let index = self.ensure_text(&mut frames);
                    frames.push(sse_frame(
                        "content_block_delta",
                        json!({"type":"content_block_delta", "index":index,
                            "delta":{"type":"text_delta", "text":text}}),
                    ));
                }
                if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                    for call in calls {
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

    #[test]
    fn request_converts_multimodal_tools_and_tool_result() {
        let converted = convert_request(
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
        assert_eq!(converted.body["reasoning_effort"], "low");
    }

    #[test]
    fn nonstream_response_converts_reasoning_usage_and_parallel_tools() {
        let converted = convert_response(
            &json!({
                "id":"chat_1","model":"upstream-model",
                "choices":[{"message":{
                    "content":"answer","reasoning":"think","tool_calls":[
                        {"id":"call_1","function":{"name":"Read","arguments":"{\"file_path\":\"/tmp/é\"}"}},
                        {"id":"call_2","function":{"name":"Write","arguments":"{}"}}
                    ]
                },"finish_reason":"tool_calls"}],
                "usage":{"prompt_tokens":12,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":5}}
            }),
            "req_1",
            "fallback",
        )
        .unwrap();
        assert_eq!(converted["content"][0]["type"], "thinking");
        assert_eq!(converted["content"][2]["input"]["file_path"], "/tmp/é");
        assert_eq!(converted["content"][3]["name"], "Write");
        assert_eq!(converted["stop_reason"], "tool_use");
        assert_eq!(converted["usage"]["input_tokens"], 7);
        assert_eq!(converted["usage"]["output_tokens"], 3);
    }

    #[test]
    fn malformed_tool_arguments_return_an_error_without_panicking() {
        let result = convert_response(
            &json!({"choices":[{"message":{"tool_calls":[
                {"id":"x","function":{"name":"Read","arguments":"{"}}
            ]},"finish_reason":"tool_calls"}]}),
            "req",
            "model",
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
        let mut state = StreamState::new("req".into(), "model".into());
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
    }

    #[test]
    fn malformed_and_truncated_streams_are_terminal_errors() {
        let mut malformed = StreamState::new("req".into(), "model".into());
        let text = String::from_utf8(malformed.handle("not json")[0].to_vec()).unwrap();
        assert!(text.starts_with("event: error"));
        assert!(malformed.terminal);

        let mut truncated = StreamState::new("req".into(), "model".into());
        let frames = truncated.handle(r#"{"choices":[{"delta":{"content":"partial"}}]}"#);
        assert!(!frames.is_empty());
        assert!(truncated.finish_reason.is_none());
    }
}
