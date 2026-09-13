//! Anthropic Messages -> GLM chat-template message representation.
//!
//! This conversion is deliberately separate from `crate::anthropic` (the
//! OpenAI wire conversion): the official GLM template expects tool-call
//! arguments as JSON *objects*, image blocks as placeholders, and assistant
//! reasoning as `reasoning_content`. Shape parity with the golden fixtures
//! (`tests/fixtures/glm53/`) is enforced by unit tests.

use serde_json::{json, Map, Value};

use crate::glm53::reasoning::{from_output_effort, from_thinking, GlmReasoningEffort};
use crate::glm53::CountError;

pub struct GlmConversion {
    pub messages: Vec<Value>,
    /// `None` when the request has no tools; the template treats an empty
    /// tool list the same as absent.
    pub tools: Option<Vec<Value>>,
    pub reasoning_effort: Option<GlmReasoningEffort>,
}

pub fn convert_anthropic_to_glm(request: &Value) -> Result<GlmConversion, CountError> {
    let object = request
        .as_object()
        .ok_or_else(|| CountError::invalid("request body must be a JSON object"))?;
    let model = object
        .get("model")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| CountError::invalid("model must be a non-empty string"))?;
    let _ = model;
    let max_tokens = object.get("max_tokens").and_then(Value::as_u64);
    let input_messages = object
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| CountError::invalid("messages must be an array"))?;

    let mut messages = Vec::with_capacity(input_messages.len() + 1);
    if let Some(system) = object.get("system") {
        messages.push(json!({
            "role": "system",
            "content": convert_display_content(system, "system")?,
        }));
    }
    for (index, message) in input_messages.iter().enumerate() {
        convert_message(message, index, &mut messages)?;
    }

    let tools = match object.get("tools") {
        Some(Value::Array(tools)) if !tools.is_empty() => {
            let mut converted = Vec::with_capacity(tools.len());
            for (index, tool) in tools.iter().enumerate() {
                if !tool.is_object() {
                    return Err(CountError::invalid(format!(
                        "tools[{index}] must be an object"
                    )));
                }
                match crate::websearch::convert_declaration(tool) {
                    Ok(Some(function)) => converted.push(function),
                    Ok(None) => converted.push(tool.clone()),
                    Err(error) => return Err(CountError::invalid(error.message)),
                }
            }
            Some(converted)
        }
        Some(Value::Array(_)) => None,
        Some(_) => return Err(CountError::invalid("tools must be an array")),
        None => None,
    };

    let mut reasoning_effort = match object.get("thinking") {
        Some(thinking) => {
            from_thinking(thinking, max_tokens.unwrap_or(u64::MAX)).map_err(CountError::invalid)?
        }
        None => None,
    };
    if let Some(effort) = object
        .get("output_config")
        .and_then(|config| config.get("effort"))
        .and_then(Value::as_str)
    {
        reasoning_effort = Some(from_output_effort(effort).map_err(CountError::invalid)?);
    }

    Ok(GlmConversion {
        messages,
        tools,
        reasoning_effort,
    })
}

fn convert_message(
    message: &Value,
    index: usize,
    output: &mut Vec<Value>,
) -> Result<(), CountError> {
    let object = message
        .as_object()
        .ok_or_else(|| CountError::invalid(format!("messages[{index}] must be an object")))?;
    let role = object
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| CountError::invalid(format!("messages[{index}].role must be a string")))?;
    let content = object
        .get("content")
        .ok_or_else(|| CountError::invalid(format!("messages[{index}] content is required")))?;
    match role {
        "system" => {
            output.push(json!({
                "role": "system",
                "content": convert_display_content(content, &format!("messages[{index}]"))?,
            }));
        }
        "user" => {
            if content.is_string() {
                output.push(json!({"role": "user", "content": content}));
                return Ok(());
            }
            let mut ordinary: Vec<Value> = Vec::new();
            let blocks = content.as_array().ok_or_else(|| {
                CountError::invalid(format!(
                    "messages[{index}] content must be a string or array"
                ))
            })?;
            for (block_index, block) in blocks.iter().enumerate() {
                let block_object = block.as_object().ok_or_else(|| {
                    CountError::invalid(format!(
                        "messages[{index}][{block_index}] must be an object"
                    ))
                })?;
                match block_object.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        ordinary.push(convert_text_block(block_object, index, block_index)?)
                    }
                    Some("image") => ordinary.push(json!({"type": "image"})),
                    Some("document") => {
                        ordinary.push(convert_document_block(block_object, index, block_index)?)
                    }
                    Some("tool_result") => {
                        flush_user(&mut ordinary, output);
                        output.push(convert_tool_result(block_object, index, block_index)?);
                    }
                    other => {
                        return Err(CountError::invalid(format!(
                            "messages[{index}][{block_index}] has unsupported block type {other:?}"
                        )))
                    }
                }
            }
            flush_user(&mut ordinary, output);
        }
        "assistant" => output.extend(convert_assistant(content, index)?),
        other => {
            return Err(CountError::invalid(format!(
                "messages[{index}].role {other:?} is not supported by the GLM template"
            )))
        }
    }
    Ok(())
}

fn flush_user(ordinary: &mut Vec<Value>, output: &mut Vec<Value>) {
    if !ordinary.is_empty() {
        output.push(json!({"role": "user", "content": std::mem::take(ordinary)}));
    }
}

fn convert_text_block(
    object: &Map<String, Value>,
    message_index: usize,
    block_index: usize,
) -> Result<Value, CountError> {
    let text = object.get("text").and_then(Value::as_str).ok_or_else(|| {
        CountError::invalid(format!(
            "messages[{message_index}][{block_index}] text block needs a string text"
        ))
    })?;
    Ok(json!({"type": "text", "text": text}))
}

/// Images enter the template as placeholders (`<|begin_of_image|><|image|>
/// <|end_of_image|>`); the base64 payload never reaches the text stream, so
/// the placeholder is all that contributes to the prompt token count.
fn convert_document_block(
    object: &Map<String, Value>,
    message_index: usize,
    block_index: usize,
) -> Result<Value, CountError> {
    let source = object
        .get("source")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            CountError::invalid(format!(
                "messages[{message_index}][{block_index}] document needs a source object"
            ))
        })?;
    match source.get("type").and_then(Value::as_str) {
        Some("text") => {
            let data = source.get("data").and_then(Value::as_str).ok_or_else(|| {
                CountError::invalid(format!(
                    "messages[{message_index}][{block_index}] text document needs string data"
                ))
            })?;
            Ok(json!({"type": "text", "text": data}))
        }
        // URL and base64 documents are rendered by the serving stack in ways
        // the text template does not represent; counting them as zero would
        // silently undercount, so they are rejected instead.
        Some(other) => Err(CountError::invalid(format!(
            "messages[{message_index}][{block_index}] document source {other:?} cannot be counted exactly; use a text document"
        ))),
        None => Err(CountError::invalid(format!(
            "messages[{message_index}][{block_index}] document source needs a type"
        ))),
    }
}

fn convert_tool_result(
    object: &Map<String, Value>,
    message_index: usize,
    block_index: usize,
) -> Result<Value, CountError> {
    let tool_use_id = object
        .get("tool_use_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            CountError::invalid(format!(
                "messages[{message_index}][{block_index}] tool_result needs a tool_use_id"
            ))
        })?;
    let content = match object.get("content") {
        None | Some(Value::Null) => Value::String(String::new()),
        Some(Value::String(text)) => Value::String(text.clone()),
        Some(Value::Array(blocks)) => {
            let mut converted = Vec::with_capacity(blocks.len());
            for (item_index, block) in blocks.iter().enumerate() {
                let block_object = block.as_object().ok_or_else(|| {
                    CountError::invalid(format!(
                        "messages[{message_index}][{block_index}] content[{item_index}] must be an object"
                    ))
                })?;
                match block_object.get("type").and_then(Value::as_str) {
                    Some("text") => converted.push(convert_text_block(
                        block_object,
                        message_index,
                        block_index,
                    )?),
                    Some("image") => converted.push(json!({"type": "image"})),
                    Some("document") => converted.push(convert_document_block(
                        block_object,
                        message_index,
                        block_index,
                    )?),
                    Some("tool_reference") => {
                        let name = block_object
                            .get("tool_name")
                            .and_then(Value::as_str)
                            .unwrap_or("?");
                        converted.push(json!({
                            "type": "text",
                            "text": format!("[tool reference: {name}]"),
                        }));
                    }
                    Some("search_result") => {
                        let title = block_object
                            .get("title")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        let url = block_object.get("url").and_then(Value::as_str).unwrap_or("");
                        let mut text = format!("{title}\n{url}");
                        if let Some(quote) = block_object.get("content").and_then(Value::as_str) {
                            text.push('\n');
                            text.push_str(quote);
                        }
                        converted.push(json!({"type": "text", "text": text}));
                    }
                    other => {
                        return Err(CountError::invalid(format!(
                            "messages[{message_index}][{block_index}] content[{item_index}] has unsupported type {other:?}"
                        )))
                    }
                }
            }
            Value::Array(converted)
        }
        Some(_) => {
            return Err(CountError::invalid(format!(
                "messages[{message_index}][{block_index}] tool_result content must be a string or array"
            )))
        }
    };
    Ok(json!({
        "role": "tool",
        "tool_call_id": tool_use_id,
        "content": content,
    }))
}

fn convert_assistant(content: &Value, index: usize) -> Result<Vec<Value>, CountError> {
    match content {
        Value::String(inline) => Ok(vec![json!({
            "role": "assistant",
            "content": inline,
        })]),
        Value::Null => Ok(vec![json!({
            "role": "assistant",
            "content": "",
        })]),
        Value::Array(blocks) => convert_assistant_blocks(blocks, index),
        _ => Err(CountError::invalid(format!(
            "messages[{index}] assistant content must be a string or array"
        ))),
    }
}

fn convert_assistant_blocks(blocks: &[Value], index: usize) -> Result<Vec<Value>, CountError> {
    #[derive(Default)]
    struct Segment {
        text: String,
        reasoning: String,
        tool_calls: Vec<Value>,
    }
    impl Segment {
        fn is_empty(&self) -> bool {
            self.text.is_empty() && self.reasoning.is_empty() && self.tool_calls.is_empty()
        }
        fn into_message(self) -> Value {
            let mut message = Map::new();
            message.insert("role".into(), Value::String("assistant".into()));
            message.insert("content".into(), Value::String(self.text));
            if !self.reasoning.is_empty() {
                message.insert("reasoning_content".into(), Value::String(self.reasoning));
            }
            if !self.tool_calls.is_empty() {
                message.insert("tool_calls".into(), Value::Array(self.tool_calls));
            }
            Value::Object(message)
        }
    }
    let mut output = Vec::new();
    let mut segment = Segment::default();
    let mut server_calls: Vec<Value> = Vec::new();
    let flush = |output: &mut Vec<Value>, segment: &mut Segment| {
        if !segment.is_empty() {
            output.push(std::mem::take(segment).into_message());
        }
    };
    for (block_index, block) in blocks.iter().enumerate() {
        let object = block.as_object().ok_or_else(|| {
            CountError::invalid(format!(
                "messages[{index}][{block_index}] must be an object"
            ))
        })?;
        match object.get("type").and_then(Value::as_str) {
            Some("text") => {
                segment
                    .text
                    .push_str(object.get("text").and_then(Value::as_str).ok_or_else(|| {
                        CountError::invalid(format!(
                            "messages[{index}][{block_index}] text block needs a string text"
                        ))
                    })?);
            }
            Some("thinking") => {
                if let Some(thinking) = object.get("thinking").and_then(Value::as_str) {
                    segment.reasoning.push_str(thinking);
                }
            }
            Some("redacted_thinking") => {}
            Some("tool_use") | Some("server_tool_use") => {
                let name = object.get("name").and_then(Value::as_str).ok_or_else(|| {
                    CountError::invalid(format!(
                        "messages[{index}][{block_index}] tool_use needs a name"
                    ))
                })?;
                let input = object.get("input").cloned().unwrap_or_else(|| json!({}));
                if !input.is_object() {
                    return Err(CountError::invalid(format!(
                        "messages[{index}][{block_index}] tool_use input must be an object"
                    )));
                }
                let call = json!({"function": {"name": name, "arguments": input}});
                if object.get("type").and_then(Value::as_str) == Some("server_tool_use") {
                    flush(&mut output, &mut segment);
                    server_calls.push(call);
                } else {
                    segment.tool_calls.push(call);
                }
            }
            Some("web_search_tool_result") => {
                let tool_use_id = object
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or("srvtoolu_unknown");
                if !server_calls.is_empty() {
                    output.push(json!({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": std::mem::take(&mut server_calls)
                    }));
                }
                output.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": crate::websearch::replayed_result_text(block),
                }));
            }
            other => {
                return Err(CountError::invalid(format!(
                "messages[{index}][{block_index}] has unsupported assistant block type {other:?}"
            )))
            }
        }
    }
    if !server_calls.is_empty() {
        output.push(json!({
            "role": "assistant",
            "content": "",
            "tool_calls": server_calls
        }));
    }
    flush(&mut output, &mut segment);
    if output.is_empty() {
        output.push(json!({"role": "assistant", "content": ""}));
    }
    Ok(output)
}

/// System and user display content: strings stay strings; block arrays keep
/// the shape the template's `visible_text` understands (text/image/document
/// already normalized by the block converters).
fn convert_display_content(content: &Value, label: &str) -> Result<Value, CountError> {
    match content {
        Value::String(text) => Ok(Value::String(text.clone())),
        Value::Array(blocks) => {
            let mut converted = Vec::with_capacity(blocks.len());
            for (index, block) in blocks.iter().enumerate() {
                let object = block.as_object().ok_or_else(|| {
                    CountError::invalid(format!("{label}[{index}] must be an object"))
                })?;
                match object.get("type").and_then(Value::as_str) {
                    Some("text") => converted.push(convert_text_block(object, 0, index)?),
                    Some("image") => converted.push(json!({"type": "image"})),
                    Some("document") => converted.push(convert_document_block(object, 0, index)?),
                    other => {
                        return Err(CountError::invalid(format!(
                            "{label}[{index}] has unsupported block type {other:?}"
                        )))
                    }
                }
            }
            Ok(Value::Array(converted))
        }
        _ => Err(CountError::invalid(format!(
            "{label} content must be a string or array"
        ))),
    }
}
