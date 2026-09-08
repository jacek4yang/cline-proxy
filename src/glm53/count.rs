//! Exact `/v1/messages/count_tokens` pipeline:
//! Anthropic request -> GLM messages -> official template -> official tokenizer.

use serde_json::Value;

use crate::glm53::messages::convert_anthropic_to_glm;
use crate::glm53::{template, tokenizer, CountError};

/// Count the exact prompt tokens for an Anthropic Messages request, matching
/// the official GLM-5.3-Flash tokenizer and chat template byte-for-byte.
pub fn count_input_tokens(request: &Value) -> Result<u32, CountError> {
    let conversion = convert_anthropic_to_glm(request)?;
    let tools = conversion.tools.map(Value::Array);
    let messages = Value::Array(conversion.messages);
    let rendered = template::render(&template::TemplateInput {
        messages: &messages,
        tools: tools.as_ref(),
        reasoning_effort: conversion.reasoning_effort,
        clear_thinking: false,
        add_generation_prompt: true,
    })?;
    let ids = tokenizer::encode_no_special(&rendered)?;
    u32::try_from(ids.len()).map_err(|_| CountError {
        error_type: "api_error",
        message: "token count overflowed".into(),
    })
}
