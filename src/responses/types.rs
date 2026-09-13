//! Responses wire types: usage mapping, SSE event formatting, IDs, and the
//! request validation error. Event field names follow the current Responses
//! schema as consumed by Grok Build (typed deserialization: unknown event
//! types and missing required fields fail the client, so shapes here are
//! exact, never approximate).

use serde_json::{json, Map, Value};

/// Conversion/validation error surfaced as a Responses 400. Same semantics as
/// the Anthropic frontend's `ProtocolError`; the server layer renders it in
/// the OpenAI error envelope.
#[derive(Debug, Clone)]
pub struct ProtocolError {
    pub message: String,
}

impl ProtocolError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// OpenAI Chat Completions usage → Responses usage tuple
/// `(input_tokens, output_tokens, cached_tokens, reasoning_tokens)`.
///
/// Unlike the Anthropic mapping, Responses `input_tokens` is the FULL prompt
/// count: `input_tokens_details.cached_tokens` is a subset (the cached slice),
/// never subtracted. Inventing a subtraction here would under-report billing
/// and break Grok Build's context accounting. Missing fields map to 0 (schema
/// requires the fields; the data is genuinely unavailable upstream).
pub fn map_usage(usage: &Value) -> (u64, u64, u64, u64) {
    let input = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached = usage
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let reasoning = usage
        .pointer("/completion_tokens_details/reasoning_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    (input, output, cached.min(input), reasoning.min(output))
}

/// Render the Responses usage object (all schema fields always present).
pub fn usage_json(usage: (u64, u64, u64, u64)) -> Value {
    let (input, output, cached, reasoning) = usage;
    json!({
        "input_tokens": input,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": input + output,
    })
}

/// Format one Responses SSE frame. Responses events are pure `data:` lines —
/// the client's typed decoder reads only `data:`; there is no `event:` name
/// in the OpenAI/xAI Responses wire format.
pub fn sse_event(data: &Value) -> String {
    format!("data: {data}\n\n")
}

/// SSE comment keepalive. Comments (`: …`) are ignored by every eventsource
/// decoder — they carry no `data:` line, so they can never fail the client's
/// typed deserialization (unlike a synthetic "ping" event, which would).
pub fn sse_ping() -> &'static str {
    ": ping\n\n"
}

/// `resp_<opaque>` response id. Random hex — no session, credential, or
/// account material is ever embedded in client-visible ids.
pub fn new_response_id() -> String {
    format!("resp_{}", uuid::Uuid::new_v4().simple())
}

/// `msg_<opaque>` message item id.
pub fn new_message_id() -> String {
    format!("msg_{}", uuid::Uuid::new_v4().simple())
}

/// `rs_<opaque>` reasoning item id.
pub fn new_reasoning_id() -> String {
    format!("rs_{}", uuid::Uuid::new_v4().simple())
}

/// `fc_<opaque>` function-call item id.
pub fn new_function_call_id() -> String {
    format!("fc_{}", uuid::Uuid::new_v4().simple())
}

/// Build the base response object carried by `response.created`,
/// `response.in_progress`, and the terminal events. `output` is the caller's
/// accumulated item list (the terminal frame must reconstruct everything that
/// streamed — Grok Build replays it as next-turn conversation state).
pub fn base_response(
    response_id: &str,
    model: &str,
    status: &str,
    output: Vec<Value>,
    usage: Option<(u64, u64, u64, u64)>,
    created_at: u64,
) -> Value {
    let mut response = Map::new();
    response.insert("id".into(), json!(response_id));
    response.insert("object".into(), json!("response"));
    response.insert("created_at".into(), json!(created_at));
    response.insert("status".into(), json!(status));
    response.insert("model".into(), json!(model));
    response.insert("output".into(), Value::Array(output));
    response.insert("usage".into(), usage_json(usage.unwrap_or((0, 0, 0, 0))));
    response.insert("error".into(), Value::Null);
    response.insert("incomplete_details".into(), Value::Null);
    response.insert("parallel_tool_calls".into(), json!(true));
    Value::Object(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_mapping_keeps_input_total_and_clamps_details() {
        let usage = json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 30},
            "completion_tokens_details": {"reasoning_tokens": 8}
        });
        assert_eq!(map_usage(&usage), (100, 20, 30, 8));
        // cached_tokens is a subset of input: never larger after clamping.
        let usage = json!({
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 99},
        });
        let (input, output, cached, reasoning) = map_usage(&usage);
        assert_eq!((input, output, cached, reasoning), (10, 5, 10, 0));
        // Missing fields map to schema-required zeros.
        assert_eq!(map_usage(&json!({})), (0, 0, 0, 0));
    }

    #[test]
    fn usage_json_has_all_schema_fields() {
        let rendered = usage_json((100, 20, 30, 8));
        assert_eq!(rendered["input_tokens"], 100);
        assert_eq!(rendered["input_tokens_details"]["cached_tokens"], 30);
        assert_eq!(rendered["output_tokens_details"]["reasoning_tokens"], 8);
        assert_eq!(rendered["total_tokens"], 120);
    }

    #[test]
    fn sse_formatting_matches_responses_wire() {
        let frame = sse_event(&json!({"type": "response.created"}));
        assert!(frame.starts_with("data: {\"type\":\"response.created\"}"));
        assert!(frame.ends_with("\n\n"));
        // Ping is an SSE comment: ignored by eventsource decoders.
        assert_eq!(sse_ping(), ": ping\n\n");
    }

    #[test]
    fn ids_are_opaque_and_prefixed() {
        assert!(new_response_id().starts_with("resp_"));
        assert!(new_message_id().starts_with("msg_"));
        assert!(new_reasoning_id().starts_with("rs_"));
        assert!(new_function_call_id().starts_with("fc_"));
    }
}
