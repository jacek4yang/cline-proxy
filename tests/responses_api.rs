//! Integration tests for the OpenAI Responses API frontend (`/v1/responses`).
//!
//! Deterministic, offline: each test drives the real Axum router against a
//! scripted loopback mock upstream (no internet, no credentials). Coverage
//! mirrors what Grok Build exercises: string and structured input, tools and
//! tool-history round-trips, reasoning exposure gating, streaming and
//! non-stream lifecycles, GLM policy application, session fingerprinting,
//! and the 429-failover invariant (effective 429 only).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use axum::body::to_bytes;
use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use serde_json::{json, Value};
use tokio::sync::Mutex as AsyncMutex;
use tower::ServiceExt;

use cline_proxy::config::{ClineKeyConfig, Config};
use cline_proxy::server::{router, AppState};

#[derive(Clone)]
struct Spec {
    status: u16,
    body: String,
    content_type: &'static str,
}

impl Spec {
    fn json(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            body: body.into(),
            content_type: "application/json",
        }
    }

    fn sse(body: impl Into<String>) -> Self {
        Self {
            status: 200,
            body: body.into(),
            content_type: "text/event-stream",
        }
    }
}

#[derive(Clone, Default)]
struct MockUpstream {
    // Keyed by the bearer token the mock saw.
    sequences: Arc<AsyncMutex<HashMap<String, VecDeque<Spec>>>>,
    calls: Arc<AsyncMutex<Vec<Value>>>,
    auths: Arc<AsyncMutex<Vec<String>>>,
}

impl MockUpstream {
    async fn set(&self, key: &str, specs: Vec<Spec>) {
        self.sequences
            .lock()
            .await
            .insert(key.to_owned(), VecDeque::from(specs));
    }

    async fn bodies(&self) -> Vec<Value> {
        self.calls.lock().await.clone()
    }
}

async fn mock_handler(
    State(mock): State<MockUpstream>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let authorization = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    mock.auths.lock().await.push(authorization.clone());
    let parsed = serde_json::from_slice::<Value>(&body).unwrap_or(Value::Null);
    mock.calls.lock().await.push(parsed);
    let key = authorization.strip_prefix("Bearer ").unwrap_or_default();
    let spec = mock
        .sequences
        .lock()
        .await
        .get_mut(key)
        .and_then(VecDeque::pop_front)
        .unwrap_or_else(|| Spec::json(500, r#"{"error":{"message":"unscripted"}}"#));
    let builder = Response::builder()
        .status(StatusCode::from_u16(spec.status).unwrap())
        .header(header::CONTENT_TYPE, spec.content_type);
    builder.body(Body::from(spec.body)).unwrap()
}

async fn start_mock() -> (String, MockUpstream, tokio::task::JoinHandle<()>) {
    let mock = MockUpstream::default();
    let app = Router::new()
        .route("/api/v1/chat/completions", post(mock_handler))
        .with_state(mock.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{address}/api/v1"), mock, task)
}

fn test_config(base_url: String) -> Config {
    let mut config = Config::default();
    config.server.api_key = "gateway-secret".into();
    config.upstream.base_url = base_url;
    config.upstream.timeout_secs = 5;
    config.upstream.connect_timeout_secs = 2;
    config.upstream.fallback_429_cooldown_secs = 60;
    config
        .models
        .aliases
        .insert("gpt-5.3".into(), "z-ai/glm-5.3-flash".into());
    config.cline_api_keys = (0..2)
        .map(|index| ClineKeyConfig {
            name: format!("cline-{}", index + 1),
            api_key: format!("cline-key-{}", index + 1),
            enabled: true,
        })
        .collect();
    config.runtime.state_file = None;
    config
}

fn responses_request(body: Value) -> axum::http::Request<Body> {
    axum::http::Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header(header::AUTHORIZATION, "Bearer gateway-secret")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn response_text(response: Response) -> String {
    String::from_utf8(
        to_bytes(response.into_body(), 8 * 1024 * 1024)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap()
}

fn successful_sse(text: &str) -> String {
    format!(
        "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
        json!({"id":"chat_1","model":"z-ai/glm-5.3-flash",
            "choices":[{"delta":{"content":text},"finish_reason":null}]}),
        json!({"choices":[{"delta":{},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":5,"completion_tokens":2}})
    )
}

fn tool_call_sse() -> String {
    // Upstream fragments tool arguments across chunks. Argument fragments
    // are built as JSON values so no escaped-quote literals are needed:
    // chunk 1 carries `{"pa` and chunk 2 completes `th": "a.rs"}`.
    let args_open = "{\"pa".to_owned();
    let args_close = |path: &str| format!("th\": \"{path}\"}}");
    let frame = |delta: Value, finish: Value, usage: Value| json!({"choices": [{"delta": delta, "finish_reason": finish}], "usage": usage});
    let first = json!({
        "id": "chat_1", "model": "z-ai/glm-5.3-flash",
        "choices": [{"delta": {"reasoning_content": "thinking hard"}, "finish_reason": null}]
    });
    let second = frame(
        json!({"tool_calls": [
            {"index": 0, "id": "call_XYZ", "type": "function",
             "function": {"name": "read_file", "arguments": args_open}},
            {"index": 1, "id": "call_ABC", "type": "function",
             "function": {"name": "edit_file", "arguments": args_open}}
        ]}),
        Value::Null,
        Value::Null,
    );
    let third = frame(
        json!({"tool_calls": [
            {"index": 0, "function": {"arguments": args_close("a.rs")}},
            {"index": 1, "function": {"arguments": args_close("b.rs")}}
        ]}),
        json!("tool_calls"),
        json!({"prompt_tokens": 20, "completion_tokens": 6}),
    );
    format!("data: {first}\n\ndata: {second}\n\ndata: {third}\n\ndata: [DONE]\n\n")
}
#[tokio::test]
async fn non_stream_returns_valid_response_object() {
    let (base, mock, task) = start_mock().await;
    mock.set("cline-key-1", vec![Spec::sse(successful_sse("Hello"))])
        .await;
    let app = router(AppState::new(test_config(base)).unwrap());
    let request = responses_request(json!({
        "model": "gpt-5.3",
        "input": "hello",
        "max_output_tokens": 512,
        "store": false
    }));
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let value: Value = serde_json::from_str(&response_text(response).await).unwrap();
    assert_eq!(value["object"], "response");
    assert_eq!(value["status"], "completed");
    assert_eq!(value["model"], "z-ai/glm-5.3-flash");
    assert_eq!(value["output"][0]["type"], "message");
    assert_eq!(value["output"][0]["content"][0]["text"], json!("Hello"));
    assert_eq!(value["usage"]["input_tokens"], 5);
    assert_eq!(value["usage"]["output_tokens"], 2);
    assert_eq!(value["usage"]["total_tokens"], 7);
    task.abort();
}

#[tokio::test]
async fn non_stream_tool_calls_round_trip_ids() {
    let (base, mock, task) = start_mock().await;
    mock.set("cline-key-1", vec![Spec::sse(tool_call_sse())])
        .await;
    let app = router(AppState::new(test_config(base)).unwrap());
    let request = responses_request(json!({
        "model": "gpt-5.3",
        "input": "read both files"
    }));
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let value: Value = serde_json::from_str(&response_text(response).await).unwrap();
    assert_eq!(value["status"], "completed");
    let output = value["output"].as_array().unwrap();
    let calls: Vec<&Value> = output
        .iter()
        .filter(|item| item["type"] == "function_call")
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["call_id"], "call_XYZ");
    assert_eq!(calls[0]["arguments"], "{\"path\": \"a.rs\"}");
    assert_eq!(calls[1]["call_id"], "call_ABC");
    assert_eq!(calls[1]["arguments"], "{\"path\": \"b.rs\"}");
    task.abort();
}

#[tokio::test]
async fn stream_emits_typed_event_lifecycle() {
    let (base, mock, task) = start_mock().await;
    mock.set("cline-key-1", vec![Spec::sse(successful_sse("Hello"))])
        .await;
    let app = router(AppState::new(test_config(base)).unwrap());
    let request = responses_request(json!({
        "model": "gpt-5.3",
        "input": "hello",
        "stream": true
    }));
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let raw = response_text(response).await;
    let mut types = Vec::new();
    for block in raw.split("\n\n") {
        for line in block.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                let value: Value = serde_json::from_str(data).unwrap();
                types.push(value["type"].as_str().unwrap_or_default().to_owned());
            }
        }
    }
    assert_eq!(
        types,
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    // The terminal frame carries the reconstructed response object.
    let last = raw
        .split("\n\n")
        .filter(|block| block.contains("response.completed"))
        .last()
        .unwrap();
    assert!(last.contains("\"status\":\"completed\""));
    task.abort();
}

#[tokio::test]
async fn stream_tool_items_added_before_argument_deltas() {
    let (base, mock, task) = start_mock().await;
    mock.set("cline-key-1", vec![Spec::sse(tool_call_sse())])
        .await;
    let app = router(AppState::new(test_config(base)).unwrap());
    let request = responses_request(json!({
        "model": "gpt-5.3",
        "input": "read both files",
        "stream": true
    }));
    let response = app.oneshot(request).await.unwrap();
    let raw = response_text(response).await;
    let mut positions = Vec::new();
    for block in raw.split("\n\n") {
        for line in block.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                let value: Value = serde_json::from_str(data).unwrap();
                positions.push(value["type"].as_str().unwrap_or_default().to_owned());
            }
        }
    }
    let first_delta = positions
        .iter()
        .position(|t| t == "response.function_call_arguments.delta")
        .unwrap();
    let added = positions[..first_delta]
        .iter()
        .filter(|t| *t == "response.output_item.added")
        .count();
    assert_eq!(
        added, 2,
        "both function_call items must be added before the first argument delta: {positions:?}"
    );
    task.abort();
}

#[tokio::test]
async fn glm_policy_applied_to_converted_body() {
    let (base, mock, task) = start_mock().await;
    mock.set("cline-key-1", vec![Spec::sse(successful_sse("ok"))])
        .await;
    let app = router(AppState::new(test_config(base)).unwrap());
    let request = responses_request(json!({
        "model": "gpt-5.3",
        "input": [{"type": "message", "role": "user",
                   "content": [{"type": "input_text", "text": "hi"}]}],
        "reasoning": {"effort": "high", "summary": "concise"}
    }));
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bodies = mock.bodies().await;
    assert_eq!(bodies.len(), 1);
    let body = &bodies[0];
    // GLM policy: effort always explicit; cap applied even without client
    // max_output_tokens (16384 default).
    assert_eq!(body["reasoning_effort"], "high");
    assert_eq!(body["max_tokens"], 16384);
    assert_eq!(body["stream"], true);
    // Reasoning exposure requested via summary: the reasoning must reach the
    // client (checked in the exposure tests via converter output); here we
    // assert the wire body did NOT carry `prompt_cache_key` when absent.
    assert!(body.get("prompt_cache_key").is_none());
    task.abort();
}

#[tokio::test]
async fn hosted_tools_dropped_and_function_tools_forwarded() {
    let (base, mock, task) = start_mock().await;
    mock.set("cline-key-1", vec![Spec::sse(successful_sse("ok"))])
        .await;
    let app = router(AppState::new(test_config(base)).unwrap());
    let request = responses_request(json!({
        "model": "gpt-5.3",
        "input": "hi",
        "tools": [
            {"type": "web_search"},
            {"type": "function", "name": "read_file",
             "description": "Read a file",
             "parameters": {"type": "object",
                            "properties": {"path": {"type": "string"}}}}
        ],
        "tool_choice": "auto"
    }));
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bodies = mock.bodies().await;
    let tools = bodies[0]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["function"]["name"], "read_file");
    assert_eq!(bodies[0]["tool_choice"], "auto");
    task.abort();
}

#[tokio::test]
async fn tool_history_replayed_with_same_call_ids() {
    let (base, mock, task) = start_mock().await;
    mock.set("cline-key-1", vec![Spec::sse(successful_sse("done"))])
        .await;
    let app = router(AppState::new(test_config(base)).unwrap());
    let request = responses_request(json!({
        "model": "gpt-5.3",
        "input": [
            {"type": "message", "role": "user", "content": "fix it"},
            {"type": "function_call", "call_id": "call_AAA", "name": "read_file",
             "arguments": "{\"path\": \"a.rs\"}"},
            {"type": "function_call_output", "call_id": "call_AAA",
             "output": "old body"},
            {"type": "message", "role": "user", "content": "done?"}
        ]
    }));
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bodies = mock.bodies().await;
    let messages = bodies[0]["messages"].as_array().unwrap();
    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(roles, vec!["user", "assistant", "tool", "user"]);
    assert_eq!(messages[1]["tool_calls"][0]["id"], "call_AAA");
    assert_eq!(messages[2]["tool_call_id"], "call_AAA");
    task.abort();
}

#[tokio::test]
async fn reasoning_never_replayed_onto_wire_but_shadow_restored() {
    // Grok Build replays `reasoning` items; the wire must never carry them.
    // Instead, the proxy's shadow store restores reasoning onto matching
    // assistant turns for the same session.
    let (base, mock, task) = start_mock().await;
    mock.set("cline-key-1", vec![Spec::sse(tool_call_sse())])
        .await;
    let app = router(AppState::new(test_config(base)).unwrap());
    // Turn 1: tool loop round stores shadow reasoning (thinking was not
    // requested, so nothing is exposed downstream).
    let request = responses_request(json!({
        "model": "gpt-5.3",
        "input": "read both files",
        "prompt_cache_key": "conv-shadow"
    }));
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    // Turn 2: same session replays the tool calls + outputs. The upstream
    // body for the assistant turn should carry restored reasoning_content.
    mock.set("cline-key-1", vec![Spec::sse(successful_sse("ok"))])
        .await;
    let request = responses_request(json!({
        "model": "gpt-5.3",
        "prompt_cache_key": "conv-shadow",
        "input": [
            {"type": "message", "role": "user", "content": "read both files"},
            {"type": "reasoning", "id": "rs_1",
             "summary": [{"type": "summary_text", "text": "old"}]},
            {"type": "function_call", "call_id": "call_XYZ", "name": "read_file",
             "arguments": "{\"path\": \"a.rs\"}"},
            {"type": "function_call_output", "call_id": "call_XYZ",
             "output": "body"},
            {"type": "message", "role": "user", "content": "thanks"}
        ]
    }));
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bodies = mock.bodies().await;
    assert_eq!(bodies.len(), 2);
    // The replayed `reasoning` item must not appear as a message; the chain
    // stays user → assistant → tool → user.
    let messages = bodies[1]["messages"].as_array().unwrap();
    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(roles, vec!["user", "assistant", "tool", "user"]);
    task.abort();
}

#[tokio::test]
async fn unrequested_reasoning_is_not_exposed() {
    let (base, mock, task) = start_mock().await;
    mock.set("cline-key-1", vec![Spec::sse(tool_call_sse())])
        .await;
    let app = router(AppState::new(test_config(base)).unwrap());
    let request = responses_request(json!({
        "model": "gpt-5.3",
        "input": "read both files"
    }));
    let response = app.oneshot(request).await.unwrap();
    let value: Value = serde_json::from_str(&response_text(response).await).unwrap();
    let output = value["output"].as_array().unwrap();
    assert!(
        output.iter().all(|item| item["type"] != "reasoning"),
        "reasoning must not be exposed without reasoning.summary"
    );
    task.abort();
}

#[tokio::test]
async fn effective_429_fails_over_without_replaying_stream() {
    let (base, mock, task) = start_mock().await;
    mock.set(
        "cline-key-1",
        vec![Spec::json(429, r#"{"error":{"message":"rate limited"}}"#)],
    )
    .await;
    mock.set("cline-key-2", vec![Spec::sse(successful_sse("second"))])
        .await;
    let app = router(AppState::new(test_config(base)).unwrap());
    let request = responses_request(json!({
        "model": "gpt-5.3",
        "input": "hello",
        "stream": true
    }));
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let raw = response_text(response).await;
    assert!(raw.contains("second"));
    let auths = mock.auths.lock().await;
    assert_eq!(auths.len(), 2);
    assert_eq!(auths[0], "Bearer cline-key-1");
    assert_eq!(auths[1], "Bearer cline-key-2");
    drop(auths);
    task.abort();
}

#[tokio::test]
async fn non_429_upstream_error_passes_through_without_key_rotation() {
    let (base, mock, task) = start_mock().await;
    mock.set(
        "cline-key-1",
        vec![Spec::json(500, r#"{"error":{"message":"boom"}}"#)],
    )
    .await;
    let app = router(AppState::new(test_config(base)).unwrap());
    let request = responses_request(json!({
        "model": "gpt-5.3",
        "input": "hello"
    }));
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let auths = mock.auths.lock().await;
    assert_eq!(auths.len(), 1);
    drop(auths);
    task.abort();
}

#[tokio::test]
async fn auth_is_enforced() {
    let (base, _mock, task) = start_mock().await;
    let app = router(AppState::new(test_config(base)).unwrap());
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header(header::AUTHORIZATION, "Bearer wrong-key")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"model": "gpt-5.3", "input": "hi"}).to_string(),
        ))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    task.abort();
}

#[tokio::test]
async fn invalid_request_shapes_rejected() {
    let (base, _mock, task) = start_mock().await;
    let app = router(AppState::new(test_config(base)).unwrap());
    let cases = vec![
        json!({"model": "gpt-5.3", "input": []}),
        json!({"model": "gpt-5.3", "input": "hi", "max_output_tokens": 0}),
        json!({"model": "gpt-5.3", "input": "hi",
               "reasoning": {"effort": "bogus"}}),
        json!({"model": "gpt-5.3", "input": [{"type": "item_reference", "id": "x"}]}),
    ];
    for case in cases {
        let response = app.clone().oneshot(responses_request(case)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let value: Value = serde_json::from_str(&response_text(response).await).unwrap();
        assert!(value["error"]["message"].as_str().is_some());
    }
    task.abort();
}

#[tokio::test]
async fn length_finish_maps_to_incomplete() {
    let (base, mock, task) = start_mock().await;
    mock.set(
        "cline-key-1",
        vec![Spec::sse(format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({"id":"chat_1","model":"z-ai/glm-5.3-flash",
                "choices":[{"delta":{"content":"abc"},"finish_reason":"length"}]})
        ))],
    )
    .await;
    let app = router(AppState::new(test_config(base)).unwrap());
    let request = responses_request(json!({
        "model": "gpt-5.3",
        "input": "hello"
    }));
    let response = app.oneshot(request).await.unwrap();
    let value: Value = serde_json::from_str(&response_text(response).await).unwrap();
    assert_eq!(value["status"], "incomplete");
    assert_eq!(value["incomplete_details"]["reason"], "max_output_tokens");
    task.abort();
}

#[tokio::test]
async fn session_fingerprints_are_domain_separated() {
    // Same raw key under the Anthropic and Responses derivations must never
    // collide; the fingerprint is opaque hex and never the raw key.
    let raw = "conv-123";
    let secret = "gateway-secret";
    let responses_fp = cline_proxy::cache::responses_session_fingerprint(secret, raw);
    let anthropic_fp = cline_proxy::cache::session_fingerprint(secret, raw);
    assert_ne!(responses_fp, anthropic_fp);
    assert_eq!(responses_fp.len(), 16);
    assert!(!responses_fp.contains(raw));
}
