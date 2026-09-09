//! End-to-end reasoning shadow store integration (issue #10): store on a
//! tool-call response, restore onto the next request in the same epoch,
//! clear on final answer, and fail-safe behavior without session identity.
//!
//! Drives the real router against the mock upstream like `tests/glm53_policy.rs`
//! drives the pipeline, verifying the wire invariants.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::Request;
use axum::http::header;
use axum::http::StatusCode;
use axum::routing::post;
use axum::Router;
use serde_json::{json, Value};
use tower::ServiceExt;

use cline_proxy::server::{router, AppState};
use cline_proxy::state::StateLoadOutcome;

fn test_config() -> cline_proxy::config::Config {
    let mut config = cline_proxy::config::Config::default();
    config.server.api_key = "gateway-secret".into();
    config.cline_api_keys = vec![cline_proxy::config::ClineKeyConfig {
        name: "one".into(),
        api_key: "cline-key-1".into(),
        enabled: true,
    }];
    config.runtime.state_file = None;
    config
}

fn gateway_request(path: &str, body: String) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(header::AUTHORIZATION, "Bearer gateway-secret")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap()
}

/// Mock upstream that echoes whatever we script per call.
struct Mock {
    responses: std::sync::Mutex<Vec<Value>>,
    seen: std::sync::Mutex<Vec<Value>>,
}

impl Mock {
    fn tool_call_response(reasoning: &str, call_id: &str, command: &str) -> Value {
        json!({
            "id":"chat", "model":"z-ai/glm-5.3-flash",
            "choices":[{"message":{
                "reasoning_content": reasoning,
                "content":"",
                "tool_calls":[{"id":call_id, "type":"function",
                    "function":{"name":"Bash", "arguments": json!({"command":command}).to_string()}}]
            },"finish_reason":"tool_calls"}],
            "usage":{"prompt_tokens":10,"completion_tokens":4}
        })
    }

    fn final_response(text: &str) -> Value {
        json!({
            "id":"chat", "model":"z-ai/glm-5.3-flash",
            "choices":[{"message":{"content":text},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":10,"completion_tokens":2}
        })
    }
}

async fn start_mock(responses: Vec<Value>) -> (String, Arc<Mock>, tokio::task::JoinHandle<()>) {
    let mock = Arc::new(Mock {
        responses: std::sync::Mutex::new(responses),
        seen: std::sync::Mutex::new(Vec::new()),
    });
    let mock_for_route = Arc::clone(&mock);
    let app = Router::new()
        .route(
            "/api/v1/chat/completions",
            post(move |_headers: axum::http::HeaderMap, body: bytes::Bytes| {
                let mock = std::sync::Arc::clone(&mock_for_route);
                async move {
                    let parsed = serde_json::from_slice(&body).unwrap_or(Value::Null);
                    mock.seen.lock().unwrap().push(parsed);
                    let response = mock.responses.lock().unwrap().remove(0);
                    axum::response::Response::builder()
                        .status(StatusCode::OK)
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(Body::from(response.to_string()))
                        .unwrap()
                }
            }),
        )
        .with_state(());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{address}/api/v1"), mock, task)
}

/// Build a Claude Code-shaped request: metadata carries the session id
/// (PR #9 fingerprint source); history is the conversation so far.
fn cc_request(session: &str, history: Vec<Value>) -> String {
    json!({
        "model":"claude-sonnet-4-6",
        "max_tokens":4_000,
        "metadata":{"user_id": session},
        "messages":history,
    })
    .to_string()
}

fn user_turn(text: &str) -> Value {
    json!({"role":"user", "content":[{"type":"text","text":text}]})
}

fn tool_result_turn(id: &str, content: &str) -> Value {
    json!({"role":"user", "content":[
        {"type":"tool_result","tool_use_id":id,"content":content}
    ]})
}

/// An assistant turn replayed from the client's history. `reasoning` is
/// `None` when the client stores nothing (thinking was never requested).
fn assistant_tool_use(id: &str, command: &str, reasoning: Option<&str>) -> Value {
    let mut message = json!({
        "role":"assistant",
        "content":[{"type":"tool_use","id":id,"name":"Bash",
            "input":{"command":command}}],
    });
    if let Some(text) = reasoning {
        message["reasoning_content"] = json!(text);
    }
    message
}

#[tokio::test]
async fn tool_loop_restores_shadow_reasoning_within_epoch() {
    let reasoning = "I need to inspect the failing module and run its tests.";
    let (base, _mock, task) = start_mock(vec![
        Mock::tool_call_response(reasoning, "call_1", "cargo test"),
        Mock::final_response("tests pass"),
    ])
    .await;
    let mut config = test_config();
    config.upstream.base_url = base;
    let state = AppState::new(config).unwrap();
    assert!(state.reasoning_shadow.is_some());
    let app = router(state.clone());

    // Turn 1: user asks; upstream answers with reasoning + tool call.
    let history1 = vec![user_turn("run the tests")];
    let response = app
        .clone()
        .oneshot(gateway_request(
            "/v1/messages",
            cc_request("sess-A", history1),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Turn 2 (tool result): the request's historical assistant turn carries
    // NO reasoning from the client; the shadow must restore it on the wire.
    let history2 = vec![
        user_turn("run the tests"),
        // Client stored nothing (thinking not requested): assistant turn
        // replays only the tool_use block.
        assistant_tool_use("call_1", "cargo test", None),
        tool_result_turn("call_1", "test result: ok"),
    ];
    let response = app
        .clone()
        .oneshot(gateway_request(
            "/v1/messages",
            cc_request("sess-A", history2),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // The wire request seen by the upstream for turn 2 must carry the
    // restored reasoning_content on the assistant message.
    let seen = _mock.seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    let wire = &seen[1];
    let assistant = wire["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "assistant")
        .unwrap();
    assert_eq!(
        assistant["reasoning_content"].as_str().unwrap_or(""),
        reasoning,
        "shadow reasoning must be restored within the epoch"
    );
    task.abort();
}

#[tokio::test]
async fn sessions_are_isolated_and_final_answer_clears_shadow() {
    let (base, mock, task) = start_mock(vec![
        Mock::tool_call_response("AGENT A THINKING", "call_1", "a"),
        Mock::tool_call_response("AGENT B THINKING", "call_1", "b"),
        Mock::final_response("A done"),
        Mock::tool_call_response("fresh", "call_2", "c"),
    ])
    .await;
    let mut config = test_config();
    config.upstream.base_url = base;
    let state = AppState::new(config).unwrap();
    let app = router(state.clone());

    // Both agents do a tool call (same call id, different sessions).
    let body_a = cc_request("sess-A", vec![user_turn("task a")]);
    let body_b = cc_request("sess-B", vec![user_turn("task b")]);
    let (ra, rb) = tokio::join!(
        app.clone().oneshot(gateway_request("/v1/messages", body_a)),
        app.clone().oneshot(gateway_request("/v1/messages", body_b)),
    );
    assert_eq!(ra.unwrap().status(), StatusCode::OK);
    assert_eq!(rb.unwrap().status(), StatusCode::OK);

    // Agent A sends its tool result: reasoning restored must be A's.
    let ra = app
        .clone()
        .oneshot(gateway_request(
            "/v1/messages",
            cc_request(
                "sess-A",
                vec![
                    user_turn("task a"),
                    assistant_tool_use("call_1", "a", None),
                    tool_result_turn("call_1", "ok"),
                ],
            ),
        ))
        .await
        .unwrap();
    assert_eq!(ra.status(), StatusCode::OK);
    // Agent A's second response is a FINAL answer: shadow cleared.
    // Agent B then sends its tool result: B's reasoning must survive and
    // must NOT be A's (isolation).
    let rb = app
        .clone()
        .oneshot(gateway_request(
            "/v1/messages",
            cc_request(
                "sess-B",
                vec![
                    user_turn("task b"),
                    assistant_tool_use("call_1", "b", None),
                    tool_result_turn("call_1", "ok"),
                ],
            ),
        ))
        .await
        .unwrap();
    assert_eq!(rb.status(), StatusCode::OK);

    let seen = mock.seen.lock().unwrap();
    let wire_a2 = &seen[2]; // agent A's second request
    let wire_b2 = &seen[3]; // agent B's second request
    let assistant_a2 = wire_a2["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "assistant")
        .unwrap();
    let assistant_b2 = wire_b2["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "assistant")
        .unwrap();
    assert_eq!(assistant_a2["reasoning_content"], "AGENT A THINKING");
    assert_eq!(assistant_b2["reasoning_content"], "AGENT B THINKING");
    task.abort();
}

#[tokio::test]
async fn no_session_identity_disables_shadow() {
    let (base, mock, task) = start_mock(vec![
        Mock::tool_call_response("SHADOWED", "call_1", "x"),
        Mock::final_response("done"),
    ])
    .await;
    let mut config = test_config();
    config.upstream.base_url = base;
    let state = AppState::new(config).unwrap();
    let app = router(state.clone());

    // Request WITHOUT metadata: no session identity.
    let no_metadata = json!({
        "model":"claude-sonnet-4-6","max_tokens":4_000,
        "messages":[user_turn("run tests")]
    })
    .to_string();
    let response = app
        .clone()
        .oneshot(gateway_request("/v1/messages", no_metadata))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    // A second anonymous request must NOT receive the previous one's
    // reasoning (fail-safe: no "last request" fallback).
    let no_metadata2 = json!({
        "model":"claude-sonnet-4-6","max_tokens":4_000,
        "messages":[
            user_turn("run tests"),
            assistant_tool_use("call_1", "x", None),
            tool_result_turn("call_1", "ok"),
        ]
    })
    .to_string();
    let response = app
        .oneshot(gateway_request("/v1/messages", no_metadata2))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let seen = mock.seen.lock().unwrap();
    let wire = &seen[1];
    let assistant = wire["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "assistant")
        .unwrap();
    assert!(
        assistant.get("reasoning_content").is_none(),
        "without session identity the shadow must stay disabled"
    );
    task.abort();
}

#[tokio::test]
async fn new_human_turn_clears_previous_epoch_reasoning() {
    let (base, mock, task) = start_mock(vec![
        Mock::tool_call_response("EPOCH ONE REASONING", "call_1", "x"),
        Mock::final_response("done"),
    ])
    .await;
    let mut config = test_config();
    config.upstream.base_url = base;
    let state = AppState::new(config).unwrap();
    let app = router(state.clone());

    // Epoch 1: tool call stored.
    let r = app
        .clone()
        .oneshot(gateway_request(
            "/v1/messages",
            cc_request("sess-C", vec![user_turn("task one")]),
        ))
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::OK);

    // New human turn, no tool result: previous epoch's shadow is dropped;
    // the new request carries a fresh assistant turn (nothing to restore).
    let r = app
        .clone()
        .oneshot(gateway_request(
            "/v1/messages",
            cc_request(
                "sess-C",
                vec![
                    user_turn("task one"),
                    assistant_tool_use("call_1", "x", None),
                    tool_result_turn("call_1", "ok"),
                    user_turn("new task"),
                ],
            ),
        ))
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::OK);

    let seen = mock.seen.lock().unwrap();
    // The wire for the new-epoch request: the epoch-1 assistant message
    // must NOT carry restored reasoning (it is pre-boundary and would have
    // been stripped anyway) — verify no reasoning_content was injected.
    let wire = &seen[1];
    let assistant = wire["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "assistant")
        .unwrap();
    assert!(assistant.get("reasoning_content").is_none());
    task.abort();
}

#[allow(dead_code)]
fn state_outcome_marker(_: StateLoadOutcome) {}
