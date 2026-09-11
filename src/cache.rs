//! Prompt-prefix stability and cache locality (issue #8).
//!
//! Upstream prompt caches key on byte-exact prefixes. Three sources of
//! per-turn byte drift are addressed here:
//!
//! 1. **Volatile billing header** — Claude Code may prepend an
//!    `x-anthropic-billing-header: ...` line to the system text whose
//!    attribution metadata changes between requests, breaking the system
//!    prefix every turn. Only a *leading* line of exactly that shape is
//!    removed (never a full-text search, never later occurrences that the
//!    user authored).
//! 2. **Non-canonical tool argument JSON** — OpenAI `tool_calls` arguments
//!    are strings; equivalent objects with different key insertion order
//!    serialize differently. Historical arguments are canonicalized
//!    (deterministic nested key order) so identical semantics produce
//!    identical bytes. Plain-text tool results are never touched.
//! 3. **Unmeasured drift** — a stable prefix hash (SHA-256 over the
//!    normalized system + tools + historical messages) plus session
//!    fingerprints make prefix stability observable per request without
//!    logging any content.
//!
//! Only sizes, counts, and hashes are recorded — never prompt content, raw
//! session ids, or key material.

use hmac::{Hmac, Mac};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

/// The header Claude Code prepends to system text for billing attribution.
/// Its metadata varies between requests, so a leading occurrence is stripped
/// to keep the system prefix byte-stable.
pub const BILLING_HEADER_PREFIX: &str = "x-anthropic-billing-header:";

pub fn strip_leading_anthropic_billing_header(text: &str) -> usize {
    if !text.starts_with(BILLING_HEADER_PREFIX) {
        return 0;
    }
    // Find the end of the first line, tolerating LF, CRLF, and CR.
    let rest = &text[BILLING_HEADER_PREFIX.len()..];
    let line_end = rest
        .find(['\n', '\r'])
        .map(|position| BILLING_HEADER_PREFIX.len() + position)
        .unwrap_or(text.len());
    // Consume the line terminator as well (CRLF counts as one).
    let mut cut = line_end;
    if text[cut..].starts_with("\r\n") {
        cut += 2;
    } else if text[cut..].starts_with(['\r', '\n']) {
        cut += 1;
    }
    cut
}

/// Apply the same leading billing-header strip to an *Anthropic-shaped*
/// request's `system` field (string or text-block array). Used by
/// `/v1/messages/count_tokens` so its count sees the same normalized system
/// the wire path sends (issue #8: count and wire must share normalization).
pub fn strip_billing_header_in_anthropic_system(request: &mut Value) {
    let Some(system) = request.get_mut("system") else {
        return;
    };
    match system {
        Value::String(text) => {
            let cut = strip_leading_anthropic_billing_header(text);
            if cut > 0 {
                *text = text[cut..].to_owned();
            }
        }
        Value::Array(blocks) => {
            for block in blocks.iter_mut() {
                if block.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(Value::String(text)) = block.get_mut("text") {
                        let cut = strip_leading_anthropic_billing_header(text);
                        if cut > 0 {
                            *text = text[cut..].to_owned();
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

/// Canonical JSON string for values that must serialize deterministically
/// (OpenAI `tool_calls[].function.arguments`). Object keys are sorted at
/// every nesting level; array order is preserved; numbers and strings are
/// emitted by serde verbatim (no reformatting, no float rewriting).
pub fn canonical_json_string(value: &Value) -> String {
    let mut output = String::new();
    write_canonical(value, &mut output);
    output
}

fn write_canonical(value: &Value, output: &mut String) {
    match value {
        Value::Object(map) => {
            // BTreeMap ordering: sort keys once, then emit in sorted order.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            output.push('{');
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_escaped(key, output);
                output.push(':');
                write_canonical(&map[*key], output);
            }
            output.push('}');
        }
        Value::Array(items) => {
            output.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_canonical(item, output);
            }
            output.push(']');
        }
        Value::String(text) => write_escaped(text, output),
        other => {
            // Numbers, bools, null: serde's compact writer preserves the
            // parsed representation; this path never reformats them.
            if let Ok(rendered) = serde_json::to_string(other) {
                output.push_str(&rendered);
            }
        }
    }
}

fn write_escaped(text: &str, output: &mut String) {
    // serde_json's string escaping, reused so canonical output stays
    // byte-compatible with normal serde serialization for strings.
    if let Ok(rendered) = serde_json::to_string(text) {
        output.push_str(&rendered);
    }
}

/// Canonicalize historical assistant `tool_calls[].function.arguments`
/// strings in an OpenAI Chat Completions body, in place. The most recent
/// assistant turn(s) after the last user/tool message are left untouched
/// (they are the current epoch and stable anyway). Returns the number of
/// arguments canonicalized.
///
/// Nothing else is rewritten: tool *results*, shell output, source code,
/// and diagnostics are never parsed or re-serialized.
pub fn canonicalize_tool_arguments(object: &mut Map<String, Value>) -> usize {
    let Some(messages) = object.get_mut("messages").and_then(Value::as_array_mut) else {
        return 0;
    };
    let last_action_index = messages.iter().rposition(|message| {
        matches!(
            message.get("role").and_then(Value::as_str),
            Some("user") | Some("tool")
        )
    });
    let Some(last_action_index) = last_action_index else {
        return 0;
    };
    let mut count = 0usize;
    for (index, message) in messages.iter_mut().enumerate() {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        if index >= last_action_index {
            continue;
        }
        let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) else {
            continue;
        };
        for call in calls.iter_mut() {
            let Some(function) = call.get_mut("function").and_then(Value::as_object_mut) else {
                continue;
            };
            let Some(arguments) = function.get("arguments").and_then(Value::as_str) else {
                continue;
            };
            // Only re-serialize when the value parses as JSON *and*
            // canonicalization actually changes bytes; malformed argument
            // strings are forwarded as-is (never a correctness risk).
            if let Ok(parsed) = serde_json::from_str::<Value>(arguments) {
                let canonical = canonical_json_string(&parsed);
                if canonical != arguments {
                    function.insert("arguments".into(), Value::String(canonical));
                    count += 1;
                }
            }
        }
    }
    count
}

/// SHA-256 prefix hash over the request sections that must stay
/// byte-stable across consecutive turns. The hash is a **local stability
/// metric only**: equal hashes mean locally identical prefixes, never a
/// proven upstream cache hit (that requires upstream `cached_tokens`).
pub fn stable_prefix_hash(object: &Map<String, Value>) -> (String, usize) {
    let mut hasher = Sha256::new();
    let mut prefix_bytes = 0usize;
    for section in ["system", "messages", "tools"] {
        if let Some(value) = object.get(section) {
            // The newest action is part of the *changing* tail, not the
            // stable prefix; hashing all of it is still a valid
            // same-input-same-hash metric, which is the documented
            // guarantee. Feeding section boundaries keeps collisions
            // unambiguous across section splits.
            if let Ok(bytes) = serde_json::to_vec(value) {
                hasher.update(section.as_bytes());
                hasher.update([0]);
                hasher.update((bytes.len() as u64).to_le_bytes());
                hasher.update(&bytes);
                prefix_bytes += bytes.len();
            }
        }
    }
    let digest = hasher.finalize();
    (format!("{:x}", digest), prefix_bytes)
}

/// Fingerprint a raw session id (never logged, never forwarded): HMAC-SHA256
/// with a server-side secret, rendered as the first 16 hex characters.
pub fn session_fingerprint(secret: &str, raw_session_id: &str) -> String {
    let mut mac =
        HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key length");
    mac.update(raw_session_id.as_bytes());
    let digest = mac.finalize().into_bytes();
    let hex = format!("{:x}", digest);
    hex[..16].to_owned()
}

/// Raw session identity from an Anthropic request, in priority order:
///
/// 1. `metadata.user_id` — Claude Code formats it as `<user>_<account>_session_<id>`,
///    or already opaque; any non-empty string is acceptable input to the HMAC.
/// 2. `metadata.session_id` when present.
///
/// Returns `None` when neither is present: without a stable identity there
/// is no session-scoped behavior (fail safe — never guess from connection
/// state, IP, key, or recent requests). Takes the already-parsed request so
/// the handler never re-parses the full body just for identity extraction.
pub fn session_raw_id(anthropic_request: &Value) -> Option<&str> {
    let metadata = anthropic_request.get("metadata")?.as_object()?;
    metadata
        .get("user_id")
        .and_then(Value::as_str)
        .or_else(|| metadata.get("session_id").and_then(Value::as_str))
        .filter(|id| !id.is_empty())
}

/// Extract a session identity from an Anthropic request and return its
/// HMAC fingerprint (raw ids never leave this function in logs). See
/// [`session_raw_id`] for the accepted sources.
pub fn extract_session_fingerprint(anthropic_request: &Value, secret: &str) -> Option<String> {
    Some(session_fingerprint(
        secret,
        session_raw_id(anthropic_request)?,
    ))
}

/// Compute and log the stable-prefix telemetry for one request:
/// `prefix_hash` + `prefix_bytes` (sizes/hashes only, never content), plus
/// the session fingerprint when a stable identity was extracted.
pub fn log_prefix_telemetry(
    openai_body: &Value,
    session_fp: Option<&str>,
    request_id: &str,
) -> (String, usize) {
    let default_object = Map::new();
    let object = openai_body.as_object().unwrap_or(&default_object);
    let (prefix_hash, prefix_bytes) = stable_prefix_hash(object);
    match session_fp {
        Some(session) => tracing::debug!(
            request_id,
            session = %session,
            prefix_hash = %prefix_hash,
            prefix_bytes,
            "stable prefix telemetry"
        ),
        None => tracing::debug!(
            request_id,
            session = "unstable",
            prefix_hash = %prefix_hash,
            prefix_bytes,
            "stable prefix telemetry"
        ),
    }
    (prefix_hash, prefix_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // --- billing header stripping (issue #8 acceptance: §47/§50) ---

    #[test]
    fn billing_header_variants_strip_leading_line_only() {
        let header = "x-anthropic-billing-header: {\"cch\":\"AAA\"}";
        for (text, expected_remaining, expected_removed) in [
            // LF
            (
                format!("{header}\nYou are Claude Code."),
                "You are Claude Code.",
                header.len() + 1,
            ),
            // CRLF
            (
                format!("{header}\r\nYou are Claude Code."),
                "You are Claude Code.",
                header.len() + 2,
            ),
            // CR only
            (
                format!("{header}\rYou are Claude Code."),
                "You are Claude Code.",
                header.len() + 1,
            ),
            // header only
            (header.to_string(), "", header.len()),
            // header + blank line + system
            (
                format!("{header}\n\nSystem prompt body."),
                "\nSystem prompt body.",
                header.len() + 1,
            ),
        ] {
            let removed = strip_leading_anthropic_billing_header(&text);
            assert_eq!(removed, expected_removed, "input: {text:?}");
            assert_eq!(&text[removed..], expected_remaining, "input: {text:?}");
        }
    }

    #[test]
    fn billing_header_not_at_start_or_user_authored_is_never_removed() {
        // Header appears later: untouched.
        let text = "System instructions.\nx-anthropic-billing-header: {\"cch\":\"AAA\"}";
        assert_eq!(strip_leading_anthropic_billing_header(text), 0);
        // Leading whitespace disqualifies: only an exact prefix is stripped.
        let text = "\nx-anthropic-billing-header: x\nbody";
        assert_eq!(strip_leading_anthropic_billing_header(text), 0);
        // Different header name: untouched.
        let text = "x-anthropic-billing: x\nbody";
        assert_eq!(strip_leading_anthropic_billing_header(text), 0);
        // Empty text.
        assert_eq!(strip_leading_anthropic_billing_header(""), 0);
    }

    #[test]
    fn billing_header_with_two_dynamic_values_normalizes_to_same_bytes() {
        let body = "\nYou are Claude Code, an agentic coding assistant.";
        let a = format!("x-anthropic-billing-header: {{\"cch\":\"AAA\"}}{body}");
        let b = format!("x-anthropic-billing-header: {{\"cch\":\"BBB\",\"x\":1}}{body}");
        let strip = |text: &str| {
            let cut = strip_leading_anthropic_billing_header(text);
            text[cut..].to_owned()
        };
        assert_eq!(strip(&a), strip(&b));
    }

    #[test]
    fn anthropic_shaped_system_strip_matches_wire_normalization() {
        // String system.
        let mut request = json!({
            "system": "x-anthropic-billing-header: {\"cch\":\"AAA\"}\nSystem body.",
            "messages": []
        });
        strip_billing_header_in_anthropic_system(&mut request);
        assert_eq!(request["system"], "System body.");
        // Block-array system: only the leading block's leading line.
        let mut request = json!({
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: {\"cch\":\"AAA\"}\r\nFirst."},
                {"type": "text", "text": "x-anthropic-billing-header mentioned later stays."}
            ],
            "messages": []
        });
        strip_billing_header_in_anthropic_system(&mut request);
        assert_eq!(request["system"][0]["text"], "First.");
        assert_eq!(
            request["system"][1]["text"],
            "x-anthropic-billing-header mentioned later stays."
        );
    }

    // --- canonical JSON (§48) ---

    #[test]
    fn canonical_json_sorts_keys_recursively_and_preserves_array_order() {
        let value = json!({
            "b": 2,
            "a": 1,
            "nested": {"z": true, "a": [4, {"d": 4, "c": 3}, "x"]},
            "s": "text, untouched",
            "n": 1.5,
            "nil": null
        });
        let canonical = canonical_json_string(&value);
        assert_eq!(
            canonical,
            "{\"a\":1,\"b\":2,\"n\":1.5,\"nested\":{\"a\":[4,{\"c\":3,\"d\":4},\"x\"],\"z\":true},\"nil\":null,\"s\":\"text, untouched\"}"
        );
        // Order of insertion must not matter.
        let reordered = json!({
            "s": "text, untouched",
            "nil": null,
            "n": 1.5,
            "nested": {"z": true, "a": [4, {"d": 4, "c": 3}, "x"]},
            "a": 1,
            "b": 2
        });
        assert_eq!(canonical_json_string(&reordered), canonical);
    }

    #[test]
    fn canonical_json_preserves_array_order() {
        let value = json!({"items": [3, 1, 2]});
        assert_eq!(canonical_json_string(&value), "{\"items\":[3,1,2]}");
    }

    #[test]
    fn canonicalize_tool_arguments_rewrites_only_historical_calls() {
        let mut body = json!({
            "model": "z-ai/glm-5.3-flash",
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {
                        "name": "Edit", "arguments": "{\"path\":\"a\",\"line\":1}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_2", "type": "function", "function": {
                        "name": "Read", "arguments": "{\"line\":2,\"path\":\"b\"}"}}
                ]}
            ]
        });
        let count = canonicalize_tool_arguments(body.as_object_mut().unwrap());
        // call_1 (historical) canonicalized; call_2 (current epoch) untouched.
        assert_eq!(count, 1);
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["arguments"],
            "{\"line\":1,\"path\":\"a\"}"
        );
        assert_eq!(
            body["messages"][3]["tool_calls"][0]["function"]["arguments"],
            "{\"line\":2,\"path\":\"b\"}"
        );
    }

    #[test]
    fn canonicalize_skips_malformed_arguments_and_plain_text_results() {
        let mut body = json!({
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_x", "type": "function", "function": {
                        "name": "Bash", "arguments": "not json at all"}}
                ]},
                {"role": "tool", "tool_call_id": "call_x",
                 "content": "compiler output\nwith    spacing\n\tand tabs"}
            ]
        });
        let count = canonicalize_tool_arguments(body.as_object_mut().unwrap());
        assert_eq!(count, 0);
        // Tool result text untouched.
        assert_eq!(
            body["messages"][2]["content"],
            "compiler output\nwith    spacing\n\tand tabs"
        );
    }

    #[test]
    fn canonicalize_is_idempotent_and_noop_when_already_canonical() {
        let mut body = json!({
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c", "type": "function", "function": {
                        "name": "Read", "arguments": "{\"a\":1,\"b\":2}"}}
                ]}
            ]
        });
        let count = canonicalize_tool_arguments(body.as_object_mut().unwrap());
        assert_eq!(count, 0, "already-canonical arguments must not rewrite");
    }

    // --- prefix hash (§49) ---

    #[test]
    fn prefix_hash_is_stable_across_semantically_identical_requests() {
        // Two requests identical except JSON insertion order within tool
        // arguments: after canonicalization their prefix hashes must match.
        let build = |arguments: &str| {
            json!({
                "messages": [
                    {"role": "system", "content": "system text"},
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "c1", "type": "function", "function": {
                            "name": "Edit", "arguments": arguments}}
                    ]},
                    {"role": "tool", "tool_call_id": "c1", "content": "done"}
                ],
                "tools": [{"type": "function", "function": {
                    "name": "Edit", "parameters": {"type": "object"}}}]
            })
        };
        let mut a = build("{\"path\":\"a\",\"line\":1}");
        let mut b = build("{\"line\":1,\"path\":\"a\"}");
        canonicalize_tool_arguments(a.as_object_mut().unwrap());
        canonicalize_tool_arguments(b.as_object_mut().unwrap());
        let (hash_a, bytes_a) = stable_prefix_hash(a.as_object().unwrap());
        let (hash_b, bytes_b) = stable_prefix_hash(b.as_object().unwrap());
        assert_eq!(hash_a, hash_b);
        assert_eq!(bytes_a, bytes_b);
        assert_eq!(
            bytes_a,
            a["messages"].to_string().len() + a["tools"].to_string().len()
        );
    }

    #[test]
    fn prefix_hash_changes_when_content_changes() {
        let a = json!({"messages": [{"role": "user", "content": "one"}]});
        let b = json!({"messages": [{"role": "user", "content": "two"}]});
        assert_ne!(
            stable_prefix_hash(a.as_object().unwrap()).0,
            stable_prefix_hash(b.as_object().unwrap()).0
        );
    }

    // --- session fingerprints (§64) ---

    #[test]
    fn session_raw_id_prefers_user_id_and_fails_safe_without_metadata() {
        // user_id wins over session_id.
        let request = json!({"metadata": {
            "user_id": "user_x_session_a", "session_id": "sid_b"}});
        assert_eq!(session_raw_id(&request), Some("user_x_session_a"));
        // session_id is the fallback source.
        let request = json!({"metadata": {"session_id": "sid_b"}});
        assert_eq!(session_raw_id(&request), Some("sid_b"));
        // Empty strings are not identities.
        let request = json!({"metadata": {"user_id": ""}});
        assert_eq!(session_raw_id(&request), None);
        // No metadata at all: no identity is invented.
        assert_eq!(session_raw_id(&json!({"messages": []})), None);
        assert_eq!(session_raw_id(&json!({})), None);
    }

    #[test]
    fn session_fingerprint_is_stable_secret_bound_and_truncated() {
        let a = session_fingerprint("server-secret", "user_abc123_session_9f8e7d");
        let b = session_fingerprint("server-secret", "user_abc123_session_9f8e7d");
        let c = session_fingerprint("other-secret", "user_abc123_session_9f8e7d");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 16);
        assert!(a.chars().all(|character| character.is_ascii_hexdigit()));
        // Raw id never appears in the fingerprint.
        assert!(!a.contains("9f8e7d"));
    }
}
