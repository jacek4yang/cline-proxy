//! Exact GLM-5.3-Flash token accounting.
//!
//! Pipeline (see docs/GLM53_FLASH.md and docs/adr/0003-glm53-exact-tokenizer.md):
//!
//! ```text
//! Anthropic Messages request
//!     -> semantic GLM message representation (messages.rs)
//!     -> official chat_template.jinja rendered in-process (template.rs)
//!     -> official tokenizer.json, in-process (tokenizer.rs)
//!     -> exact input_tokens
//! ```
//!
//! The embedded assets are the official `zai-org/GLM-5.3-Flash` files at
//! revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` (MIT). Golden fixtures
//! under `tests/fixtures/glm53/` were produced by
//! `tools/glm_reference/generate_fixtures.py` from those exact files.

pub mod count;
pub mod messages;
pub mod reasoning;
pub mod template;
pub mod tokenizer;

/// Errors from the exact counting pipeline. These are request-shape problems
/// (unsupported or malformed content), never tokenizer internals.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct CountError {
    pub error_type: &'static str,
    pub message: String,
}

impl CountError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            error_type: "invalid_request_error",
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod fixture_tests {
    use serde_json::Value;

    fn fixture(name: &str) -> Value {
        let raw = std::fs::read_to_string(format!(
            "{}/tests/fixtures/glm53/{name}.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        serde_json::from_str(&raw).unwrap()
    }

    /// Full parity check against one golden fixture: message conversion,
    /// tool passthrough, effort mapping, and the exact final token count.
    fn check_fixture(name: &str) {
        let value = fixture(name);
        let request = &value["anthropic_request"];
        let conversion = crate::glm53::messages::convert_anthropic_to_glm(request).unwrap();
        assert_eq!(
            serde_json::to_value(&conversion.messages).unwrap(),
            value["glm_messages"],
            "{name}: message conversion must match the official-oracle fixture"
        );
        assert_eq!(
            conversion
                .tools
                .as_ref()
                .map(|tools| serde_json::to_value(tools).unwrap())
                .unwrap_or(Value::Array(Vec::new())),
            value["glm_tools"],
            "{name}: tool passthrough must match the fixture"
        );
        let expected_effort = match value["reasoning_effort"].as_str() {
            Some("low") => Some(crate::glm53::reasoning::GlmReasoningEffort::Low),
            Some("high") => Some(crate::glm53::reasoning::GlmReasoningEffort::High),
            Some(other) => panic!("fixture {name} has unexpected effort {other}"),
            None => None,
        };
        assert_eq!(
            conversion.reasoning_effort, expected_effort,
            "{name}: effort"
        );
        let count = crate::glm53::count::count_input_tokens(request).unwrap();
        assert_eq!(
            count as u64, value["token_count"],
            "{name}: exact token count must equal the official reference"
        );
    }

    macro_rules! fixture_parity {
        ($($name:ident => $fixture:literal),* $(,)?) => {
            $(
                #[test]
                fn $name() {
                    check_fixture($fixture);
                }
            )*
        };
    }

    fixture_parity! {
        en_simple_matches_official_count => "en_simple",
        zh_simple_matches_official_count => "zh_simple",
        mixed_zh_en_matches_official_count => "mixed_zh_en",
        rust_code_matches_official_count => "rust_code",
        json_payload_matches_official_count => "json_payload",
        emoji_unicode_matches_official_count => "emoji_unicode",
        system_prompt_matches_official_count => "system_prompt",
        multi_turn_matches_official_count => "multi_turn",
        tools_small_matches_official_count => "tools_small",
        tools_large_strict_dropped_matches_official_count => "tools_large_strict_dropped",
        tool_loop_matches_official_count => "tool_loop",
        parallel_tool_calls_matches_official_count => "parallel_tool_calls",
        thinking_history_kept_matches_official_count => "thinking_history_kept",
        adaptive_thinking_high_matches_official_count => "adaptive_thinking_high",
        claude_code_like_matches_official_count => "claude_code_like",
    }

    #[test]
    fn unsupported_document_sources_are_rejected_not_undercounted() {
        let request = serde_json::json!({
            "model": "z-ai/glm-5.3-flash",
            "messages": [{"role": "user", "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "AAAA"}}
            ]}]
        });
        let error = crate::glm53::count::count_input_tokens(&request).unwrap_err();
        assert_eq!(error.error_type, "invalid_request_error");
        assert!(error.message.contains("cannot be counted exactly"));
    }
}
