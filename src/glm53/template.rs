//! Official GLM-5.3-Flash chat template, rendered in-process with minijinja.
//!
//! The template file is the unmodified official `chat_template.jinja` at the
//! pinned revision. Rendering parity with the Python/transformers oracle is
//! enforced by the golden fixtures in `tests/fixtures/glm53/`.

use std::sync::OnceLock;

use minijinja::Environment;
use serde_json::Value;

use crate::glm53::reasoning::GlmReasoningEffort;
use crate::glm53::CountError;

/// Official zai-org/GLM-5.3-Flash `chat_template.jinja`.
static CHAT_TEMPLATE_RAW: &str =
    include_str!("../../tools/glm_reference/assets/chat_template.jinja");

static ENVIRONMENT: OnceLock<Environment<'static>> = OnceLock::new();

/// The repository stores the template with LF endings, but a Windows
/// checkout with `core.autocrlf=true` materializes CRLF, and `include_str!`
/// would embed those bytes — silently changing every rendered prompt and
/// breaking exact token parity (5 extra tokens on tool fixtures). Normalize
/// to the official bytes regardless of checkout.
static CHAT_TEMPLATE: OnceLock<String> = OnceLock::new();

fn chat_template() -> &'static str {
    CHAT_TEMPLATE.get_or_init(|| CHAT_TEMPLATE_RAW.replace("\r\n", "\n"))
}

/// Python `json.dumps(..., ensure_ascii=False)`-compatible serialization:
/// insertion-ordered keys (serde_json `preserve_order`), `", "` item
/// separators, `": "` key/value separators, UTF-8 kept literal.
fn python_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".into(),
        serde_json::Value::Bool(flag) => flag.to_string(),
        serde_json::Value::Number(number) => number.to_string(),
        serde_json::Value::String(text) => {
            serde_json::to_string(text).unwrap_or_else(|_| "\"\"".into())
        }
        serde_json::Value::Array(items) => format!(
            "[{}]",
            items.iter().map(python_json).collect::<Vec<_>>().join(", ")
        ),
        serde_json::Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    serde_json::to_string(key).unwrap_or_else(|_| "\"\"".into()),
                    python_json(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn environment() -> &'static Environment<'static> {
    ENVIRONMENT.get_or_init(|| {
        let mut environment = Environment::new();
        // tojson in minijinja emits UTF-8 unescaped, matching the official
        // template's `tojson(ensure_ascii=False)`.
        // pycompat supplies the Python-style methods the official template
        // relies on (dict.items(), str.strip(), str.split()).
        environment.set_unknown_method_callback(|state, value, name, args| {
            minijinja_contrib::pycompat::unknown_method_callback(state, value, name, args)
        });
        // transformers renders chat templates with
        // ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True);
        // match those so whitespace is byte-identical to the official oracle.
        environment.set_trim_blocks(true);
        environment.set_lstrip_blocks(true);
        // The official template calls `tojson(ensure_ascii=False)`, i.e.
        // Python's json.dumps with default separators (", ", ": ") and
        // insertion-ordered keys. serde_json's compact output differs, so
        // serialize Python-style here. `ensure_ascii=False` means UTF-8
        // stays literal, which serde_json string escaping already does.
        environment.add_filter(
            "tojson",
            |value: minijinja::Value, kwargs: minijinja::value::Kwargs| {
                let ensure_ascii = kwargs.get::<Option<bool>>("ensure_ascii")?;
                let _ = ensure_ascii;
                kwargs.assert_all_used()?;
                let json_value: serde_json::Value =
                    serde_json::to_value(&value).map_err(|error| {
                        minijinja::Error::new(
                            minijinja::ErrorKind::InvalidOperation,
                            format!("tojson failed: {error}"),
                        )
                    })?;
                Ok(minijinja::Value::from(python_json(&json_value)))
            },
        );
        environment.add_template("chat", chat_template()).expect(
            "the embedded official chat template must compile; a minijinja \
             incompatibility is a build-time bug",
        );
        environment
    })
}

pub struct TemplateInput<'a> {
    pub messages: &'a Value,
    pub tools: Option<&'a Value>,
    pub reasoning_effort: Option<GlmReasoningEffort>,
    pub clear_thinking: bool,
    pub add_generation_prompt: bool,
}

/// Render the official template exactly as the reference oracle does.
pub fn render(input: &TemplateInput<'_>) -> Result<String, CountError> {
    let mut context = std::collections::BTreeMap::<&str, minijinja::Value>::new();
    context.insert("messages", minijinja::Value::from_serialize(input.messages));
    if let Some(tools) = input.tools {
        context.insert("tools", minijinja::Value::from_serialize(tools));
    }
    context.insert(
        "clear_thinking",
        minijinja::Value::from(input.clear_thinking),
    );
    context.insert(
        "add_generation_prompt",
        minijinja::Value::from(input.add_generation_prompt),
    );
    if let Some(effort) = input.reasoning_effort {
        context.insert("reasoning_effort", minijinja::Value::from(effort.as_str()));
    }
    environment()
        .get_template("chat")
        .expect("chat template is registered")
        .render(context)
        .map_err(|error| CountError {
            error_type: "api_error",
            message: format!("chat template rendering failed: {error}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture(name: &str) -> Value {
        let raw = std::fs::read_to_string(format!(
            "{}/tests/fixtures/glm53/{name}.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        serde_json::from_str(&raw).unwrap()
    }

    fn render_fixture(name: &str) {
        let value = fixture(name);
        let tools = value["glm_tools"]
            .as_array()
            .filter(|tools| !tools.is_empty())
            .map(|tools| Value::Array(tools.clone()));
        let input = TemplateInput {
            messages: &value["glm_messages"],
            tools: tools.as_ref(),
            reasoning_effort: match value["reasoning_effort"].as_str() {
                Some("low") => Some(GlmReasoningEffort::Low),
                Some("high") => Some(GlmReasoningEffort::High),
                Some(other) => panic!("unexpected effort {other}"),
                None => None,
            },
            clear_thinking: value["clear_thinking"].as_bool().unwrap(),
            add_generation_prompt: value["add_generation_prompt"].as_bool().unwrap(),
        };
        let rendered = render(&input).unwrap();
        let expected = value["rendered_prompt"].as_str().unwrap();
        assert_eq!(
            rendered, expected,
            "{name}: Rust template rendering must be byte-identical to the official oracle"
        );
    }

    #[test]
    fn rendering_is_byte_identical_to_official_oracle() {
        for name in [
            "en_simple",
            "zh_simple",
            "mixed_zh_en",
            "rust_code",
            "json_payload",
            "emoji_unicode",
            "system_prompt",
            "multi_turn",
            "tools_small",
            "tools_large_strict_dropped",
            "tool_loop",
            "parallel_tool_calls",
            "thinking_history_kept",
            "adaptive_thinking_high",
            "claude_code_like",
        ] {
            render_fixture(name);
        }
    }

    #[test]
    fn effort_coercion_matches_official_template() {
        // The official template coerces anything outside {low, high} to max.
        let messages = json!([{"role": "user", "content": "hi"}]);
        for (effort, expected) in [
            (None, "Reasoning Effort: Max"),
            (Some(GlmReasoningEffort::Max), "Reasoning Effort: Max"),
            (Some(GlmReasoningEffort::High), "Reasoning Effort: High"),
            (Some(GlmReasoningEffort::Low), "Reasoning Effort: Low"),
        ] {
            let rendered = render(&TemplateInput {
                messages: &messages,
                tools: None,
                reasoning_effort: effort,
                clear_thinking: false,
                add_generation_prompt: true,
            })
            .unwrap();
            assert!(
                rendered.starts_with(&format!("[gMASK]<sop><|system|>{expected}<|user|>")),
                "effort {effort:?} => {rendered}"
            );
        }
    }
}
