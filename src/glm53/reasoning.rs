//! Anthropic reasoning controls -> GLM-5.3-Flash reasoning_effort mapping.
//!
//! GLM-5.3-Flash supports `low`, `high`, and `max` (the official chat
//! template renders the chosen effort as a `Reasoning Effort:` preamble and
//! coerces anything outside {low, high} — including `max` given via the
//! default path — to `Max`). Mapping policy and evidence:
//! docs/GLM53_FLASH.md.

use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmReasoningEffort {
    Low,
    High,
    Max,
}

impl GlmReasoningEffort {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::High => "high",
            Self::Max => "max",
        }
    }
}

/// Budget threshold between the two routine tiers. GLM has no `medium`;
/// 8,192 is the documented midpoint (docs/GLM53_FLASH.md).
pub const LOW_BUDGET_LIMIT: u64 = 8_192;

/// Map an Anthropic `thinking` object to a GLM reasoning effort.
/// `disabled` maps to `low` (minimal thinking), NOT to unset — the official
/// template default for unset is `max`, the opposite of what a caller
/// disabling thinking asked for.
pub fn from_thinking(value: &Value, max_tokens: u64) -> Result<Option<GlmReasoningEffort>, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "thinking must be an object".to_string())?;
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| "thinking.type must be a string".to_string())?;
    match kind {
        "disabled" => Ok(Some(GlmReasoningEffort::Low)),
        "adaptive" => Ok(Some(GlmReasoningEffort::High)),
        "enabled" => {
            let budget = object
                .get("budget_tokens")
                .and_then(Value::as_u64)
                .filter(|value| *value > 0)
                .ok_or_else(|| "thinking budget_tokens must be positive".to_string())?;
            if budget >= max_tokens {
                return Err("thinking budget_tokens must be less than max_tokens".into());
            }
            Ok(Some(if budget < LOW_BUDGET_LIMIT {
                GlmReasoningEffort::Low
            } else {
                GlmReasoningEffort::High
            }))
        }
        other => Err(format!("unsupported thinking type {other:?}")),
    }
}

/// Map an Anthropic `output_config.effort` string. GLM has no `medium`;
/// moderate values map to `high`, and an explicit `max` is preserved.
pub fn from_output_effort(effort: &str) -> Result<GlmReasoningEffort, String> {
    match effort {
        "low" => Ok(GlmReasoningEffort::Low),
        "medium" | "high" | "xhigh" => Ok(GlmReasoningEffort::High),
        "max" => Ok(GlmReasoningEffort::Max),
        other => Err(format!("unsupported output_config.effort {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn thinking_mapping_is_documented_and_bounded() {
        assert_eq!(
            from_thinking(&json!({"type": "disabled"}), 1024).unwrap(),
            Some(GlmReasoningEffort::Low)
        );
        assert_eq!(
            from_thinking(&json!({"type": "adaptive"}), 1024).unwrap(),
            Some(GlmReasoningEffort::High)
        );
        assert_eq!(
            from_thinking(&json!({"type": "enabled", "budget_tokens": 8_191}), 16_384).unwrap(),
            Some(GlmReasoningEffort::Low)
        );
        assert_eq!(
            from_thinking(&json!({"type": "enabled", "budget_tokens": 8_192}), 4_096),
            Err("thinking budget_tokens must be less than max_tokens".into())
        );
        assert_eq!(
            from_thinking(
                &json!({"type": "enabled", "budget_tokens": 100_000}),
                200_000
            )
            .unwrap(),
            Some(GlmReasoningEffort::High)
        );
        assert!(from_thinking(&json!({"type": "galactic"}), 1024).is_err());
    }

    #[test]
    fn output_effort_mapping_preserves_max() {
        assert_eq!(from_output_effort("low").unwrap(), GlmReasoningEffort::Low);
        assert_eq!(
            from_output_effort("medium").unwrap(),
            GlmReasoningEffort::High
        );
        assert_eq!(
            from_output_effort("high").unwrap(),
            GlmReasoningEffort::High
        );
        assert_eq!(
            from_output_effort("xhigh").unwrap(),
            GlmReasoningEffort::High
        );
        assert_eq!(from_output_effort("max").unwrap(), GlmReasoningEffort::Max);
        assert!(from_output_effort("ultra").is_err());
    }
}
