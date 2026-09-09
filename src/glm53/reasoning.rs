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

/// When upstream `reasoning_content` may be forwarded to the client as
/// Anthropic `thinking` blocks.
///
/// `requested_only` (the default) is the anti-amplification gate: GLM's
/// reasoning is consumed in-turn and never mirrored to Claude Code unless
/// the request explicitly carried a `thinking` object. Unexposed reasoning
/// cannot be stored by the client, sent back, and re-counted on every
/// subsequent turn. See docs/GLM53_FLASH.md.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingExposure {
    /// Expose thinking only when the request explicitly carried
    /// `thinking` with type `enabled` or `adaptive` (not `disabled`).
    #[default]
    RequestedOnly,
    /// Legacy behavior: always expose upstream reasoning (durable test hook).
    Always,
    /// Never expose upstream reasoning, even when requested.
    Never,
}

impl ThinkingExposure {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RequestedOnly => "requested_only",
            Self::Always => "always",
            Self::Never => "never",
        }
    }
}

/// Resolve a request's reasoning effort and thinking-exposure decision from
/// explicit Anthropic controls plus proxy defaults.
///
/// This is the single source of truth for both the OpenAI wire conversion
/// (`crate::anthropic`) and the exact-token-count pipeline
/// (`crate::glm53::messages`). Precedence, fixed and tested:
///
/// 1. explicit `output_config.effort`
/// 2. explicit `thinking`
/// 3. proxy default (`default_effort`)
///
/// The critical invariant: an effort is **always** produced. The official
/// template coerces unset/unknown efforts to `max`
/// (`reasoning_effort if reasoning_effort in ['low','high'] else 'max'`), so
/// "send nothing" silently selects maximal reasoning on every coding turn.
pub fn resolve_reasoning_policy(
    thinking: Option<&Value>,
    output_effort: Option<&str>,
    max_tokens: u64,
    default_effort: GlmReasoningEffort,
    adaptive_effort: GlmReasoningEffort,
    exposure: ThinkingExposure,
) -> Result<ReasoningPolicy, String> {
    let effort = if let Some(effort) = output_effort {
        Some(from_output_effort(effort)?)
    } else if let Some(thinking) = thinking {
        // from_thinking already maps `adaptive` to high; honoring a
        // separately configured adaptive tier keeps the knob meaningful
        // without forking the mapping rules.
        match from_thinking(thinking, max_tokens)? {
            Some(GlmReasoningEffort::High)
                if thinking.get("type").and_then(Value::as_str) == Some("adaptive") =>
            {
                Some(adaptive_effort)
            }
            other => other,
        }
    } else {
        Some(default_effort)
    };
    let expose = match exposure {
        ThinkingExposure::Always => true,
        ThinkingExposure::Never => false,
        ThinkingExposure::RequestedOnly => matches!(
            thinking
                .and_then(|value| value.get("type"))
                .and_then(Value::as_str),
            Some("enabled") | Some("adaptive")
        ),
    };
    Ok(ReasoningPolicy {
        effort: effort.unwrap_or(default_effort),
        expose_thinking: expose,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningPolicy {
    pub effort: GlmReasoningEffort,
    pub expose_thinking: bool,
}

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

    const DEFAULTS: (GlmReasoningEffort, GlmReasoningEffort) =
        (GlmReasoningEffort::High, GlmReasoningEffort::High);

    fn resolve(
        thinking: Option<&serde_json::Value>,
        output_effort: Option<&str>,
        exposure: ThinkingExposure,
    ) -> Result<ReasoningPolicy, String> {
        resolve_reasoning_policy(
            thinking,
            output_effort,
            64_000,
            DEFAULTS.0,
            DEFAULTS.1,
            exposure,
        )
    }

    /// The critical invariant: Claude Code does not send `thinking`, so the
    /// default must be explicit. `None` effort would let the official
    /// template coerce unset to `max`.
    #[test]
    fn missing_thinking_defaults_to_high_never_unset() {
        let policy = resolve(None, None, ThinkingExposure::RequestedOnly).unwrap();
        assert_eq!(policy.effort, GlmReasoningEffort::High);
        assert!(!policy.expose_thinking);
    }

    #[test]
    fn thinking_controls_map_to_effort_and_exposure() {
        let disabled = json!({"type": "disabled"});
        let policy = resolve(Some(&disabled), None, ThinkingExposure::RequestedOnly).unwrap();
        assert_eq!(policy.effort, GlmReasoningEffort::Low);
        // `disabled` is an explicit opt-out, not a request for thinking.
        assert!(!policy.expose_thinking);

        let adaptive = json!({"type": "adaptive"});
        let policy = resolve(Some(&adaptive), None, ThinkingExposure::RequestedOnly).unwrap();
        assert_eq!(policy.effort, GlmReasoningEffort::High);
        assert!(policy.expose_thinking);

        let small = json!({"type": "enabled", "budget_tokens": 4_096});
        let policy = resolve(Some(&small), None, ThinkingExposure::RequestedOnly).unwrap();
        assert_eq!(policy.effort, GlmReasoningEffort::Low);
        assert!(policy.expose_thinking);

        let large = json!({"type": "enabled", "budget_tokens": 32_768});
        let policy = resolve(Some(&large), None, ThinkingExposure::RequestedOnly).unwrap();
        assert_eq!(policy.effort, GlmReasoningEffort::High);
        assert!(policy.expose_thinking);
    }

    #[test]
    fn explicit_output_effort_wins_over_thinking_and_max_is_preserved() {
        let small = json!({"type": "enabled", "budget_tokens": 1_024});
        let policy = resolve(Some(&small), Some("max"), ThinkingExposure::RequestedOnly).unwrap();
        assert_eq!(policy.effort, GlmReasoningEffort::Max);

        let large = json!({"type": "enabled", "budget_tokens": 32_768});
        let policy = resolve(Some(&large), Some("low"), ThinkingExposure::RequestedOnly).unwrap();
        assert_eq!(policy.effort, GlmReasoningEffort::Low);
        // Exposure follows the thinking control, not the effort override.
        assert!(policy.expose_thinking);
    }

    #[test]
    fn exposure_modes_gate_independently_of_effort() {
        let adaptive = json!({"type": "adaptive"});
        let never = resolve(Some(&adaptive), None, ThinkingExposure::Never).unwrap();
        assert_eq!(never.effort, GlmReasoningEffort::High);
        assert!(!never.expose_thinking);

        let unrequested = resolve(None, None, ThinkingExposure::Always).unwrap();
        assert!(unrequested.expose_thinking);

        let disabled = json!({"type": "disabled"});
        let always = resolve(Some(&disabled), None, ThinkingExposure::Always).unwrap();
        assert!(always.expose_thinking);
    }
}
