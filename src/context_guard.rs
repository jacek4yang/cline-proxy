//! Optional provider-context budget.
//!
//! The guard is inert unless `upstream_context_window_tokens` is configured.
//! It never truncates user/tool/source text. Ordinary requests well below
//! the window skip exact tokenization; near the boundary the caller may
//! supply an exact GLM count.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextLimits {
    pub window_tokens: u64,
    pub safety_margin_tokens: u64,
    pub min_output_tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextDecision {
    /// Input + requested output fit; use `effective_max_tokens` unchanged.
    Allow {
        reserved_output_tokens: u64,
        total_context_budget: u64,
        headroom_tokens: u64,
    },
    /// Input fits; output cap reduced to remaining safe headroom.
    ReduceOutput {
        reserved_output_tokens: u64,
        total_context_budget: u64,
        headroom_tokens: u64,
        requested_output_tokens: u64,
    },
    /// Input itself leaves no meaningful output headroom — reject before
    /// paying for an upstream generation.
    RejectInput { total_needed: u64, window: u64 },
}

impl ContextDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Allow { .. } => "allow",
            Self::ReduceOutput { .. } => "reduce_output",
            Self::RejectInput { .. } => "reject_input",
        }
    }
}

/// Cheap conservative token estimate from UTF-8 bytes (1 byte → 1 token).
/// Over-estimates tokens, which is the fail-safe direction.
pub fn conservative_tokens_from_bytes(bytes: usize) -> u64 {
    u64::try_from(bytes).unwrap_or(u64::MAX)
}

/// True when the conservative byte estimate is far enough below the window
/// that an exact count is unnecessary.
pub fn well_below_window(estimated_input_tokens: u64, limits: ContextLimits, output: u64) -> bool {
    let needed = estimated_input_tokens
        .saturating_add(output)
        .saturating_add(limits.safety_margin_tokens);
    // Require 25% headroom under the conservative estimate so a 4× tokenizer
    // overestimate still cannot hide a real overflow.
    needed.saturating_mul(4) < limits.window_tokens
}

pub fn evaluate(
    input_tokens: u64,
    requested_output_tokens: u64,
    limits: ContextLimits,
) -> ContextDecision {
    let window = limits.window_tokens;
    let margin = limits.safety_margin_tokens;
    let min_out = limits.min_output_tokens.max(1);
    let usable = window.saturating_sub(margin);
    let remaining = usable.saturating_sub(input_tokens);
    let total_with_requested = input_tokens.saturating_add(requested_output_tokens);
    if remaining < min_out {
        return ContextDecision::RejectInput {
            total_needed: input_tokens.saturating_add(min_out).saturating_add(margin),
            window,
        };
    }
    if requested_output_tokens <= remaining {
        return ContextDecision::Allow {
            reserved_output_tokens: requested_output_tokens,
            total_context_budget: total_with_requested,
            headroom_tokens: remaining.saturating_sub(requested_output_tokens),
        };
    }
    let reserved = remaining;
    ContextDecision::ReduceOutput {
        reserved_output_tokens: reserved,
        total_context_budget: input_tokens.saturating_add(reserved),
        headroom_tokens: 0,
        requested_output_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(window: u64) -> ContextLimits {
        ContextLimits {
            window_tokens: window,
            safety_margin_tokens: 1_024,
            min_output_tokens: 256,
        }
    }

    #[test]
    fn context_budget_arithmetic_allow() {
        let decision = evaluate(10_000, 1_000, limits(32_768));
        assert_eq!(
            decision,
            ContextDecision::Allow {
                reserved_output_tokens: 1_000,
                total_context_budget: 11_000,
                headroom_tokens: 32_768 - 1_024 - 10_000 - 1_000,
            }
        );
    }

    #[test]
    fn output_reserve_is_reduced_near_the_edge() {
        let decision = evaluate(30_000, 8_000, limits(32_768));
        match decision {
            ContextDecision::ReduceOutput {
                reserved_output_tokens,
                requested_output_tokens,
                headroom_tokens,
                ..
            } => {
                assert_eq!(requested_output_tokens, 8_000);
                assert_eq!(reserved_output_tokens, 32_768 - 1_024 - 30_000);
                assert_eq!(headroom_tokens, 0);
            }
            other => panic!("expected reduce, got {other:?}"),
        }
    }

    #[test]
    fn input_that_cannot_fit_is_rejected() {
        let decision = evaluate(32_000, 8_000, limits(32_768));
        match decision {
            ContextDecision::RejectInput { window, .. } => assert_eq!(window, 32_768),
            other => panic!("expected reject, got {other:?}"),
        }
    }

    #[test]
    fn small_requests_skip_exact_count() {
        let limits = limits(256_000);
        assert!(well_below_window(1_000, limits, 16_384));
        assert!(!well_below_window(200_000, limits, 16_384));
    }

    #[test]
    fn byte_estimate_is_conservative() {
        assert_eq!(conservative_tokens_from_bytes(199_634), 199_634);
    }
}
