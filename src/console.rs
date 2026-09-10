//! Console destination and color policy.
//!
//! Default `auto`: ANSI only when the destination is an actual terminal and
//! `NO_COLOR` is unset. Redirected files must contain no CSI escapes.

use std::io::IsTerminal;

/// How the process should emit ANSI color on the diagnostic stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorMode {
    #[default]
    Auto,
    Always,
    Never,
}

impl ColorMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "always" | "on" | "true" => Some(Self::Always),
            "never" | "off" | "false" => Some(Self::Never),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Always => "always",
            Self::Never => "never",
        }
    }
}

/// `NO_COLOR` is honored when the variable is present and non-empty.
pub fn no_color_set() -> bool {
    std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty())
}

/// Whether the diagnostic writer should emit ANSI.
///
/// Precedence: explicit `Never` / `Always` override the environment;
/// `Auto` disables color when `NO_COLOR` is set or the stream is not a
/// terminal.
pub fn enable_ansi(mode: ColorMode, is_terminal: bool, no_color: bool) -> bool {
    match mode {
        ColorMode::Never => false,
        ColorMode::Always => true,
        ColorMode::Auto => is_terminal && !no_color,
    }
}

pub fn stderr_is_terminal() -> bool {
    std::io::stderr().is_terminal()
}

pub fn resolve_stderr_ansi(mode: ColorMode) -> bool {
    enable_ansi(mode, stderr_is_terminal(), no_color_set())
}

/// Compact token figure: 199634 -> "199.6K".
pub fn format_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{:.1}K", tokens as f64 / 1_000.0)
    } else {
        tokens.to_string()
    }
}

/// Compact duration: 8400 -> "8.4s", 420 -> "420ms".
pub fn format_duration_ms(ms: u64) -> String {
    if ms >= 1_000 {
        format!("{:.1}s", ms as f64 / 1_000.0)
    } else {
        format!("{ms}ms")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_color_is_off_when_redirected_or_no_color() {
        assert!(!enable_ansi(ColorMode::Auto, false, false));
        assert!(!enable_ansi(ColorMode::Auto, true, true));
        assert!(!enable_ansi(ColorMode::Auto, false, true));
        assert!(enable_ansi(ColorMode::Auto, true, false));
    }

    #[test]
    fn explicit_always_and_never_override_environment() {
        assert!(enable_ansi(ColorMode::Always, false, true));
        assert!(!enable_ansi(ColorMode::Never, true, false));
    }

    #[test]
    fn color_mode_parses_aliases() {
        assert_eq!(ColorMode::parse("AUTO"), Some(ColorMode::Auto));
        assert_eq!(ColorMode::parse("on"), Some(ColorMode::Always));
        assert_eq!(ColorMode::parse("off"), Some(ColorMode::Never));
        assert_eq!(ColorMode::parse("nope"), None);
    }

    #[test]
    fn compact_formatters_are_stable() {
        assert_eq!(format_tokens(199_634), "199.6K");
        assert_eq!(format_tokens(216_018), "216.0K");
        assert_eq!(format_tokens(312), "312");
        assert_eq!(format_duration_ms(8_400), "8.4s");
        assert_eq!(format_duration_ms(420), "420ms");
    }

    #[test]
    fn ansi_csi_must_not_appear_in_plain_formatter_output() {
        let line = format!(
            "✓ GLM53 agent=abcd key=k in={} budget={} cache=99.8% out=312 ttft={} dur={}",
            format_tokens(199_634),
            format_tokens(216_018),
            format_duration_ms(8_400),
            format_duration_ms(11_700),
        );
        assert!(!line.as_bytes().contains(&0x1b));
        assert!(!line.contains('['));
    }
}
