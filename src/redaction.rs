//! Privacy-safe upstream error and log text sanitization.

use serde_json::Value;

const MAX_PUBLIC_TEXT_CHARS: usize = 1_024;

pub fn sanitize_text(input: &str, exact_secrets: &[&str]) -> String {
    let mut output = input.to_owned();
    for secret in exact_secrets {
        if !secret.is_empty() {
            output = output.replace(secret, "[REDACTED]");
        }
    }
    let words = output
        .split_whitespace()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let mut sanitized = Vec::with_capacity(words.len());
    let mut redact_next = false;
    for word in words {
        let trimmed = word.trim_matches(|character: char| {
            matches!(
                character,
                ',' | ';' | ':' | '"' | '\'' | '(' | ')' | '[' | ']'
            )
        });
        let lower = trimmed.to_ascii_lowercase();
        let sensitive_assignment = [
            "authorization=",
            "authorization:",
            "x-api-key=",
            "x-api-key:",
            "api_key=",
            "api-key=",
            "access_token=",
            "refresh_token=",
            "cookie=",
        ]
        .iter()
        .any(|prefix| lower.starts_with(prefix));
        let sensitive_value = redact_next
            || sensitive_assignment
            || lower.starts_with("sk-")
            || lower.starts_with("key-") && trimmed.len() > 24
            || looks_like_jwt(trimmed);
        if sensitive_value {
            sanitized.push("[REDACTED]".to_string());
        } else {
            sanitized.push(word);
        }
        redact_next = lower == "bearer" || lower == "authorization:" || lower == "x-api-key:";
    }
    truncate_chars(&sanitized.join(" "), MAX_PUBLIC_TEXT_CHARS)
}

pub fn sanitize_json(mut value: Value, exact_secrets: &[&str]) -> Value {
    redact_value(&mut value, exact_secrets);
    value
}

fn redact_value(value: &mut Value, exact_secrets: &[&str]) {
    match value {
        Value::Object(object) => {
            for (name, value) in object {
                if sensitive_name(name) {
                    *value = Value::String("[REDACTED]".into());
                } else {
                    redact_value(value, exact_secrets);
                }
            }
        }
        Value::Array(array) => {
            for value in array {
                redact_value(value, exact_secrets);
            }
        }
        Value::String(text) => *text = sanitize_text(text, exact_secrets),
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn sensitive_name(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase().replace('-', "_");
    normalized == "authorization"
        || normalized == "x_api_key"
        || normalized == "api_key"
        || normalized == "access_token"
        || normalized == "refresh_token"
        || normalized == "cookie"
        || normalized == "set_cookie"
        || normalized.ends_with("_secret")
}

fn looks_like_jwt(value: &str) -> bool {
    let mut segments = value.split('.');
    matches!(
        (segments.next(), segments.next(), segments.next(), segments.next()),
        (Some(first), Some(second), Some(third), None)
            if first.len() >= 8 && second.len() >= 8 && third.len() >= 8
                && [first, second, third].iter().all(|segment| segment
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_')))
    )
}

fn truncate_chars(input: &str, max: usize) -> String {
    let mut characters = input.chars();
    let prefix = characters.by_ref().take(max).collect::<String>();
    if characters.next().is_some() {
        format!("{prefix}…")
    } else {
        prefix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn recursive_redaction_removes_credentials() {
        let value = sanitize_json(
            json!({
                "authorization":"Bearer secret-token",
                "nested":{"api_key":"secret-key", "message":"Bearer another-secret"},
                "safe":"hello"
            }),
            &["secret-key"],
        );
        let rendered = value.to_string();
        assert!(!rendered.contains("secret-token"));
        assert!(!rendered.contains("another-secret"));
        assert!(!rendered.contains("secret-key"));
        assert!(rendered.contains("hello"));
    }

    #[test]
    fn exact_gateway_and_cline_keys_are_removed() {
        let output = sanitize_text(
            "gateway-secret and cline-secret and sk-provider-secret",
            &["gateway-secret", "cline-secret"],
        );
        assert!(!output.contains("secret"));
        assert_eq!(output.matches("[REDACTED]").count(), 3);
    }
}
