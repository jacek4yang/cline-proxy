//! Safe HTTP 429 retry-hint extraction.

use std::time::{Duration, SystemTime};

use axum::http::HeaderMap;
use serde_json::Value;

const MAX_PARSED_COOLDOWN: Duration = Duration::from_secs(366 * 24 * 60 * 60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryHintSource {
    RetryAfter,
    StructuredJson,
    HumanText,
    Fallback,
}

#[derive(Debug, Clone)]
pub struct RetryHint {
    pub duration: Duration,
    pub source: RetryHintSource,
    pub message: Option<String>,
    pub model: Option<String>,
}

pub fn retry_hint(headers: &HeaderMap, body: &[u8], fallback: Duration) -> RetryHint {
    let value = serde_json::from_slice::<Value>(body).ok();
    let text = String::from_utf8_lossy(body);
    let message = value
        .as_ref()
        .and_then(extract_error_message)
        .or_else(|| (!text.trim().is_empty()).then(|| text.trim().to_owned()));
    let model = value
        .as_ref()
        .and_then(extract_model)
        .or_else(|| message.as_deref().and_then(model_from_text));

    if let Some(duration) = headers
        .get("retry-after")
        .and_then(|header| header.to_str().ok())
        .and_then(parse_retry_after_header)
    {
        return RetryHint {
            duration,
            source: RetryHintSource::RetryAfter,
            message,
            model,
        };
    }
    if let Some(duration) = value.as_ref().and_then(structured_retry_duration) {
        return RetryHint {
            duration,
            source: RetryHintSource::StructuredJson,
            message,
            model,
        };
    }
    if let Some(duration) = message.as_deref().and_then(parse_retry_duration) {
        return RetryHint {
            duration,
            source: RetryHintSource::HumanText,
            message,
            model,
        };
    }
    RetryHint {
        duration: fallback.min(MAX_PARSED_COOLDOWN),
        source: RetryHintSource::Fallback,
        message,
        model,
    }
}

fn parse_retry_after_header(input: &str) -> Option<Duration> {
    let input = input.trim();
    if let Ok(seconds) = input.parse::<u64>() {
        return duration_with_cap(Duration::from_secs(seconds));
    }
    let when = httpdate::parse_http_date(input).ok()?;
    let duration = when.duration_since(SystemTime::now()).unwrap_or_default();
    duration_with_cap(duration)
}

fn structured_retry_duration(value: &Value) -> Option<Duration> {
    match value {
        Value::Object(object) => {
            for (name, value) in object {
                let lower = name.to_ascii_lowercase().replace(['-', '_'], "");
                let milliseconds = matches!(lower.as_str(), "retryafterms" | "retryinms");
                if matches!(
                    lower.as_str(),
                    "retryafter"
                        | "retryafterseconds"
                        | "retryin"
                        | "retryinseconds"
                        | "retryafterms"
                        | "retryinms"
                        | "cooldown"
                        | "cooldownseconds"
                ) {
                    let parsed = match value {
                        Value::Number(number) => number.as_u64().and_then(|amount| {
                            if milliseconds {
                                duration_with_cap(Duration::from_millis(amount))
                            } else {
                                duration_with_cap(Duration::from_secs(amount))
                            }
                        }),
                        Value::String(text) => parse_retry_duration(text),
                        _ => None,
                    };
                    if parsed.is_some() {
                        return parsed;
                    }
                }
            }
            object.values().find_map(structured_retry_duration)
        }
        Value::Array(values) => values.iter().find_map(structured_retry_duration),
        _ => None,
    }
}

/// Parse a duration embedded in common Cline rate-limit text. Invalid and
/// overflowing inputs return `None`; this function never panics.
pub fn parse_retry_duration(input: &str) -> Option<Duration> {
    let lower = input.to_ascii_lowercase();
    let mut candidate = lower.as_str();
    for marker in ["try again in", "retry after", "retry in"] {
        if let Some(position) = lower.find(marker) {
            candidate = &lower[position + marker.len()..];
            break;
        }
    }
    candidate = candidate.trim_start();
    if candidate.starts_with('-') {
        return None;
    }

    let bytes = candidate.as_bytes();
    let mut index = 0usize;
    let mut total_ms = 0u128;
    let mut components = 0usize;
    while index < bytes.len() {
        while index < bytes.len() && (bytes[index].is_ascii_whitespace() || bytes[index] == b',') {
            index += 1;
        }
        if index >= bytes.len() || !bytes[index].is_ascii_digit() {
            break;
        }
        let number_start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        let number = candidate[number_start..index].parse::<u128>().ok()?;
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        let (unit_ms, unit_len) = if candidate[index..].starts_with("ms") {
            (1u128, 2usize)
        } else if let Some(unit) = bytes.get(index).copied() {
            match unit {
                b'd' => (86_400_000, 1),
                b'h' => (3_600_000, 1),
                b'm' => (60_000, 1),
                b's' => (1_000, 1),
                _ => break,
            }
        } else {
            break;
        };
        let part = number.checked_mul(unit_ms)?;
        total_ms = total_ms.checked_add(part)?;
        components = components.saturating_add(1);
        index = index.saturating_add(unit_len);
    }
    if components == 0 {
        return None;
    }
    let total_ms = u64::try_from(total_ms).ok()?;
    duration_with_cap(Duration::from_millis(total_ms))
}

fn duration_with_cap(duration: Duration) -> Option<Duration> {
    (duration <= MAX_PARSED_COOLDOWN).then_some(duration)
}

fn extract_error_message(value: &Value) -> Option<String> {
    if let Some(text) = value.as_str() {
        return Some(text.to_owned());
    }
    let object = value.as_object()?;
    for key in ["message", "detail", "error_description"] {
        if let Some(text) = object.get(key).and_then(Value::as_str) {
            return Some(text.to_owned());
        }
    }
    object.get("error").and_then(extract_error_message)
}

fn extract_model(value: &Value) -> Option<String> {
    match value {
        Value::Object(object) => {
            if let Some(model) = object.get("model").and_then(Value::as_str) {
                return Some(model.to_owned());
            }
            object.values().find_map(extract_model)
        }
        Value::Array(values) => values.iter().find_map(extract_model),
        _ => None,
    }
}

fn model_from_text(text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let position = lower.find(" on model ")? + " on model ".len();
    let tail = text.get(position..)?.trim_start();
    let model = tail
        .split(char::is_whitespace)
        .next()
        .unwrap_or_default()
        .trim_end_matches(['.', ',', ';'])
        .trim();
    (!model.is_empty()).then(|| model.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use proptest::prelude::*;

    #[test]
    fn parses_required_human_formats() {
        assert_eq!(
            parse_retry_duration("Try again in 2h 30m"),
            Some(Duration::from_secs(9_000))
        );
        assert_eq!(
            parse_retry_duration("Try again in 17h 59m"),
            Some(Duration::from_secs(64_740))
        );
        assert_eq!(
            parse_retry_duration("Try again in 3h"),
            Some(Duration::from_secs(10_800))
        );
        assert_eq!(
            parse_retry_duration("Try again in 45m"),
            Some(Duration::from_secs(2_700))
        );
        assert_eq!(
            parse_retry_duration("Try again in 30s"),
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            parse_retry_duration("Retry after 30ms"),
            Some(Duration::from_millis(30))
        );
        assert_eq!(
            parse_retry_duration("TRY AGAIN IN 1D 3H 59M 9S"),
            Some(Duration::from_secs(100_749))
        );
    }

    #[test]
    fn malformed_inputs_are_safe() {
        for input in [
            "",
            "nonsense",
            "Try again",
            "Try again in -1h",
            "Try again in 999999999999h",
            "malformed JSON",
        ] {
            assert_eq!(parse_retry_duration(input), None, "{input}");
        }
    }

    #[test]
    fn retry_after_precedes_json_and_text() {
        let mut headers = HeaderMap::new();
        headers.insert("retry-after", HeaderValue::from_static("17"));
        let hint = retry_hint(
            &headers,
            br#"{"retry_after_seconds":99,"error":{"message":"Try again in 3h"}}"#,
            Duration::from_secs(5),
        );
        assert_eq!(hint.duration, Duration::from_secs(17));
        assert_eq!(hint.source, RetryHintSource::RetryAfter);
    }

    #[test]
    fn malformed_json_uses_human_or_fallback() {
        let hint = retry_hint(
            &HeaderMap::new(),
            b"not-json: Retry in 30m",
            Duration::from_secs(9),
        );
        assert_eq!(hint.duration, Duration::from_secs(1_800));
        let hint = retry_hint(&HeaderMap::new(), b"\xff\xfe", Duration::from_secs(9));
        assert_eq!(hint.duration, Duration::from_secs(9));
        assert_eq!(hint.source, RetryHintSource::Fallback);
    }

    proptest! {
        #[test]
        fn arbitrary_text_never_panics(input in any::<String>()) {
            let _ = parse_retry_duration(&input);
        }

        #[test]
        fn arbitrary_bytes_never_panic(input in proptest::collection::vec(any::<u8>(), 0..4096)) {
            let _ = retry_hint(&HeaderMap::new(), &input, Duration::from_secs(1));
        }
    }
}
