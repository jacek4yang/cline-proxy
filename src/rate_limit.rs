//! Effective upstream HTTP 429 classification and safe retry-hint extraction.

use std::time::{Duration, SystemTime};

use axum::http::{HeaderMap, StatusCode};
use serde_json::Value;

const MAX_PARSED_COOLDOWN: Duration = Duration::from_secs(366 * 24 * 60 * 60);

/// Sub-classification applied only AFTER an effective upstream HTTP 429 has
/// already been confirmed. Kind names must never be used as failover
/// triggers; they only describe how the confirmed 429 should be treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RateLimitKind {
    /// Daily (or otherwise long-lived) free-quota exhaustion. Strong retry
    /// deadline, persistent until the quota window resets.
    DailyQuota,
    /// Short-lived request-rate limiting (e.g. "too many requests, retry in
    /// 10s"). Expected to clear quickly.
    Transient,
    /// Confirmed 429 that cannot be confidently sub-classified.
    Unknown,
}

impl RateLimitKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DailyQuota => "daily_quota",
            Self::Transient => "transient",
            Self::Unknown => "unknown",
        }
    }
}

/// Classify a confirmed effective 429 into a rate-limit kind. Classification
/// is deliberately conservative: only the explicit daily-free-limit wording
/// used by Cline quota errors yields DailyQuota, short retry windows yield
/// Transient, and everything else stays Unknown rather than guessed.
pub fn classify_rate_limit_kind(message: Option<&str>, duration: Duration) -> RateLimitKind {
    const TRANSIENT_WINDOW: Duration = Duration::from_secs(60);
    if let Some(message) = message {
        let lower = message.to_ascii_lowercase();
        if lower.contains("daily free limit")
            || (lower.contains("daily") && lower.contains("limit"))
        {
            return RateLimitKind::DailyQuota;
        }
    }
    if duration <= TRANSIENT_WINDOW {
        return RateLimitKind::Transient;
    }
    RateLimitKind::Unknown
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpstreamDisposition {
    Success,
    DirectRateLimited429,
    ProxyWrappedRateLimited429,
    NonFailoverHttpError,
}

impl UpstreamDisposition {
    pub fn error_class(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::DirectRateLimited429 => "direct_rate_limit",
            Self::ProxyWrappedRateLimited429 => "proxy_wrapped_rate_limit",
            Self::NonFailoverHttpError => "non_failover_http_error",
        }
    }

    pub fn is_rate_limited(self) -> bool {
        matches!(
            self,
            Self::DirectRateLimited429 | Self::ProxyWrappedRateLimited429
        )
    }
}

#[derive(Debug, Clone)]
pub struct ClassifiedUpstreamResponse {
    pub outer_status: StatusCode,
    pub effective_status: StatusCode,
    pub disposition: UpstreamDisposition,
    pub retry_hint: Option<RetryHint>,
}

/// Classify one concrete HTTP response from the configured upstream. Transport
/// errors never enter this function. Text wrappers are deliberately restricted
/// to outer 5xx responses and require both a known wrapper phrase and explicit
/// rate-limit semantics.
pub fn classify_upstream_response(
    outer_status: StatusCode,
    headers: &HeaderMap,
    body: &[u8],
    fallback: Duration,
) -> ClassifiedUpstreamResponse {
    let disposition = if outer_status.is_success() {
        UpstreamDisposition::Success
    } else if outer_status == StatusCode::TOO_MANY_REQUESTS {
        UpstreamDisposition::DirectRateLimited429
    } else if outer_status.is_server_error() && is_proxy_wrapped_429(body) {
        UpstreamDisposition::ProxyWrappedRateLimited429
    } else {
        UpstreamDisposition::NonFailoverHttpError
    };
    let effective_status = if disposition.is_rate_limited() {
        StatusCode::TOO_MANY_REQUESTS
    } else {
        outer_status
    };
    let retry_hint = disposition
        .is_rate_limited()
        .then(|| retry_hint(headers, body, fallback));
    ClassifiedUpstreamResponse {
        outer_status,
        effective_status,
        disposition,
        retry_hint,
    }
}

fn is_proxy_wrapped_429(body: &[u8]) -> bool {
    if serde_json::from_slice::<Value>(body)
        .ok()
        .as_ref()
        .is_some_and(structured_upstream_429)
    {
        return true;
    }
    let text = String::from_utf8_lossy(body).to_ascii_lowercase();
    wrapper_reports_status(&text, 429) && has_rate_limit_semantics(&text)
}

fn structured_upstream_429(value: &Value) -> bool {
    structured_status(value, false)
}

fn structured_status(value: &Value, inside_error: bool) -> bool {
    match value {
        Value::Object(object) => object.iter().any(|(name, value)| {
            let normalized = name.to_ascii_lowercase().replace(['-', '_'], "");
            let explicit_status = matches!(
                normalized.as_str(),
                "upstreamstatus" | "statuscode" | "httpstatus"
            );
            if (explicit_status || inside_error && normalized == "status")
                && value_is_status(value, 429)
            {
                return true;
            }
            let child_inside_error = inside_error || normalized == "error";
            structured_status(value, child_inside_error)
        }),
        Value::Array(values) => values
            .iter()
            .any(|value| structured_status(value, inside_error)),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => false,
    }
}

fn value_is_status(value: &Value, expected: u64) -> bool {
    value.as_u64() == Some(expected)
        || value
            .as_str()
            .and_then(|value| value.trim().parse::<u64>().ok())
            == Some(expected)
}

fn wrapper_reports_status(text: &str, expected: u16) -> bool {
    const WRAPPER_STEMS: &[&str] = &[
        "upstream returned",
        "upstream response status",
        "upstream status",
        "upstream error",
    ];
    WRAPPER_STEMS.iter().any(|stem| {
        let mut remaining = text;
        while let Some(position) = remaining.find(stem) {
            let tail = &remaining[position + stem.len()..];
            if leading_status(tail) == Some(expected) {
                return true;
            }
            remaining = tail;
        }
        false
    })
}

fn leading_status(input: &str) -> Option<u16> {
    let mut input = input.trim_start_matches(|character: char| {
        character.is_ascii_whitespace() || matches!(character, ':' | '=' | '-')
    });
    if input
        .get(..4)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http"))
        && input.as_bytes().get(4).is_none_or(u8::is_ascii_whitespace)
    {
        input = input[4..].trim_start_matches(|character: char| {
            character.is_ascii_whitespace() || matches!(character, ':' | '=' | '-')
        });
    }
    let digits = input
        .as_bytes()
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digits == 0 {
        return None;
    }
    input.get(..digits)?.parse().ok()
}

fn has_rate_limit_semantics(text: &str) -> bool {
    [
        "daily free limit reached",
        "rate limit",
        "rate-limit",
        "rate limited",
        "rate-limited",
        "quota exceeded",
        "too many requests",
        "try again in",
        "retry after",
        "retry-after",
    ]
    .iter()
    .any(|phrase| text.contains(phrase))
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
    fn rate_limit_kind_classification_is_conservative() {
        assert_eq!(
            classify_rate_limit_kind(
                Some("Daily free limit reached on model z-ai/glm-5.3-flash. Try again in 8h 37m"),
                Duration::from_secs(31_020),
            ),
            RateLimitKind::DailyQuota
        );
        assert_eq!(
            classify_rate_limit_kind(Some("daily limit exceeded"), Duration::from_secs(60)),
            RateLimitKind::DailyQuota
        );
        assert_eq!(
            classify_rate_limit_kind(Some("too many requests"), Duration::from_secs(10)),
            RateLimitKind::Transient
        );
        assert_eq!(
            classify_rate_limit_kind(None, Duration::from_secs(5)),
            RateLimitKind::Transient
        );
        // Long cooldown without explicit daily wording must not be guessed.
        assert_eq!(
            classify_rate_limit_kind(Some("quota exhausted"), Duration::from_secs(83_820)),
            RateLimitKind::Unknown
        );
        assert_eq!(
            classify_rate_limit_kind(None, Duration::from_secs(83_820)),
            RateLimitKind::Unknown
        );
        // Kind words alone are only consulted after a confirmed 429; the
        // classifier itself never returns a disposition.
    }

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
        assert_eq!(
            parse_retry_duration("Try again in 23h 17m"),
            Some(Duration::from_secs(83_820))
        );
        assert_eq!(
            parse_retry_duration("Try again in 1d 2h 3m"),
            Some(Duration::from_secs(93_780))
        );
        assert_eq!(
            parse_retry_duration("Retry after 500ms"),
            Some(Duration::from_millis(500))
        );
    }

    #[test]
    fn exact_proxy_wrapped_429_is_classified_and_parsed() {
        let body = b"upstream returned 429: Error 429: Daily free limit reached on model z-ai/glm-5.3-flash. Try again in 23h 17m";
        let classified = classify_upstream_response(
            StatusCode::BAD_GATEWAY,
            &HeaderMap::new(),
            body,
            Duration::from_secs(60),
        );
        assert_eq!(classified.outer_status, StatusCode::BAD_GATEWAY);
        assert_eq!(classified.effective_status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            classified.disposition,
            UpstreamDisposition::ProxyWrappedRateLimited429
        );
        let hint = classified.retry_hint.expect("wrapped 429 needs a hint");
        assert_eq!(hint.duration, Duration::from_secs(83_820));
        assert_eq!(hint.source, RetryHintSource::HumanText);
        assert_eq!(hint.model.as_deref(), Some("z-ai/glm-5.3-flash"));
    }

    #[test]
    fn proxy_wrapped_429_uses_outer_retry_after_first() {
        let mut headers = HeaderMap::new();
        headers.insert("retry-after", HeaderValue::from_static("17"));
        let classified = classify_upstream_response(
            StatusCode::BAD_GATEWAY,
            &headers,
            b"upstream returned 429: rate limited; Try again in 23h 17m",
            Duration::from_secs(60),
        );
        let hint = classified.retry_hint.expect("wrapped 429 needs a hint");
        assert_eq!(hint.duration, Duration::from_secs(17));
        assert_eq!(hint.source, RetryHintSource::RetryAfter);
    }

    #[test]
    fn known_text_wrappers_require_rate_limit_semantics() {
        for body in [
            "upstream returned 429: rate limited",
            "upstream returned HTTP 429: Too Many Requests",
            "upstream status 429: quota exceeded",
            "upstream response status: 429; Retry-After 30s",
            "upstream error: 429; Try again in 3h",
        ] {
            let classified = classify_upstream_response(
                StatusCode::BAD_GATEWAY,
                &HeaderMap::new(),
                body.as_bytes(),
                Duration::from_secs(60),
            );
            assert_eq!(
                classified.disposition,
                UpstreamDisposition::ProxyWrappedRateLimited429,
                "{body}"
            );
        }

        let weak = classify_upstream_response(
            StatusCode::BAD_GATEWAY,
            &HeaderMap::new(),
            b"upstream returned 429: proxy operation failed",
            Duration::from_secs(60),
        );
        assert_eq!(weak.disposition, UpstreamDisposition::NonFailoverHttpError);
    }

    #[test]
    fn structured_proxy_status_fields_are_explicit() {
        for body in [
            json_bytes(serde_json::json!({"error":{"upstream_status":429}})),
            json_bytes(serde_json::json!({"error":{"upstreamStatus":"429"}})),
            json_bytes(serde_json::json!({"error":{"status_code":429}})),
            json_bytes(serde_json::json!({"error":{"statusCode":429}})),
            json_bytes(serde_json::json!({"error":{"http_status":429}})),
            json_bytes(serde_json::json!({"error":{"httpStatus":429}})),
            json_bytes(serde_json::json!({"status":502,"error":{"status":429}})),
        ] {
            let classified = classify_upstream_response(
                StatusCode::BAD_GATEWAY,
                &HeaderMap::new(),
                &body,
                Duration::from_secs(60),
            );
            assert_eq!(
                classified.disposition,
                UpstreamDisposition::ProxyWrappedRateLimited429
            );
        }

        for body in [
            json_bytes(serde_json::json!({"error":{"operation_id":429}})),
            json_bytes(serde_json::json!({"details":{"status":429}})),
            json_bytes(serde_json::json!({"status":429})),
        ] {
            let classified = classify_upstream_response(
                StatusCode::BAD_GATEWAY,
                &HeaderMap::new(),
                &body,
                Duration::from_secs(60),
            );
            assert_eq!(
                classified.disposition,
                UpstreamDisposition::NonFailoverHttpError
            );
        }
    }

    #[test]
    fn arbitrary_429_text_and_success_output_are_not_proxy_wrappers() {
        for (status, body) in [
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal operation 429 failed",
            ),
            (StatusCode::BAD_GATEWAY, "proxy request id 429123 failed"),
            (StatusCode::INTERNAL_SERVER_ERROR, "error 429"),
            (StatusCode::BAD_GATEWAY, "bad gateway"),
            (
                StatusCode::OK,
                "upstream returned 429: rate limited; model output only",
            ),
            (
                StatusCode::BAD_REQUEST,
                "upstream returned 429: rate limited; user input",
            ),
        ] {
            let classified = classify_upstream_response(
                status,
                &HeaderMap::new(),
                body.as_bytes(),
                Duration::from_secs(60),
            );
            let expected = if status.is_success() {
                UpstreamDisposition::Success
            } else {
                UpstreamDisposition::NonFailoverHttpError
            };
            assert_eq!(classified.disposition, expected, "{body}");
        }
    }

    #[test]
    fn direct_429_needs_no_body_semantics() {
        let classified = classify_upstream_response(
            StatusCode::TOO_MANY_REQUESTS,
            &HeaderMap::new(),
            b"arbitrary body",
            Duration::from_secs(60),
        );
        assert_eq!(
            classified.disposition,
            UpstreamDisposition::DirectRateLimited429
        );
        assert_eq!(classified.effective_status, StatusCode::TOO_MANY_REQUESTS);
    }

    fn json_bytes(value: Value) -> Vec<u8> {
        serde_json::to_vec(&value).expect("test JSON must serialize")
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
            let _ = classify_upstream_response(
                StatusCode::BAD_GATEWAY,
                &HeaderMap::new(),
                input.as_bytes(),
                Duration::from_secs(1),
            );
        }

        #[test]
        fn arbitrary_bytes_never_panic(input in proptest::collection::vec(any::<u8>(), 0..4096)) {
            let _ = retry_hint(&HeaderMap::new(), &input, Duration::from_secs(1));
            let _ = classify_upstream_response(
                StatusCode::BAD_GATEWAY,
                &HeaderMap::new(),
                &input,
                Duration::from_secs(1),
            );
        }
    }
}
