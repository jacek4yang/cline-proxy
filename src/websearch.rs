//! Anthropic WebSearch server-tool support over Cline `/search/websearch`.
//!
//! Claude Code does not execute `web_search` itself. When the model calls the
//! OpenAI function we expose, this gateway runs Cline's first-party search
//! endpoint and answers with `server_tool_use` + `web_search_tool_result`.
//! Protocol parsing lives here; HTTP execution lives on `ClineUpstream`.

use serde_json::{json, Map, Value};

use crate::anthropic::ProtocolError;

/// Anthropic function name reserved for this server tool.
pub const FUNCTION_NAME: &str = "web_search";

/// Implementation safety cap on searches per logical request.
pub const MAX_WEB_SEARCH_USES_PER_REQUEST: u32 = 5;

/// Defensive cap on internal GLM generations for one logical request
/// (initial round plus continuations). Prevents a malformed model response
/// from looping forever even when `max_uses` is small.
pub const MAX_INTERNAL_GENERATION_ROUNDS: u32 = 8;

/// Separate from the chat inactivity timeout; Cline's own agent runtime
/// uses a short search timeout.
pub const SEARCH_TIMEOUT_SECS: u64 = 20;

/// Hard bound on the Cline search HTTP body (bytes).
pub const MAX_SEARCH_RESPONSE_BYTES: usize = 256 * 1024;

/// Hard bound on grounding text appended to the upstream conversation.
pub const MAX_GROUNDING_BYTES: usize = 32 * 1024;

/// Hard bound on forwarded result objects.
pub const MAX_SEARCH_RESULTS: usize = 10;

/// Reject model queries longer than this without calling Cline.
pub const MAX_QUERY_CHARS: usize = 512;

/// Observed WebSearch server-tool protocol versions. A version is listed
/// only after its schema is known; similar names are never assumed compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebSearchProtocolVersion {
    /// Basic direct search. Fully supported.
    V20250305,
    /// Adds dynamic filtering via code execution. Direct-search only when
    /// the caller opts out of that path.
    V20260209,
    /// Adds `response_inclusion` for agentic/code-execution workflows.
    /// Direct-search only when that extra semantics is not requested.
    V20260318,
}

impl WebSearchProtocolVersion {
    pub fn type_str(self) -> &'static str {
        match self {
            Self::V20250305 => "web_search_20250305",
            Self::V20260209 => "web_search_20260209",
            Self::V20260318 => "web_search_20260318",
        }
    }
}

/// Domain restriction taken from the Anthropic declaration. The model cannot
/// weaken this via function arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DomainFilter {
    Allowed(Vec<String>),
    Blocked(Vec<String>),
}

/// Parsed WebSearch declaration. Execution state (uses remaining, outcomes)
/// is tracked separately by the server-tool loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebSearchConfig {
    pub version: WebSearchProtocolVersion,
    pub max_uses: u32,
    pub domains: Option<DomainFilter>,
}

/// Static Responses-frontend server-tool context held on `AppState`: the
/// per-request budget cap for the hosted `web_search` declaration plus the
/// optional domain restriction. The Anthropic frontend derives the same
/// values from each `web_search_*` declaration; the Responses frontend has
/// no encrypted declaration to parse, so these are the deployment-level
/// defaults (`max_uses` clamped to the same hard cap).
#[derive(Debug, Clone, Default)]
pub struct ResponsesWebSearchContext {
    pub max_uses: u32,
    pub domains: Option<DomainFilter>,
}

impl ResponsesWebSearchContext {
    /// Clamp a declared or configured `max_uses` to the implementation cap.
    pub fn clamped_uses(uses: u64) -> u32 {
        uses.clamp(1, u64::from(MAX_WEB_SEARCH_USES_PER_REQUEST)) as u32
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServerToolConfig {
    pub web_search: Option<WebSearchConfig>,
}

impl ServerToolConfig {
    pub fn web_search_enabled(&self) -> bool {
        self.web_search.is_some()
    }

    pub fn server_function_names(&self) -> std::collections::HashSet<String> {
        let mut names = std::collections::HashSet::new();
        if self.web_search.is_some() {
            names.insert(FUNCTION_NAME.to_owned());
        }
        names
    }
}

/// Anthropic-compatible search error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchErrorCode {
    TooManyRequests,
    InvalidToolInput,
    MaxUsesExceeded,
    QueryTooLong,
    RequestTooLarge,
    Unavailable,
}

impl SearchErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TooManyRequests => "too_many_requests",
            Self::InvalidToolInput => "invalid_tool_input",
            Self::MaxUsesExceeded => "max_uses_exceeded",
            Self::QueryTooLong => "query_too_long",
            Self::RequestTooLarge => "request_too_large",
            Self::Unavailable => "unavailable",
        }
    }
}

/// One executed search or a structured failure. Result objects are the
/// upstream entries verbatim (title + url required; extras preserved).
#[derive(Debug, Clone)]
pub struct SearchOutcome {
    pub query: String,
    pub results: Vec<Value>,
    pub error: Option<SearchErrorCode>,
}

impl SearchOutcome {
    pub fn failed(query: impl Into<String>, error: SearchErrorCode) -> Self {
        Self {
            query: query.into(),
            results: Vec::new(),
            error: Some(error),
        }
    }

    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolRoundKind {
    None,
    ClientOnly,
    ServerOnly,
    Mixed,
}

#[derive(Debug, Clone)]
pub struct OpenAiToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

pub fn classify_round(
    calls: &[OpenAiToolCall],
    server_names: &std::collections::HashSet<String>,
) -> ToolRoundKind {
    let has_server = calls.iter().any(|call| server_names.contains(&call.name));
    let has_client = calls.iter().any(|call| !server_names.contains(&call.name));
    match (has_server, has_client) {
        (false, false) => ToolRoundKind::None,
        (false, true) => ToolRoundKind::ClientOnly,
        (true, false) => ToolRoundKind::ServerOnly,
        (true, true) => ToolRoundKind::Mixed,
    }
}

/// True when the tool `type` is a `web_search_*` server-tool declaration,
/// including unknown versions that must not become client tools.
pub fn is_web_search_type(tool: &Value) -> bool {
    tool.get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind.starts_with("web_search_"))
}

/// Convert one Anthropic tools[] entry. `Ok(Some(function))` is the
/// deterministic upstream `web_search` function. `Ok(None)` means this is
/// not a web-search server tool (caller converts it as a client tool).
pub fn convert_declaration(tool: &Value) -> Result<Option<Value>, ProtocolError> {
    if !is_web_search_type(tool) {
        return Ok(None);
    }
    let _config = parse_declaration(tool)?;
    Ok(Some(upstream_function()))
}

/// Parse and validate a `web_search_*` declaration.
pub fn parse_declaration(tool: &Value) -> Result<WebSearchConfig, ProtocolError> {
    let object = tool
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("web_search tool must be an object"))?;
    let type_str = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| ProtocolError::invalid("web_search tool type is required"))?;
    let version = match type_str {
        "web_search_20250305" => WebSearchProtocolVersion::V20250305,
        "web_search_20260209" => WebSearchProtocolVersion::V20260209,
        "web_search_20260318" => WebSearchProtocolVersion::V20260318,
        other => {
            return Err(ProtocolError::invalid(format!(
                "web_search tool type {other:?} is not supported"
            )))
        }
    };
    let name = object.get("name").and_then(Value::as_str).unwrap_or("");
    if name != FUNCTION_NAME {
        return Err(ProtocolError::invalid(
            "web_search tool name must be \"web_search\"",
        ));
    }
    validate_supported_semantics(version, object)?;
    let max_uses = parse_max_uses(object)?;
    let domains = parse_domain_filter(object)?;
    Ok(WebSearchConfig {
        version,
        max_uses,
        domains,
    })
}

fn validate_supported_semantics(
    version: WebSearchProtocolVersion,
    object: &Map<String, Value>,
) -> Result<(), ProtocolError> {
    // Newer versions default `allowed_callers` to code execution (dynamic
    // filtering). This gateway can only run direct search.
    match version {
        WebSearchProtocolVersion::V20250305 => {
            if let Some(callers) = object.get("allowed_callers") {
                require_direct_only_callers(callers, version)?;
            }
        }
        WebSearchProtocolVersion::V20260209 | WebSearchProtocolVersion::V20260318 => {
            match object.get("allowed_callers") {
                None => {
                    return Err(ProtocolError::invalid(format!(
                        "{} defaults to code-execution dynamic filtering, which this gateway cannot provide; set allowed_callers to [\"direct\"] for basic web search",
                        version.type_str()
                    )))
                }
                Some(callers) => require_direct_only_callers(callers, version)?,
            }
        }
    }
    if version == WebSearchProtocolVersion::V20260318 {
        if let Some(inclusion) = object.get("response_inclusion").and_then(Value::as_str) {
            if inclusion != "full" {
                return Err(ProtocolError::invalid(
                    "web_search_20260318 response_inclusion other than \"full\" requires code-execution semantics this gateway cannot represent",
                ));
            }
        }
    } else if object.contains_key("response_inclusion") {
        return Err(ProtocolError::invalid(format!(
            "{} does not support response_inclusion",
            version.type_str()
        )));
    }
    if object.contains_key("user_location") {
        return Err(ProtocolError::invalid(
            "web_search user_location is not supported by this gateway",
        ));
    }
    Ok(())
}

fn require_direct_only_callers(
    callers: &Value,
    version: WebSearchProtocolVersion,
) -> Result<(), ProtocolError> {
    let array = callers.as_array().ok_or_else(|| {
        ProtocolError::invalid("web_search allowed_callers must be an array of strings")
    })?;
    if array.is_empty() {
        return Err(ProtocolError::invalid(
            "web_search allowed_callers must not be empty",
        ));
    }
    for caller in array {
        let Some(name) = caller.as_str() else {
            return Err(ProtocolError::invalid(
                "web_search allowed_callers entries must be strings",
            ));
        };
        if name != "direct" {
            return Err(ProtocolError::invalid(format!(
                "{} caller {name:?} requires code execution, which this gateway cannot provide; set allowed_callers to [\"direct\"]",
                version.type_str()
            )));
        }
    }
    Ok(())
}

fn parse_max_uses(object: &Map<String, Value>) -> Result<u32, ProtocolError> {
    match object.get("max_uses") {
        None => Ok(MAX_WEB_SEARCH_USES_PER_REQUEST),
        Some(Value::Number(number)) => {
            let Some(uses) = number.as_u64() else {
                return Err(ProtocolError::invalid(
                    "web_search max_uses must be a positive integer",
                ));
            };
            if uses == 0 {
                return Err(ProtocolError::invalid(
                    "web_search max_uses must be a positive integer",
                ));
            }
            Ok(uses.min(u64::from(MAX_WEB_SEARCH_USES_PER_REQUEST)) as u32)
        }
        Some(_) => Err(ProtocolError::invalid(
            "web_search max_uses must be a positive integer",
        )),
    }
}

fn parse_domain_filter(object: &Map<String, Value>) -> Result<Option<DomainFilter>, ProtocolError> {
    let allowed = parse_domain_list(object, "allowed_domains")?;
    let blocked = parse_domain_list(object, "blocked_domains")?;
    match (allowed, blocked) {
        (Some(_), Some(_)) => Err(ProtocolError::invalid(
            "web_search allowed_domains and blocked_domains are mutually exclusive",
        )),
        (Some(domains), None) => Ok(Some(DomainFilter::Allowed(domains))),
        (None, Some(domains)) => Ok(Some(DomainFilter::Blocked(domains))),
        (None, None) => Ok(None),
    }
}

fn parse_domain_list(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<Vec<String>>, ProtocolError> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    let array = value
        .as_array()
        .ok_or_else(|| ProtocolError::invalid(format!("web_search {field} must be an array")))?;
    let mut domains = Vec::new();
    for entry in array {
        let Some(domain) = entry.as_str() else {
            return Err(ProtocolError::invalid(format!(
                "web_search {field} entries must be strings"
            )));
        };
        let trimmed = domain.trim();
        if trimmed.is_empty() {
            continue;
        }
        domains.push(trimmed.to_owned());
    }
    if domains.is_empty() {
        Ok(None)
    } else {
        Ok(Some(domains))
    }
}

/// Deterministic, prefix-stable OpenAI function exposed to GLM.
pub fn upstream_function() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": FUNCTION_NAME,
            "description": "Search the public web for current information and return relevant pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }
        }
    })
}

/// Cline `/search/websearch` JSON body. Never includes both domain fields.
pub fn cline_search_body(query: &str, domains: Option<&DomainFilter>) -> Value {
    let mut body = Map::new();
    body.insert("query".into(), Value::String(query.to_owned()));
    match domains {
        Some(DomainFilter::Allowed(domains)) => {
            body.insert(
                "allowed_domains".into(),
                Value::Array(domains.iter().cloned().map(Value::String).collect()),
            );
        }
        Some(DomainFilter::Blocked(domains)) => {
            body.insert(
                "blocked_domains".into(),
                Value::Array(domains.iter().cloned().map(Value::String).collect()),
            );
        }
        None => {}
    }
    Value::Object(body)
}

/// Extract the model-supplied query. Malformed arguments become
/// `invalid_tool_input` rather than a network call.
pub fn query_from_arguments(arguments: &str) -> Result<String, SearchErrorCode> {
    let parsed: Value =
        serde_json::from_str(arguments).map_err(|_| SearchErrorCode::InvalidToolInput)?;
    let query = parsed
        .get("query")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|query| !query.is_empty())
        .ok_or(SearchErrorCode::InvalidToolInput)?;
    if query.chars().count() > MAX_QUERY_CHARS {
        return Err(SearchErrorCode::QueryTooLong);
    }
    Ok(query.to_owned())
}

/// Parse Cline's search envelope. Results live under `data.results`.
/// Entries without both title and URL are dropped, never fabricated.
/// An empty result array is a successful empty search.
pub fn parse_cline_results(doc: &Value, query: &str) -> SearchOutcome {
    let results_value = doc.pointer("/data/results").or_else(|| doc.get("results"));
    let mut results = Vec::new();
    if let Some(Value::Array(items)) = results_value {
        for item in items {
            if item.get("title").and_then(Value::as_str).is_none()
                || item.get("url").and_then(Value::as_str).is_none()
            {
                continue;
            }
            results.push(item.clone());
            if results.len() >= MAX_SEARCH_RESULTS {
                break;
            }
        }
    }
    SearchOutcome {
        query: query.to_owned(),
        results,
        error: None,
    }
}

/// Grounding text for an upstream `role=tool` message. Only actual search
/// output; nothing is invented. Truncated at [`MAX_GROUNDING_BYTES`].
pub fn tool_result_text(outcome: &SearchOutcome) -> String {
    if let Some(error) = outcome.error {
        return format!("Web search failed: {}", error.as_str());
    }
    if outcome.results.is_empty() {
        return "Web search returned no results.".into();
    }
    let mut out = format!("Web search results for \"{}\":\n", outcome.query);
    for (index, item) in outcome.results.iter().enumerate() {
        let title = item.get("title").and_then(Value::as_str).unwrap_or("");
        let url = item.get("url").and_then(Value::as_str).unwrap_or("");
        out.push_str(&format!("{}. {title}\n   {url}\n", index + 1));
        if let Some(snippet) = item.get("snippet").and_then(Value::as_str) {
            out.push_str(&format!("   {snippet}\n"));
        }
        if out.len() >= MAX_GROUNDING_BYTES {
            out.truncate(MAX_GROUNDING_BYTES);
            break;
        }
    }
    out
}

/// Grounding text reconstructed from a replayed `web_search_tool_result`.
pub fn replayed_result_text(result_block: &Value) -> String {
    match result_block.get("content") {
        Some(Value::String(error)) => format!("Web search failed: {error}"),
        Some(Value::Object(object))
            if object.get("type").and_then(Value::as_str)
                == Some("web_search_tool_result_error") =>
        {
            let code = object
                .get("error_code")
                .and_then(Value::as_str)
                .unwrap_or("unavailable");
            format!("Web search failed: {code}")
        }
        Some(Value::Array(items)) => {
            let mut out = String::from("Web search results:\n");
            for (index, item) in items.iter().enumerate() {
                let title = item.get("title").and_then(Value::as_str).unwrap_or("");
                let url = item.get("url").and_then(Value::as_str).unwrap_or("");
                out.push_str(&format!("{}. {title}\n   {url}\n", index + 1));
                if let Some(snippet) = item.get("snippet").and_then(Value::as_str) {
                    out.push_str(&format!("   {snippet}\n"));
                }
                if out.len() >= MAX_GROUNDING_BYTES {
                    out.truncate(MAX_GROUNDING_BYTES);
                    break;
                }
            }
            out
        }
        _ => "Web search returned no results.".into(),
    }
}

/// Anthropic `server_tool_use` + `web_search_tool_result` pair. Result items
/// preserve title/url plus any optional upstream fields. Errors use the
/// official `web_search_tool_result_error` object. Encrypted Anthropic
/// citation fields are never fabricated.
pub fn client_blocks(tool_use_id: &str, outcome: &SearchOutcome) -> (Value, Value) {
    let use_block = json!({
        "type": "server_tool_use",
        "id": tool_use_id,
        "name": FUNCTION_NAME,
        "input": {"query": outcome.query}
    });
    let result_block = if let Some(error) = outcome.error {
        json!({
            "type": "web_search_tool_result",
            "tool_use_id": tool_use_id,
            "content": {
                "type": "web_search_tool_result_error",
                "error_code": error.as_str()
            }
        })
    } else {
        let content: Vec<Value> = outcome
            .results
            .iter()
            .map(|item| {
                let mut entry = json!({
                    "type": "web_search_result",
                    "title": item.get("title").cloned().unwrap_or(Value::Null),
                    "url": item.get("url").cloned().unwrap_or(Value::Null),
                });
                for key in ["snippet", "page_age", "site", "favicon"] {
                    if let Some(value) = item.get(key) {
                        entry[key] = value.clone();
                    }
                }
                entry
            })
            .collect();
        json!({
            "type": "web_search_tool_result",
            "tool_use_id": tool_use_id,
            "content": content
        })
    };
    (use_block, result_block)
}

pub fn new_server_tool_id() -> String {
    format!("srvtoolu_{}", uuid::Uuid::new_v4().simple())
}

/// Append one assistant tool-call round plus tool results to the already
/// optimized OpenAI conversation.
pub fn append_server_round(
    chat_body: &mut Value,
    calls: &[OpenAiToolCall],
    outcomes: &[SearchOutcome],
) {
    let Some(messages) = chat_body.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    let tool_calls: Vec<Value> = calls
        .iter()
        .map(|call| {
            json!({
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments}
            })
        })
        .collect();
    messages.push(json!({
        "role": "assistant",
        "content": Value::Null,
        "tool_calls": tool_calls
    }));
    for (call, outcome) in calls.iter().zip(outcomes.iter()) {
        messages.push(json!({
            "role": "tool",
            "tool_call_id": call.id,
            "content": tool_result_text(outcome)
        }));
    }
}

pub fn extract_openai_tool_calls(body: &Value) -> Vec<OpenAiToolCall> {
    let Some(choice) = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
    else {
        return Vec::new();
    };
    let message = choice.get("message").or_else(|| choice.get("delta"));
    let Some(calls) = message
        .and_then(|message| message.get("tool_calls"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    calls
        .iter()
        .map(|call| {
            let function = call.get("function").and_then(Value::as_object);
            OpenAiToolCall {
                id: call
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("call_unknown")
                    .to_owned(),
                name: function
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_owned(),
                arguments: function
                    .and_then(|function| function.get("arguments"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned(),
            }
        })
        .collect()
}

/// Map a search-endpoint HTTP status (and timeout/transport class) to an
/// Anthropic error code. Chat-key cooldown is never implied by this mapping.
pub fn error_code_for_status(status: u16) -> SearchErrorCode {
    match status {
        429 => SearchErrorCode::TooManyRequests,
        413 => SearchErrorCode::RequestTooLarge,
        400 | 422 => SearchErrorCode::InvalidToolInput,
        _ => SearchErrorCode::Unavailable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declaration(extra: Value) -> Value {
        let mut tool = json!({"type": "web_search_20250305", "name": "web_search"});
        if let Some(object) = extra.as_object() {
            for (key, value) in object {
                tool[key] = value.clone();
            }
        }
        tool
    }

    #[test]
    fn parses_basic_declaration() {
        let config = parse_declaration(&declaration(json!({"max_uses": 3}))).unwrap();
        assert_eq!(config.version, WebSearchProtocolVersion::V20250305);
        assert_eq!(config.max_uses, 3);
        assert!(config.domains.is_none());
    }

    #[test]
    fn hard_caps_max_uses() {
        let config = parse_declaration(&declaration(json!({"max_uses": 99}))).unwrap();
        assert_eq!(config.max_uses, MAX_WEB_SEARCH_USES_PER_REQUEST);
    }

    #[test]
    fn rejects_zero_max_uses() {
        let error = parse_declaration(&declaration(json!({"max_uses": 0}))).unwrap_err();
        assert!(error.message.contains("max_uses"));
    }

    #[test]
    fn allowed_domains_are_kept_verbatim() {
        let config = parse_declaration(&declaration(json!({
            "allowed_domains": ["Rust-Lang.org", " example.com ", ""]
        })))
        .unwrap();
        assert_eq!(
            config.domains,
            Some(DomainFilter::Allowed(vec![
                "Rust-Lang.org".into(),
                "example.com".into()
            ]))
        );
    }

    #[test]
    fn blocked_domains_parse() {
        let config = parse_declaration(&declaration(json!({
            "blocked_domains": ["tracker.example"]
        })))
        .unwrap();
        assert_eq!(
            config.domains,
            Some(DomainFilter::Blocked(vec!["tracker.example".into()]))
        );
    }

    #[test]
    fn rejects_allowed_and_blocked_together() {
        let error = parse_declaration(&declaration(json!({
            "allowed_domains": ["a.com"],
            "blocked_domains": ["b.com"]
        })))
        .unwrap_err();
        assert!(error.message.contains("mutually exclusive"));
    }

    #[test]
    fn rejects_wrong_name() {
        let error = parse_declaration(&json!({
            "type": "web_search_20250305",
            "name": "WebSearch"
        }))
        .unwrap_err();
        assert!(error.message.contains("web_search"));
    }

    #[test]
    fn unknown_version_is_not_a_client_tool() {
        assert!(is_web_search_type(
            &json!({"type": "web_search_20990101", "name": "web_search"})
        ));
        let error = parse_declaration(&json!({
            "type": "web_search_20990101",
            "name": "web_search"
        }))
        .unwrap_err();
        assert!(error.message.contains("not supported"));
        assert!(convert_declaration(&json!({
            "type": "web_search_20990101",
            "name": "web_search"
        }))
        .is_err());
    }

    #[test]
    fn newer_version_requires_direct_callers() {
        let error = parse_declaration(&json!({
            "type": "web_search_20260209",
            "name": "web_search"
        }))
        .unwrap_err();
        assert!(error.message.contains("allowed_callers"));

        let ok = parse_declaration(&json!({
            "type": "web_search_20260209",
            "name": "web_search",
            "allowed_callers": ["direct"]
        }))
        .unwrap();
        assert_eq!(ok.version, WebSearchProtocolVersion::V20260209);

        let error = parse_declaration(&json!({
            "type": "web_search_20260209",
            "name": "web_search",
            "allowed_callers": ["code_execution_20260120"]
        }))
        .unwrap_err();
        assert!(error.message.contains("code execution"));
    }

    #[test]
    fn v20260318_rejects_excluded_inclusion() {
        let error = parse_declaration(&json!({
            "type": "web_search_20260318",
            "name": "web_search",
            "allowed_callers": ["direct"],
            "response_inclusion": "excluded"
        }))
        .unwrap_err();
        assert!(error.message.contains("response_inclusion"));

        let ok = parse_declaration(&json!({
            "type": "web_search_20260318",
            "name": "web_search",
            "allowed_callers": ["direct"]
        }))
        .unwrap();
        assert_eq!(ok.version, WebSearchProtocolVersion::V20260318);
    }

    #[test]
    fn upstream_function_is_deterministic() {
        let function = upstream_function();
        assert_eq!(function["type"], "function");
        assert_eq!(function["function"]["name"], FUNCTION_NAME);
        assert_eq!(
            function["function"]["parameters"]["required"],
            json!(["query"])
        );
        assert_eq!(
            function["function"]["parameters"]["additionalProperties"],
            json!(false)
        );
        let encoded = serde_json::to_string(&function).unwrap();
        assert_eq!(
            encoded,
            serde_json::to_string(&upstream_function()).unwrap()
        );
    }

    #[test]
    fn cline_body_never_sends_both_domain_fields() {
        let allowed = cline_search_body(
            "q",
            Some(&DomainFilter::Allowed(vec!["example.com".into()])),
        );
        assert!(allowed.get("blocked_domains").is_none());
        assert_eq!(allowed["allowed_domains"], json!(["example.com"]));
        let blocked = cline_search_body(
            "q",
            Some(&DomainFilter::Blocked(vec!["ads.example".into()])),
        );
        assert!(blocked.get("allowed_domains").is_none());
        assert_eq!(blocked["query"], "q");
    }

    #[test]
    fn parses_data_results_and_optional_fields() {
        let doc = json!({
            "data": {"results": [
                {"title": "A", "url": "https://a.example", "snippet": "s", "page_age": "1 day"},
                {"title": "missing url"},
                {"url": "https://no-title.example"}
            ]}
        });
        let outcome = parse_cline_results(&doc, "q");
        assert_eq!(outcome.results.len(), 1);
        assert_eq!(outcome.results[0]["snippet"], "s");
        assert!(outcome.error.is_none());
    }

    #[test]
    fn empty_results_are_success() {
        let outcome = parse_cline_results(&json!({"data": {"results": []}}), "q");
        assert!(outcome.results.is_empty());
        assert!(outcome.error.is_none());
        assert_eq!(
            tool_result_text(&outcome),
            "Web search returned no results."
        );
    }

    #[test]
    fn tool_result_text_preserves_url_and_title() {
        let outcome = SearchOutcome {
            query: "q".into(),
            error: None,
            results: vec![json!({
                "title": "Example & Title <with markup>",
                "url": "https://example.com/a?b=1&c=2",
                "snippet": "snippet text"
            })],
        };
        let text = tool_result_text(&outcome);
        assert!(text.contains("https://example.com/a?b=1&c=2"));
        assert!(text.contains("Example & Title <with markup>"));
        assert!(text.contains("snippet text"));
    }

    #[test]
    fn client_blocks_use_official_error_object() {
        let outcome = SearchOutcome::failed("q", SearchErrorCode::Unavailable);
        let (use_block, result) = client_blocks("srvtoolu_x", &outcome);
        assert_eq!(use_block["type"], "server_tool_use");
        assert_eq!(result["content"]["type"], "web_search_tool_result_error");
        assert_eq!(result["content"]["error_code"], "unavailable");
        assert!(!serde_json::to_string(&result)
            .unwrap()
            .contains("encrypted"));
    }

    #[test]
    fn malformed_arguments_map_to_invalid_tool_input() {
        assert_eq!(
            query_from_arguments("not-json"),
            Err(SearchErrorCode::InvalidToolInput)
        );
        assert_eq!(
            query_from_arguments("{}"),
            Err(SearchErrorCode::InvalidToolInput)
        );
        assert_eq!(query_from_arguments(r#"{"query":"ok"}"#).unwrap(), "ok");
    }

    #[test]
    fn query_too_long_is_rejected_locally() {
        let long = "x".repeat(MAX_QUERY_CHARS + 1);
        let arguments = json!({"query": long}).to_string();
        assert_eq!(
            query_from_arguments(&arguments),
            Err(SearchErrorCode::QueryTooLong)
        );
    }

    #[test]
    fn classify_round_kinds() {
        let names = [FUNCTION_NAME.to_owned()].into_iter().collect();
        let server = OpenAiToolCall {
            id: "1".into(),
            name: FUNCTION_NAME.into(),
            arguments: "{}".into(),
        };
        let client = OpenAiToolCall {
            id: "2".into(),
            name: "Read".into(),
            arguments: "{}".into(),
        };
        assert_eq!(classify_round(&[], &names), ToolRoundKind::None);
        assert_eq!(
            classify_round(std::slice::from_ref(&server), &names),
            ToolRoundKind::ServerOnly
        );
        assert_eq!(
            classify_round(std::slice::from_ref(&client), &names),
            ToolRoundKind::ClientOnly
        );
        assert_eq!(
            classify_round(&[server, client], &names),
            ToolRoundKind::Mixed
        );
    }

    #[test]
    fn status_mapping() {
        assert_eq!(error_code_for_status(429), SearchErrorCode::TooManyRequests);
        assert_eq!(error_code_for_status(503), SearchErrorCode::Unavailable);
        assert_eq!(error_code_for_status(413), SearchErrorCode::RequestTooLarge);
    }
}
