//! File-backed configuration and startup validation.

use std::collections::{BTreeMap, HashSet};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use axum::http::{HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};

use crate::glm53::reasoning::ThinkingExposure;

pub mod defaults {
    pub const BIND: &str = "127.0.0.1:8788";
    pub const BASE_URL: &str = "https://api.cline.bot/api/v1";
    pub const CHAT_PATH: &str = "/chat/completions";
    pub const TIMEOUT_SECS: u64 = 600;
    pub const CONNECT_TIMEOUT_SECS: u64 = 20;
    pub const FALLBACK_COOLDOWN_SECS: u64 = 3_600;
    pub const MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;
    pub const STREAM_PROGRESS_SECS: u64 = 30;
    pub const SHUTDOWN_TIMEOUT_SECS: u64 = 30;
    pub const STATE_FILE: &str = "runtime-state.json";
    /// Coalesce window for the debounced runtime-state writer. Short enough
    /// that a confirmed quota 429 reaches disk quickly, long enough to turn
    /// a multi-key 429 burst into one write.
    pub const STATE_DEBOUNCE_MS: u64 = 150;
    /// Default upstream output ceiling. 8K is tight for complex coding
    /// turns; 32K+ invites runaway generation (issue #6). Overridable.
    pub const MAX_OUTPUT_TOKENS: u32 = 16_384;
}

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub server: ServerConfig,
    pub upstream: UpstreamConfig,
    pub cline_api_keys: Vec<ClineKeyConfig>,
    pub models: ModelsConfig,
    pub runtime: RuntimeConfig,
    pub glm53: Glm53Config,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub bind: String,
    pub api_key: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: defaults::BIND.into(),
            api_key: String::new(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UpstreamConfig {
    pub base_url: String,
    pub chat_path: String,
    pub timeout_secs: u64,
    pub connect_timeout_secs: u64,
    pub fallback_429_cooldown_secs: u64,
    pub headers: BTreeMap<String, String>,
}

impl Default for UpstreamConfig {
    fn default() -> Self {
        Self {
            base_url: defaults::BASE_URL.into(),
            chat_path: defaults::CHAT_PATH.into(),
            timeout_secs: defaults::TIMEOUT_SECS,
            connect_timeout_secs: defaults::CONNECT_TIMEOUT_SECS,
            fallback_429_cooldown_secs: defaults::FALLBACK_COOLDOWN_SECS,
            headers: default_cline_headers(),
        }
    }
}

fn default_cline_headers() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("http-referer".into(), "https://cline.bot".into()),
        ("user-agent".into(), "Cline/4.1.16".into()),
        ("x-client-type".into(), "cline-vscode".into()),
        ("x-client-version".into(), "4.1.16".into()),
        ("x-core-version".into(), "4.1.16".into()),
        ("x-platform".into(), "vscode".into()),
        ("x-platform-version".into(), "1.106.0".into()),
        ("x-title".into(), "Cline".into()),
    ])
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClineKeyConfig {
    pub name: String,
    pub api_key: String,
    pub enabled: bool,
}

impl Default for ClineKeyConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            api_key: String::new(),
            enabled: true,
        }
    }
}

impl std::fmt::Debug for ClineKeyConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ClineKeyConfig")
            .field("name", &self.name)
            .field("api_key", &"[REDACTED]")
            .field("enabled", &self.enabled)
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelsConfig {
    pub default: String,
    pub aliases: BTreeMap<String, String>,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            default: "z-ai/glm-5.3-flash".into(),
            aliases: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    #[default]
    Pretty,
    Json,
}

/// GLM-5.3-Flash request policy (issue #6): bounded reasoning, zero
/// historical-thinking amplification, capped output. All fields documented
/// in docs/GLM53_FLASH.md; defaults are the production recommendations.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Glm53Config {
    pub reasoning: Glm53ReasoningConfig,
    pub limits: Glm53LimitsConfig,
    pub context: Glm53ContextConfig,
    pub telemetry: Glm53TelemetryConfig,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Glm53ReasoningConfig {
    /// Effort used when the request carries no explicit reasoning control.
    /// `high` keeps strong coding analysis without the `max` runaway the
    /// official template would otherwise select for unset efforts.
    pub default_effort: Glm53Effort,
    /// Effort used for `thinking: {type: "adaptive"}`.
    pub adaptive_effort: Glm53Effort,
    /// Strip `thinking`/`reasoning_content` from historical assistant
    /// messages (before the last user/tool-result turn) on the upstream
    /// wire. Text, tool_calls, ids, and order are never touched.
    pub strip_historical_thinking: bool,
    /// When upstream reasoning may be surfaced to the client.
    pub expose_thinking: ThinkingExposure,
}

impl Default for Glm53ReasoningConfig {
    fn default() -> Self {
        Self {
            default_effort: Glm53Effort::High,
            adaptive_effort: Glm53Effort::High,
            strip_historical_thinking: true,
            expose_thinking: ThinkingExposure::default(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Glm53LimitsConfig {
    /// Upstream output ceiling: `effective_max_tokens = min(client, this)`.
    pub max_output_tokens: u32,
}

impl Default for Glm53LimitsConfig {
    fn default() -> Self {
        Self {
            max_output_tokens: defaults::MAX_OUTPUT_TOKENS,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Glm53ContextConfig {
    /// Lossless structural normalization only: single-text-block content
    /// arrays become strings, empty text blocks are dropped, Anthropic-only
    /// `metadata` is not forwarded. Never truncates tool results or edits
    /// tool schemas.
    pub safe_compaction: bool,
    /// Remove a *leading* `x-anthropic-billing-header: ...` line from the
    /// system text. Its dynamic attribution metadata changes between
    /// requests and would break the system prefix (and upstream prompt
    /// cache locality) every turn. Start-anchored only: a header mentioned
    /// later in the text is never touched.
    pub strip_volatile_billing_header: bool,
    /// Deterministic key order for historical assistant tool-call argument
    /// JSON strings. Equivalent semantics then serialize to identical
    /// bytes, preserving upstream prefix-cache locality. Arrays keep their
    /// order; plain-text tool results are never parsed or rewritten.
    pub canonical_tool_json: bool,
}

impl Default for Glm53ContextConfig {
    fn default() -> Self {
        Self {
            safe_compaction: true,
            strip_volatile_billing_header: true,
            canonical_tool_json: true,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Glm53TelemetryConfig {
    /// Exact GLM token accounting per request via the embedded official
    /// tokenizer. Runs in a bounded `spawn_blocking` task after the upstream
    /// request is dispatched, so it never adds to TTFT and never occupies
    /// more than `max_concurrent_token_counts` CPU slots at once. When the
    /// slot is busy the count is skipped (logged), never queued.
    pub exact_input_tokens: bool,
    /// Concurrent CPU slots for exact token counts. 1 suits most hosts;
    /// raise on high-core machines, set `exact_input_tokens: false` (or 0)
    /// to disable.
    pub max_concurrent_token_counts: u32,
    /// Stable-prefix hash telemetry per request (`prefix_hash` +
    /// `prefix_bytes`): a local byte-stability metric, never a proof of an
    /// upstream cache hit. Also enables session fingerprint extraction.
    pub prefix_hash: bool,
    /// Cache metrics from upstream usage when provided: `cached_tokens`,
    /// `cache_hit_ratio`, `reasoning_ratio`. Ratios are only logged when
    /// the upstream reports the underlying token figures.
    pub cache_metrics: bool,
}

impl Default for Glm53TelemetryConfig {
    fn default() -> Self {
        Self {
            exact_input_tokens: true,
            max_concurrent_token_counts: 1,
            prefix_hash: true,
            cache_metrics: true,
        }
    }
}

/// Config-facing effort level. Reuses the GLM effort vocabulary; `max` is
/// valid as an explicit client-driven outcome but rejected as a proxy
/// default (startup validation), because a default of `max` recreates the
/// runaway this configuration exists to prevent.
pub type Glm53Effort = crate::glm53::reasoning::GlmReasoningEffort;

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    pub max_request_bytes: usize,
    pub log_level: String,
    pub log_format: LogFormat,
    pub stream_progress_secs: u64,
    pub shutdown_timeout_secs: u64,
    /// Path of the persisted key runtime state file (relative paths resolve
    /// against the working directory). `null` or an empty string disables
    /// persistence. The file never contains key material; see
    /// `docs/adr/0001-persistent-key-runtime-state-and-stickiness.md`.
    pub state_file: Option<String>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_request_bytes: defaults::MAX_REQUEST_BYTES,
            log_level: "info".into(),
            log_format: LogFormat::Pretty,
            stream_progress_secs: defaults::STREAM_PROGRESS_SECS,
            shutdown_timeout_secs: defaults::SHUTDOWN_TIMEOUT_SECS,
            state_file: Some(defaults::STATE_FILE.into()),
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading configuration {}", path.display()))?;
        let config: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing configuration {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.server
            .bind
            .parse::<SocketAddr>()
            .with_context(|| "server.bind must be an IP socket address such as 127.0.0.1:8788")?;
        if self.server.api_key.trim().is_empty() {
            bail!("server.api_key must not be empty");
        }
        HeaderValue::try_from(format!("Bearer {}", self.server.api_key))
            .context("server.api_key must be HTTP-header-safe")?;

        let url = reqwest::Url::parse(&self.upstream.base_url)
            .context("upstream.base_url must be a valid URL")?;
        if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
            bail!("upstream.base_url must be an absolute HTTP(S) URL");
        }
        if url.query().is_some() || url.fragment().is_some() {
            bail!("upstream.base_url must not contain a query or fragment");
        }
        if !url.username().is_empty() || url.password().is_some() {
            bail!("upstream.base_url must not contain credentials");
        }
        if !self.upstream.chat_path.starts_with('/')
            || self.upstream.chat_path.contains('?')
            || self.upstream.chat_path.contains('#')
        {
            bail!("upstream.chat_path must be an absolute path without query or fragment");
        }
        if self.upstream.timeout_secs == 0
            || self.upstream.connect_timeout_secs == 0
            || self.upstream.fallback_429_cooldown_secs == 0
        {
            bail!("upstream timeout and cooldown values must be greater than zero");
        }
        validate_headers(&self.upstream.headers)?;

        if self.cline_api_keys.is_empty() {
            bail!("at least one Cline API key must be configured");
        }
        let mut names = HashSet::new();
        for (index, key) in self.cline_api_keys.iter().enumerate() {
            if key.name.trim().is_empty() {
                bail!("cline_api_keys[{index}].name must not be empty");
            }
            if key.api_key.trim().is_empty() {
                bail!("Cline API key at index {index} must not be empty");
            }
            HeaderValue::try_from(format!("Bearer {}", key.api_key)).with_context(|| {
                format!("Cline API key at index {index} is not HTTP-header-safe")
            })?;
            if !names.insert(key.name.as_str()) {
                bail!("Cline API key names must be unique; duplicate name at index {index}");
            }
        }
        if !self.cline_api_keys.iter().any(|key| key.enabled) {
            bail!("at least one Cline API key must be enabled");
        }
        if self.models.default.trim().is_empty() {
            bail!("models.default must not be empty");
        }
        for (alias, model) in &self.models.aliases {
            if alias.trim().is_empty() || model.trim().is_empty() {
                bail!("model aliases and targets must not be empty");
            }
        }
        if self.runtime.max_request_bytes == 0 || self.runtime.shutdown_timeout_secs == 0 {
            bail!("runtime size and shutdown limits must be greater than zero");
        }
        tracing_subscriber::EnvFilter::try_new(&self.runtime.log_level)
            .context("runtime.log_level must be a valid tracing filter")?;

        if !(1024..=131_072).contains(&self.glm53.limits.max_output_tokens) {
            bail!(
                "glm53.limits.max_output_tokens must be between 1024 and 131072 (got {})",
                self.glm53.limits.max_output_tokens
            );
        }
        for (field, effort) in [
            (
                "glm53.reasoning.default_effort",
                self.glm53.reasoning.default_effort,
            ),
            (
                "glm53.reasoning.adaptive_effort",
                self.glm53.reasoning.adaptive_effort,
            ),
        ] {
            if effort == crate::glm53::reasoning::GlmReasoningEffort::Max {
                bail!("{field} must not be `max`; explicit max stays available via output_config.effort");
            }
        }
        Ok(())
    }

    pub fn resolve_model(&self, requested: &str) -> String {
        self.models
            .aliases
            .get(requested)
            .cloned()
            .unwrap_or_else(|| requested.to_owned())
    }

    pub fn model_ids(&self) -> Vec<String> {
        let mut models = Vec::new();
        models.push(self.models.default.clone());
        models.extend(self.models.aliases.keys().cloned());
        models.extend(self.models.aliases.values().cloned());
        models.sort();
        models.dedup();
        models
    }

    /// Resolved runtime-state path, or `None` when persistence is disabled.
    pub fn state_file_path(&self) -> Option<PathBuf> {
        self.runtime
            .state_file
            .as_ref()
            .map(|path| path.trim())
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
    }
}

fn validate_headers(headers: &BTreeMap<String, String>) -> Result<()> {
    const RESERVED: &[&str] = &[
        "authorization",
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "accept",
        "content-type",
        "accept-encoding",
        "cookie",
        "x-api-key",
    ];
    for (name, value) in headers {
        let parsed_name = HeaderName::try_from(name.as_str())
            .with_context(|| format!("invalid upstream header name {name:?}"))?;
        HeaderValue::try_from(value.as_str())
            .with_context(|| format!("invalid value for upstream header {name:?}"))?;
        if RESERVED
            .iter()
            .any(|reserved| parsed_name.as_str().eq_ignore_ascii_case(reserved))
        {
            bail!("upstream header {name:?} is controlled by the gateway");
        }
    }
    Ok(())
}

pub fn default_config_path() -> PathBuf {
    std::env::var_os("CLINE_PROXY_CONFIG")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> Config {
        let mut config = Config::default();
        config.server.api_key = "gateway-secret".into();
        config.cline_api_keys = vec![ClineKeyConfig {
            name: "one".into(),
            api_key: "cline-secret".into(),
            enabled: true,
        }];
        config
    }

    #[test]
    fn zero_or_disabled_keys_are_rejected() {
        let mut config = valid_config();
        config.cline_api_keys.clear();
        assert!(config.validate().is_err());
        config.cline_api_keys.push(ClineKeyConfig {
            name: "one".into(),
            api_key: "secret".into(),
            enabled: false,
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn empty_keys_and_malformed_headers_are_rejected_without_secret_text() {
        let mut config = valid_config();
        config.cline_api_keys[0].api_key.clear();
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("index 0"));
        assert!(!error.contains("gateway-secret"));

        let mut config = valid_config();
        config
            .upstream
            .headers
            .insert("bad header".into(), "value".into());
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config
            .upstream
            .headers
            .insert("Authorization".into(), "secret".into());
        assert!(config.validate().is_err());
    }

    #[test]
    fn unusable_gateway_keys_and_url_credentials_are_rejected() {
        let mut config = valid_config();
        config.server.api_key = "not\nheader-safe".into();
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.upstream.base_url = "https://user:password@api.cline.bot/api/v1".into();
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("must not contain credentials"));
        assert!(!error.contains("password"));
    }

    #[test]
    fn disabled_entries_are_valid_when_another_key_is_enabled() {
        let mut config = valid_config();
        config.cline_api_keys.push(ClineKeyConfig {
            name: "two".into(),
            api_key: "unused-secret".into(),
            enabled: false,
        });
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_debug_never_exposes_key_material() {
        let key = ClineKeyConfig {
            name: "one".into(),
            api_key: "top-secret".into(),
            enabled: true,
        };
        let rendered = format!("{key:?}");
        assert!(!rendered.contains("top-secret"));
        assert!(rendered.contains("REDACTED"));
    }

    #[test]
    fn derived_config_default_preserves_runtime_defaults() {
        let config = Config::default();
        assert_eq!(config.server.bind, defaults::BIND);
        assert!(config.server.api_key.is_empty());
        assert_eq!(config.upstream.base_url, defaults::BASE_URL);
        assert_eq!(config.upstream.chat_path, defaults::CHAT_PATH);
        assert_eq!(config.upstream.timeout_secs, defaults::TIMEOUT_SECS);
        assert_eq!(
            config.upstream.connect_timeout_secs,
            defaults::CONNECT_TIMEOUT_SECS
        );
        assert_eq!(
            config.upstream.fallback_429_cooldown_secs,
            defaults::FALLBACK_COOLDOWN_SECS
        );
        assert_eq!(config.upstream.headers, default_cline_headers());
        assert_eq!(config.models.default, "z-ai/glm-5.3-flash");
        assert!(config.models.aliases.is_empty());
        assert_eq!(
            config.runtime.max_request_bytes,
            defaults::MAX_REQUEST_BYTES
        );
        assert_eq!(config.runtime.log_level, "info");
        assert!(config.runtime.log_format == LogFormat::Pretty);
        assert_eq!(
            config.runtime.stream_progress_secs,
            defaults::STREAM_PROGRESS_SECS
        );
        assert_eq!(
            config.runtime.shutdown_timeout_secs,
            defaults::SHUTDOWN_TIMEOUT_SECS
        );
        assert_eq!(
            config.runtime.state_file.as_deref(),
            Some(defaults::STATE_FILE)
        );
        assert!(config.cline_api_keys.is_empty());
    }

    #[test]
    fn example_configuration_loads_and_validates() {
        let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/config.example.json"));
        let config = Config::load(path).expect("example configuration must remain valid");
        assert_eq!(config.cline_api_keys.len(), 2);
        assert_eq!(config.upstream.headers, default_cline_headers());
        assert_eq!(
            config.glm53.reasoning.default_effort,
            crate::glm53::reasoning::GlmReasoningEffort::High
        );
        assert_eq!(
            config.glm53.limits.max_output_tokens,
            defaults::MAX_OUTPUT_TOKENS
        );
        assert!(config.glm53.reasoning.strip_historical_thinking);
        assert!(config.glm53.context.safe_compaction);
        assert!(config.glm53.context.strip_volatile_billing_header);
        assert!(config.glm53.context.canonical_tool_json);
        assert!(config.glm53.telemetry.prefix_hash);
        assert!(config.glm53.telemetry.cache_metrics);
    }

    #[test]
    fn glm53_policy_defaults_are_production_safe() {
        let config = Config::default();
        assert_eq!(
            config.glm53.reasoning.default_effort,
            crate::glm53::reasoning::GlmReasoningEffort::High
        );
        assert_eq!(
            config.glm53.reasoning.expose_thinking,
            crate::glm53::reasoning::ThinkingExposure::RequestedOnly
        );
        assert_eq!(
            config.glm53.limits.max_output_tokens,
            defaults::MAX_OUTPUT_TOKENS
        );
        assert!(config.glm53.reasoning.strip_historical_thinking);
        assert!(config.glm53.telemetry.exact_input_tokens);
    }

    #[test]
    fn glm53_policy_rejects_default_max_and_out_of_range_caps() {
        let mut config = valid_config();
        config.glm53.reasoning.default_effort = crate::glm53::reasoning::GlmReasoningEffort::Max;
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("default_effort"), "{error}");

        let mut config = valid_config();
        config.glm53.limits.max_output_tokens = 100;
        assert!(config.validate().is_err());
        let mut config = valid_config();
        config.glm53.limits.max_output_tokens = 1_000_000;
        assert!(config.validate().is_err());
    }

    #[test]
    fn glm53_policy_rejects_unknown_fields() {
        let config = valid_config();
        let mut json = serde_json::to_value(&config).unwrap();
        json["glm53"]["reasoning"]["strip_history"] = serde_json::Value::Bool(true);
        assert!(serde_json::from_value::<Config>(json).is_err());
    }
}
