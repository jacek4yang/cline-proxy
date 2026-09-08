//! Versioned persisted key runtime state.
//!
//! The state file is advisory operational cache: it survives restarts so the
//! gateway does not re-probe keys with known quota cooldowns. It never
//! contains key material, authorization values, or raw upstream error text.
//! Corrupt or incompatible files are non-fatal and start from empty state.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::rate_limit::RateLimitKind;

pub const STATE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PersistedState {
    pub version: u32,
    pub updated_at_unix_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_key: Option<String>,
    #[serde(default)]
    pub keys: BTreeMap<String, PersistedKeyState>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct PersistedKeyState {
    /// Unix wall-clock deadline in milliseconds. Never an `Instant`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooldown_until_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limit_kind: Option<RateLimitKind>,
    /// Model the confirmed 429 was scoped to (already sanitized upstream).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_429_at_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_success_at_unix_ms: Option<u64>,
}

#[derive(Debug)]
pub enum StateLoadOutcome {
    Missing,
    Loaded(PersistedState),
    /// Present but unreadable, wrong schema version, or structurally invalid.
    Corrupt(String),
}

/// Load persisted state. Missing and corrupt files are reported, never fatal.
pub fn load(path: &Path) -> StateLoadOutcome {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return StateLoadOutcome::Missing;
        }
        Err(error) => {
            return StateLoadOutcome::Corrupt(format!(
                "could not read runtime state file: {error}"
            ));
        }
    };
    let state: PersistedState = match serde_json::from_slice(&bytes) {
        Ok(state) => state,
        Err(error) => return StateLoadOutcome::Corrupt(format!("invalid JSON: {error}")),
    };
    if state.version != STATE_SCHEMA_VERSION {
        return StateLoadOutcome::Corrupt(format!(
            "unsupported schema version {} (expected {STATE_SCHEMA_VERSION})",
            state.version
        ));
    }
    if state.updated_at_unix_ms == 0 {
        return StateLoadOutcome::Corrupt("updated_at_unix_ms must be positive".into());
    }
    for (name, key) in &state.keys {
        if name.trim().is_empty() {
            return StateLoadOutcome::Corrupt("key names must not be empty".into());
        }
        if let Some(model) = &key.model {
            if model.len() > 256 {
                return StateLoadOutcome::Corrupt(format!(
                    "persisted model name for key {name:?} is implausibly long"
                ));
            }
        }
    }
    StateLoadOutcome::Loaded(state)
}

/// Atomically persist state: serialize to a temp sibling, then rename over
/// the target. `std::fs::rename` replaces existing files on both Unix and
/// Windows. The temp file is removed on serialization failure.
pub fn store(path: &Path, state: &PersistedState) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating state directory {}", parent.display()))?;
        }
    }
    let temp_path: PathBuf = path.with_extension("json.tmp");
    let serialized = serde_json::to_vec_pretty(state).context("serializing runtime state")?;
    std::fs::write(&temp_path, &serialized)
        .with_context(|| format!("writing {}", temp_path.display()))?;
    std::fs::rename(&temp_path, path)
        .with_context(|| format!("replacing {} atomically", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("cline-proxy-state-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn store_is_atomic_and_load_round_trips() {
        let dir = temp_dir("roundtrip");
        let path = dir.join("runtime-state.json");
        let state = PersistedState {
            version: STATE_SCHEMA_VERSION,
            updated_at_unix_ms: 1_788_842_155_000,
            active_key: Some("brewlogic".into()),
            keys: BTreeMap::from([(
                "google".into(),
                PersistedKeyState {
                    cooldown_until_unix_ms: Some(1_788_873_167_000),
                    rate_limit_kind: Some(RateLimitKind::DailyQuota),
                    model: Some("z-ai/glm-5.3-flash".into()),
                    last_429_at_unix_ms: Some(1_788_842_147_935),
                    last_success_at_unix_ms: None,
                },
            )]),
        };
        store(&path, &state).unwrap();
        assert!(!path.with_extension("json.tmp").exists());
        match load(&path) {
            StateLoadOutcome::Loaded(loaded) => assert_eq!(loaded, state),
            other => panic!("expected loaded state, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn missing_file_reports_missing() {
        assert!(matches!(
            load(Path::new("/nonexistent/cline-proxy/state.json")),
            StateLoadOutcome::Missing
        ));
    }

    #[test]
    fn corrupt_files_are_reported_not_fatal() {
        let dir = temp_dir("corrupt");
        let path = dir.join("runtime-state.json");
        std::fs::write(&path, b"{truncated json").unwrap();
        assert!(matches!(load(&path), StateLoadOutcome::Corrupt(_)));
        std::fs::write(&path, br#"{"version":99,"updated_at_unix_ms":1}"#).unwrap();
        assert!(matches!(load(&path), StateLoadOutcome::Corrupt(_)));
        std::fs::write(&path, br#"{"version":1,"updated_at_unix_ms":0}"#).unwrap();
        assert!(matches!(load(&path), StateLoadOutcome::Corrupt(_)));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unknown_future_fields_are_tolerated() {
        let dir = temp_dir("forward");
        let path = dir.join("runtime-state.json");
        std::fs::write(
            &path,
            json!({
                "version": 1,
                "updated_at_unix_ms": 42,
                "future_field": {"whatever": true},
                "keys": {"a": {"cooldown_until_unix_ms": 99, "future": [1, 2]}}
            })
            .to_string(),
        )
        .unwrap();
        match load(&path) {
            StateLoadOutcome::Loaded(state) => {
                assert_eq!(state.keys["a"].cooldown_until_unix_ms, Some(99));
            }
            other => panic!("forward-compatible load failed: {other:?}"),
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
