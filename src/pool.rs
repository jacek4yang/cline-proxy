//! Concurrency-safe sticky sequential Cline API-key selection.
//!
//! Routing invariants:
//! - The active key is used for every request until it confirms an effective
//!   HTTP 429. Success, 5xx, timeouts, resets, and client cancels never
//!   rotate it.
//! - An old key whose cooldown expires becomes HalfOpen (probe-eligible) but
//!   never steals active back from a healthy key.
//! - HalfOpen probing is single-flight: at most one in-flight probe per key.
//! - Cooldown state is restorable from wall-clock persisted state
//!   (`crate::state`) using the configured key name as identity.

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::Notify;

use crate::config::ClineKeyConfig;
use crate::rate_limit::RateLimitKind;
use crate::state::{PersistedKeyState, PersistedState, STATE_SCHEMA_VERSION};

const MAX_COOLDOWN: Duration = Duration::from_secs(366 * 24 * 60 * 60);

/// Why the sticky active key changed. Normal successful requests are never a
/// switch reason; if a switch is ever logged without one of these, routing
/// is broken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySwitchReason {
    /// The previous active key confirmed an effective HTTP 429.
    EffectiveRateLimit,
    /// No rate limit: the previous key lost eligibility (removed/disabled or
    /// an in-request failover fallback had no better option).
    EligibilityFallback,
}

impl KeySwitchReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EffectiveRateLimit => "effective_rate_limit",
            Self::EligibilityFallback => "eligibility_fallback",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyPhase {
    Healthy,
    Cooling,
    /// Cooldown deadline passed; awaiting a single-flight probe.
    HalfOpen,
    /// HalfOpen and a probe request currently owns the lease.
    Probing,
}

impl KeyPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Cooling => "cooling",
            Self::HalfOpen => "half_open",
            Self::Probing => "probing",
        }
    }
}

#[derive(Clone)]
pub struct KeyPool {
    inner: Arc<Inner>,
}

struct Inner {
    keys: Vec<KeyEntry>,
    active: AtomicUsize,
    /// Unix seconds at which the current active key became active.
    active_since_unix: AtomicU64,
    /// Signaled whenever materially persisted state changes (cooldown set or
    /// cleared). The debounced writer task persists on notification.
    dirty: Notify,
}

struct KeyEntry {
    configured_index: usize,
    name: Arc<str>,
    api_key: Arc<str>,
    runtime: Mutex<KeyRuntime>,
    requests: AtomicU64,
    successes: AtomicU64,
    rate_limits: AtomicU64,
}

#[derive(Default)]
struct KeyRuntime {
    /// `None` = Healthy. `Some` in the future = Cooling. `Some` elapsed =
    /// HalfOpen awaiting a probe. This representation makes restore from a
    /// wall-clock deadline trivial and keeps selection monotonic.
    cooldown_until: Option<Instant>,
    cooldown_until_wall: Option<SystemTime>,
    probe_in_flight: bool,
    rate_limit_kind: Option<RateLimitKind>,
    last_429_at: Option<SystemTime>,
    last_429_message: Option<String>,
    last_429_model: Option<String>,
    last_success_at: Option<SystemTime>,
}

#[derive(Clone)]
pub struct SelectedKey {
    pub index: usize,
    pub configured_index: usize,
    pub name: Arc<str>,
    pub is_probe: bool,
    api_key: Arc<str>,
}

impl SelectedKey {
    pub fn api_key(&self) -> &str {
        &self.api_key
    }
}

impl std::fmt::Debug for SelectedKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SelectedKey")
            .field("index", &self.index)
            .field("configured_index", &self.configured_index)
            .field("name", &self.name)
            .field("is_probe", &self.is_probe)
            .field("api_key", &"[REDACTED]")
            .finish()
    }
}

/// Releases the HalfOpen probe lease when dropped, so client cancellation or
/// transport errors can never strand a key outside single-flight probing.
pub struct ProbeLease<'a> {
    pool: &'a KeyPool,
    index: usize,
}

impl Drop for ProbeLease<'_> {
    fn drop(&mut self) {
        self.pool.release_probe(self.index);
    }
}

#[derive(Debug, Clone)]
pub struct RateLimitUpdate {
    pub cooldown: Duration,
    pub cooldown_until: SystemTime,
}

#[derive(Debug, Clone)]
pub struct KeyStateSnapshot {
    pub index: usize,
    pub configured_index: usize,
    pub name: Arc<str>,
    pub phase: KeyPhase,
    pub cooldown_remaining: Option<Duration>,
    pub cooldown_until: Option<SystemTime>,
    pub rate_limit_kind: Option<RateLimitKind>,
    pub last_429_at: Option<SystemTime>,
    pub last_429_message: Option<String>,
    pub last_429_model: Option<String>,
    pub last_success_at: Option<SystemTime>,
    pub requests: u64,
    pub successes: u64,
    pub rate_limits: u64,
}

#[derive(Debug, Clone, Default)]
pub struct RestoreSummary {
    pub restored_cooldowns: usize,
    pub expired_entries: usize,
    pub unknown_entries: usize,
    pub restored_active_key: Option<String>,
}

impl KeyPool {
    pub fn new(configured: &[ClineKeyConfig]) -> Self {
        let keys = configured
            .iter()
            .enumerate()
            .filter(|(_, key)| key.enabled)
            .map(|(configured_index, key)| KeyEntry {
                configured_index,
                name: Arc::from(key.name.as_str()),
                api_key: Arc::from(key.api_key.as_str()),
                runtime: Mutex::new(KeyRuntime::default()),
                requests: AtomicU64::new(0),
                successes: AtomicU64::new(0),
                rate_limits: AtomicU64::new(0),
            })
            .collect();
        Self {
            inner: Arc::new(Inner {
                keys,
                active: AtomicUsize::new(0),
                active_since_unix: AtomicU64::new(unix_secs(SystemTime::now())),
                dirty: Notify::new(),
            }),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.keys.is_empty()
    }

    pub fn dirty(&self) -> &Notify {
        &self.inner.dirty
    }

    /// Strict sticky selection: the active key if eligible, else the next
    /// eligible key in configured order starting at the active key. An
    /// expired-cooldown (HalfOpen) key is probed single-flight; if every key
    /// is cooling or already probing, a HalfOpen key with an in-flight probe
    /// may still be used as a fallback so behavior is never worse than plain
    /// request forwarding.
    pub fn select(&self, attempted: &HashSet<usize>) -> Option<SelectedKey> {
        let count = self.len();
        if count == 0 {
            return None;
        }
        let active = self.inner.active.load(Ordering::Acquire) % count;
        let now = Instant::now();
        self.scan(attempted, active, count, now, false)
            .or_else(|| self.scan(attempted, active, count, now, true))
    }

    fn scan(
        &self,
        attempted: &HashSet<usize>,
        active: usize,
        count: usize,
        now: Instant,
        allow_probed: bool,
    ) -> Option<SelectedKey> {
        for offset in 0..count {
            let index = (active + offset) % count;
            if attempted.contains(&index) {
                continue;
            }
            let key = &self.inner.keys[index];
            let mut runtime = lock(&key.runtime);
            if runtime.cooldown_until.is_some_and(|until| until > now) {
                continue;
            }
            let mut is_probe = false;
            if runtime.cooldown_until.is_some() {
                // HalfOpen: the cooldown deadline has passed. Only one
                // request may own the probe lease.
                if runtime.probe_in_flight && !allow_probed {
                    continue;
                }
                if !runtime.probe_in_flight {
                    runtime.probe_in_flight = true;
                    is_probe = true;
                }
            }
            drop(runtime);
            key.requests.fetch_add(1, Ordering::Relaxed);
            if index != active {
                let reason = if self.is_cooling(active, now) {
                    KeySwitchReason::EffectiveRateLimit
                } else {
                    KeySwitchReason::EligibilityFallback
                };
                self.set_active(index, active, reason);
            }
            return Some(SelectedKey {
                index,
                configured_index: key.configured_index,
                name: key.name.clone(),
                is_probe,
                api_key: key.api_key.clone(),
            });
        }
        None
    }

    /// Mark a key cooling after a classified effective HTTP 429. No other
    /// call site may set a cooldown or move the active key for rate-limit
    /// reasons.
    pub fn mark_http_429(
        &self,
        index: usize,
        cooldown: Duration,
        kind: RateLimitKind,
        message: Option<String>,
        model: Option<String>,
    ) -> RateLimitUpdate {
        let cooldown = cooldown.min(MAX_COOLDOWN);
        let now = Instant::now();
        let wall_now = SystemTime::now();
        let cooldown_until = wall_now.checked_add(cooldown).unwrap_or(wall_now);
        if let Some(key) = self.inner.keys.get(index) {
            let mut runtime = lock(&key.runtime);
            runtime.cooldown_until = now.checked_add(cooldown);
            runtime.cooldown_until_wall = Some(cooldown_until);
            runtime.probe_in_flight = false;
            runtime.rate_limit_kind = Some(kind);
            runtime.last_429_at = Some(wall_now);
            runtime.last_429_message = message;
            runtime.last_429_model = model;
            key.rate_limits.fetch_add(1, Ordering::Relaxed);
        }
        self.advance_if_active(index, now);
        self.inner.dirty.notify_one();
        RateLimitUpdate {
            cooldown,
            cooldown_until,
        }
    }

    /// A usable 2xx upstream response (headers received). Clears any
    /// cooldown and closes HalfOpen probing. Never rotates the active key.
    pub fn mark_success(&self, index: usize) {
        let now = SystemTime::now();
        if let Some(key) = self.inner.keys.get(index) {
            let mut runtime = lock(&key.runtime);
            runtime.cooldown_until = None;
            runtime.cooldown_until_wall = None;
            runtime.probe_in_flight = false;
            runtime.last_success_at = Some(now);
            key.successes.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.dirty.notify_one();
    }

    /// Release a HalfOpen probe lease after a non-429 probe failure. The key
    /// stays HalfOpen (cooldown untouched); quota verdicts only come from
    /// confirmed 429s.
    pub fn release_probe(&self, index: usize) {
        if let Some(key) = self.inner.keys.get(index) {
            lock(&key.runtime).probe_in_flight = false;
        }
    }

    pub fn probe_lease(&self, index: usize) -> ProbeLease<'_> {
        ProbeLease { pool: self, index }
    }

    pub fn earliest_retry_after(&self) -> Option<Duration> {
        let now = Instant::now();
        self.inner
            .keys
            .iter()
            .filter_map(|key| {
                lock(&key.runtime)
                    .cooldown_until
                    .and_then(|until| until.checked_duration_since(now))
            })
            .min()
    }

    pub fn snapshots(&self) -> Vec<KeyStateSnapshot> {
        let now = Instant::now();
        self.inner
            .keys
            .iter()
            .enumerate()
            .map(|(index, key)| {
                let runtime = lock(&key.runtime);
                let cooldown_remaining = runtime
                    .cooldown_until
                    .and_then(|until| until.checked_duration_since(now));
                let phase = match runtime.cooldown_until {
                    None => KeyPhase::Healthy,
                    Some(until) if until > now => KeyPhase::Cooling,
                    Some(_) if runtime.probe_in_flight => KeyPhase::Probing,
                    Some(_) => KeyPhase::HalfOpen,
                };
                KeyStateSnapshot {
                    index,
                    configured_index: key.configured_index,
                    name: key.name.clone(),
                    phase,
                    cooldown_remaining,
                    cooldown_until: runtime.cooldown_until_wall,
                    rate_limit_kind: runtime.rate_limit_kind,
                    last_429_at: runtime.last_429_at,
                    last_429_message: runtime.last_429_message.clone(),
                    last_429_model: runtime.last_429_model.clone(),
                    last_success_at: runtime.last_success_at,
                    requests: key.requests.load(Ordering::Relaxed),
                    successes: key.successes.load(Ordering::Relaxed),
                    rate_limits: key.rate_limits.load(Ordering::Relaxed),
                }
            })
            .collect()
    }

    pub fn active_key_name(&self) -> Option<Arc<str>> {
        let count = self.len();
        if count == 0 {
            return None;
        }
        let active = self.inner.active.load(Ordering::Acquire) % count;
        self.inner.keys.get(active).map(|key| key.name.clone())
    }

    pub fn active_age(&self) -> Duration {
        let since = self.inner.active_since_unix.load(Ordering::Relaxed);
        let now = unix_secs(SystemTime::now());
        Duration::from_secs(now.saturating_sub(since))
    }

    /// Build the sanitized persisted representation of current runtime state.
    pub fn persisted_state(&self) -> PersistedState {
        let wall_now = SystemTime::now();
        let mut keys = BTreeMap::new();
        for key in &self.inner.keys {
            let runtime = lock(&key.runtime);
            keys.insert(
                key.name.to_string(),
                PersistedKeyState {
                    cooldown_until_unix_ms: runtime.cooldown_until_wall.map(unix_ms),
                    rate_limit_kind: runtime.rate_limit_kind,
                    model: runtime.last_429_model.clone(),
                    last_429_at_unix_ms: runtime.last_429_at.map(unix_ms),
                    last_success_at_unix_ms: runtime.last_success_at.map(unix_ms),
                },
            );
        }
        PersistedState {
            version: STATE_SCHEMA_VERSION,
            updated_at_unix_ms: unix_ms(wall_now),
            active_key: self.active_key_name().map(|name| name.to_string()),
            keys,
        }
    }

    /// Restore runtime state from persisted state, matching strictly by key
    /// name. Unknown names are dropped; expired deadlines restore as HalfOpen
    /// (probe-eligible, not cooling); implausibly far deadlines are capped.
    pub fn restore(&self, state: &PersistedState) -> RestoreSummary {
        let now_instant = Instant::now();
        let now_wall = SystemTime::now();
        let mut summary = RestoreSummary::default();
        for (name, persisted) in &state.keys {
            let Some(key) = self.inner.keys.iter().find(|key| &*key.name == name) else {
                summary.unknown_entries = summary.unknown_entries.saturating_add(1);
                continue;
            };
            let mut runtime = lock(&key.runtime);
            if let Some(deadline_ms) = persisted.cooldown_until_unix_ms {
                let deadline = UNIX_EPOCH
                    .checked_add(Duration::from_millis(deadline_ms))
                    .unwrap_or(now_wall);
                match deadline.duration_since(now_wall) {
                    Ok(remaining) => {
                        let remaining = remaining.min(MAX_COOLDOWN);
                        runtime.cooldown_until =
                            Some(now_instant.checked_add(remaining).unwrap_or(now_instant));
                        runtime.cooldown_until_wall = Some(deadline);
                        summary.restored_cooldowns = summary.restored_cooldowns.saturating_add(1);
                    }
                    Err(_) => {
                        // Expired while the process was down: restore as
                        // HalfOpen, never as an active cooldown.
                        runtime.cooldown_until = Some(now_instant);
                        runtime.cooldown_until_wall = Some(deadline);
                        summary.expired_entries = summary.expired_entries.saturating_add(1);
                    }
                }
                runtime.rate_limit_kind = persisted.rate_limit_kind;
                runtime.last_429_model = persisted.model.clone();
            } else {
                runtime.rate_limit_kind = persisted.rate_limit_kind;
                runtime.last_429_model = persisted.model.clone();
            }
            runtime.last_429_at = persisted.last_429_at_unix_ms.and_then(wall_from_unix_ms);
            runtime.last_success_at = persisted
                .last_success_at_unix_ms
                .and_then(wall_from_unix_ms);
        }
        if let Some(active_name) = &state.active_key {
            if let Some(index) = self
                .inner
                .keys
                .iter()
                .position(|key| &*key.name == active_name)
            {
                self.inner.active.store(index, Ordering::Release);
                self.inner
                    .active_since_unix
                    .store(state.updated_at_unix_ms / 1000, Ordering::Relaxed);
                summary.restored_active_key = Some(active_name.clone());
            }
        }
        summary
    }

    fn set_active(&self, index: usize, previous: usize, reason: KeySwitchReason) {
        let old_name = self
            .inner
            .keys
            .get(previous)
            .map(|key| key.name.to_string())
            .unwrap_or_default();
        let new_name = self
            .inner
            .keys
            .get(index)
            .map(|key| key.name.to_string())
            .unwrap_or_default();
        let _ = self.inner.active.compare_exchange(
            previous,
            index,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        self.inner
            .active_since_unix
            .store(unix_secs(SystemTime::now()), Ordering::Relaxed);
        tracing::info!(
            old_key = %old_name,
            new_key = %new_name,
            reason = reason.as_str(),
            "Cline active key changed"
        );
    }

    fn advance_if_active(&self, rejected: usize, now: Instant) {
        let count = self.len();
        if count == 0 || self.inner.active.load(Ordering::Acquire) % count != rejected {
            return;
        }
        for offset in 1..count {
            let candidate = (rejected + offset) % count;
            if self.is_eligible_for_advance(candidate, now) {
                self.set_active(candidate, rejected, KeySwitchReason::EffectiveRateLimit);
                return;
            }
        }
        // Every other key is cooling: keep active pointing at the rejected
        // key so cooldown-expiry probing starts from a stable place.
    }

    fn is_eligible_for_advance(&self, index: usize, now: Instant) -> bool {
        let Some(key) = self.inner.keys.get(index) else {
            return false;
        };
        let runtime = lock(&key.runtime);
        !runtime.cooldown_until.is_some_and(|until| until > now) && !runtime.probe_in_flight
    }

    fn is_cooling(&self, index: usize, now: Instant) -> bool {
        let Some(key) = self.inner.keys.get(index) else {
            return true;
        };
        lock(&key.runtime)
            .cooldown_until
            .is_some_and(|until| until > now)
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn unix_ms(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn unix_secs(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn wall_from_unix_ms(ms: u64) -> Option<SystemTime> {
    UNIX_EPOCH.checked_add(Duration::from_millis(ms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pool(count: usize) -> KeyPool {
        KeyPool::new(
            &(0..count)
                .map(|index| ClineKeyConfig {
                    name: format!("key-{index}"),
                    api_key: format!("secret-{index}"),
                    enabled: true,
                })
                .collect::<Vec<_>>(),
        )
    }

    fn pool_named(names: &[&str]) -> KeyPool {
        KeyPool::new(
            &names
                .iter()
                .enumerate()
                .map(|(index, name)| ClineKeyConfig {
                    name: (*name).into(),
                    api_key: format!("secret-{index}"),
                    enabled: true,
                })
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn normal_selection_is_sticky_and_429_advances() {
        let pool = pool(3);
        let attempted = HashSet::new();
        assert_eq!(pool.select(&attempted).unwrap().index, 0);
        assert_eq!(pool.select(&attempted).unwrap().index, 0);
        pool.mark_http_429(
            0,
            Duration::from_secs(60),
            RateLimitKind::DailyQuota,
            None,
            None,
        );
        assert_eq!(pool.select(&attempted).unwrap().index, 1);
        assert_eq!(pool.select(&attempted).unwrap().index, 1);
    }

    #[test]
    fn attempted_keys_are_never_selected_twice() {
        let pool = pool(3);
        let mut attempted = HashSet::new();
        for expected in 0..3 {
            let selected = pool.select(&attempted).unwrap();
            assert_eq!(selected.index, expected);
            attempted.insert(selected.index);
        }
        assert!(pool.select(&attempted).is_none());
    }

    #[test]
    fn ten_thousand_successful_selections_never_rotate() {
        let pool = pool_named(&["a", "b", "c"]);
        let attempted = HashSet::new();
        for _ in 0..10_000 {
            assert_eq!(pool.select(&attempted).unwrap().name.as_ref(), "a");
        }
        assert_eq!(pool.active_key_name().unwrap().as_ref(), "a");
    }

    #[test]
    fn success_and_snapshots_never_change_active() {
        let pool = pool(2);
        pool.mark_success(0);
        assert_eq!(pool.select(&HashSet::new()).unwrap().index, 0);
        let snapshot = &pool.snapshots()[0];
        assert_eq!(snapshot.phase, KeyPhase::Healthy);
        assert_eq!(snapshot.successes, 1);
    }

    #[tokio::test]
    async fn expired_key_can_become_eligible_again() {
        let pool = pool(2);
        pool.mark_http_429(
            0,
            Duration::from_millis(5),
            RateLimitKind::Transient,
            None,
            None,
        );
        assert_eq!(pool.select(&HashSet::new()).unwrap().index, 1);
        tokio::time::sleep(Duration::from_millis(10)).await;
        pool.mark_http_429(
            1,
            Duration::from_secs(60),
            RateLimitKind::Transient,
            None,
            None,
        );
        assert_eq!(pool.select(&HashSet::new()).unwrap().index, 0);
    }

    #[test]
    fn half_open_probe_is_single_flight_and_does_not_steal_healthy_active() {
        let pool = pool_named(&["a", "b"]);
        pool.mark_http_429(
            0,
            Duration::from_secs(60),
            RateLimitKind::DailyQuota,
            None,
            None,
        );
        // Fast-forward: give b a tiny cooldown (it becomes active after the
        // a 429), then let it expire so b is HalfOpen while a stays cooling.
        pool.mark_http_429(
            1,
            Duration::from_millis(5),
            RateLimitKind::Transient,
            None,
            None,
        );
        std::thread::sleep(Duration::from_millis(10));
        // Now a is cooling (60s), b is HalfOpen and active.
        let first = pool.select(&HashSet::new()).unwrap();
        assert_eq!(first.name.as_ref(), "b");
        assert!(first.is_probe);
        let second = pool.select(&HashSet::new()).unwrap();
        // Single-flight: the probe lease is held, but with no alternative key
        // the fallback may still use b without a second probe.
        assert_eq!(second.name.as_ref(), "b");
        assert!(!second.is_probe);
        pool.mark_success(first.index);
        assert_eq!(pool.snapshots()[1].phase, KeyPhase::Healthy);
    }

    #[test]
    fn half_open_never_steals_active_from_healthy_key() {
        let pool = pool_named(&["a", "b"]);
        // a: long cooldown that we then expire. b: healthy active.
        pool.mark_http_429(
            0,
            Duration::from_secs(60),
            RateLimitKind::DailyQuota,
            None,
            None,
        );
        pool.inner.active.store(1, Ordering::Release);
        // Expire a's cooldown by writing a past deadline directly.
        lock(&pool.inner.keys[0].runtime).cooldown_until = Some(Instant::now());
        for _ in 0..100 {
            assert_eq!(pool.select(&HashSet::new()).unwrap().name.as_ref(), "b");
        }
        assert_eq!(pool.active_key_name().unwrap().as_ref(), "b");
        // b confirms a 429; only then is a reconsidered.
        pool.mark_http_429(
            1,
            Duration::from_secs(60),
            RateLimitKind::DailyQuota,
            None,
            None,
        );
        let selected = pool.select(&HashSet::new()).unwrap();
        assert_eq!(selected.name.as_ref(), "a");
        assert!(selected.is_probe);
    }

    #[test]
    fn probe_429_restores_cooldown_and_probe_success_recovers() {
        let pool = pool_named(&["a", "b"]);
        pool.mark_http_429(
            0,
            Duration::from_millis(1),
            RateLimitKind::Transient,
            None,
            None,
        );
        std::thread::sleep(Duration::from_millis(5));
        pool.inner.active.store(1, Ordering::Release);
        lock(&pool.inner.keys[0].runtime).cooldown_until = Some(Instant::now());
        pool.mark_http_429(
            1,
            Duration::from_secs(60),
            RateLimitKind::DailyQuota,
            None,
            None,
        );
        // a is the only selectable key: HalfOpen probe.
        let probe = pool.select(&HashSet::new()).unwrap();
        assert_eq!(probe.name.as_ref(), "a");
        assert!(probe.is_probe);
        // Probe hits an effective 429 again: cooldown returns.
        pool.mark_http_429(
            probe.index,
            Duration::from_secs(120),
            RateLimitKind::DailyQuota,
            None,
            None,
        );
        assert_eq!(pool.snapshots()[0].phase, KeyPhase::Cooling);
        assert!(pool.snapshots()[0].cooldown_remaining.is_some());
        assert!(pool.select(&HashSet::new()).is_none());
        // Expire again, probe succeeds this time: Healthy.
        lock(&pool.inner.keys[0].runtime).cooldown_until = Some(Instant::now());
        let probe = pool.select(&HashSet::new()).unwrap();
        assert!(probe.is_probe);
        pool.mark_success(probe.index);
        assert_eq!(pool.snapshots()[0].phase, KeyPhase::Healthy);
        assert_eq!(pool.snapshots()[0].cooldown_remaining, None);
    }

    #[test]
    fn non_429_failures_release_the_probe_without_cooling() {
        let pool = pool_named(&["a", "b"]);
        pool.mark_http_429(
            0,
            Duration::from_millis(1),
            RateLimitKind::Transient,
            None,
            None,
        );
        // b (now active) confirms a 429 too, so the only selectable key is
        // the expired HalfOpen a.
        pool.mark_http_429(
            1,
            Duration::from_secs(60),
            RateLimitKind::DailyQuota,
            None,
            None,
        );
        std::thread::sleep(Duration::from_millis(5));
        let probe = pool.select(&HashSet::new()).unwrap();
        assert!(probe.is_probe);
        drop(pool.probe_lease(probe.index)); // simulates transport failure path
        let snapshot = &pool.snapshots()[0];
        assert_eq!(snapshot.phase, KeyPhase::HalfOpen);
        assert_eq!(snapshot.cooldown_remaining, None);
        // A new request may probe again.
        assert!(pool.select(&HashSet::new()).unwrap().is_probe);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_half_open_probing_is_single_flight() {
        let pool = pool_named(&["a", "b"]);
        pool.mark_http_429(
            0,
            Duration::from_millis(1),
            RateLimitKind::Transient,
            None,
            None,
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
        pool.inner.active.store(1, Ordering::Release);
        lock(&pool.inner.keys[0].runtime).cooldown_until = Some(Instant::now());
        pool.mark_http_429(
            1,
            Duration::from_secs(60),
            RateLimitKind::DailyQuota,
            None,
            None,
        );
        let probes = Arc::new(AtomicU64::new(0));
        let mut tasks = Vec::new();
        for _ in 0..32 {
            let pool = pool.clone();
            let probes = probes.clone();
            tasks.push(tokio::spawn(async move {
                if let Some(selected) = pool.select(&HashSet::new()) {
                    if selected.is_probe {
                        probes.fetch_add(1, Ordering::Relaxed);
                        // Deliberately keep the lease: probe_in_flight stays
                        // set, so no later task can acquire a second probe.
                    }
                }
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(probes.load(Ordering::Relaxed), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_429_updates_keep_state_valid() {
        let pool = pool(4);
        let mut tasks = Vec::new();
        for _ in 0..32 {
            let pool = pool.clone();
            tasks.push(tokio::spawn(async move {
                pool.mark_http_429(
                    0,
                    Duration::from_secs(1),
                    RateLimitKind::DailyQuota,
                    None,
                    None,
                );
                pool.select(&HashSet::new()).map(|key| key.index)
            }));
        }
        for task in tasks {
            let index = task.await.unwrap().unwrap();
            assert!((1..4).contains(&index));
        }
        assert!(pool.snapshots()[0].cooldown_remaining.is_some());
    }

    #[test]
    fn selected_key_debug_is_redacted() {
        let key = pool(1).select(&HashSet::new()).unwrap();
        assert!(!format!("{key:?}").contains("secret-0"));
    }

    // --- persistence round-trip ---

    fn persisted_with_cooldowns(pool: &KeyPool) -> PersistedState {
        pool.persisted_state()
    }

    #[test]
    fn restore_keeps_cooling_key_cooling_and_restores_active() {
        let pool = pool_named(&["a", "b", "c"]);
        pool.mark_http_429(
            0,
            Duration::from_secs(3_600),
            RateLimitKind::DailyQuota,
            None,
            None,
        );
        pool.mark_http_429(
            1,
            Duration::from_secs(7_200),
            RateLimitKind::DailyQuota,
            None,
            None,
        );
        let state = persisted_with_cooldowns(&pool);
        assert_eq!(state.active_key.as_deref(), Some("c"));
        let persisted_json = serde_json::to_value(&state).unwrap();

        let restored = pool_named(&["a", "b", "c"]);
        let summary = restored.restore(&serde_json::from_value(persisted_json).unwrap());
        assert_eq!(summary.restored_cooldowns, 2);
        assert_eq!(summary.restored_active_key.as_deref(), Some("c"));
        assert_eq!(summary.unknown_entries, 0);
        let snapshots = restored.snapshots();
        assert_eq!(snapshots[0].phase, KeyPhase::Cooling);
        assert!(snapshots[0].cooldown_remaining.is_some());
        assert!(snapshots[0].cooldown_remaining.unwrap() <= Duration::from_secs(3_600));
        assert_eq!(snapshots[1].phase, KeyPhase::Cooling);
        assert_eq!(snapshots[2].phase, KeyPhase::Healthy);
        assert_eq!(restored.select(&HashSet::new()).unwrap().name.as_ref(), "c");
    }

    #[test]
    fn restore_ignores_expired_deadlines_as_half_open() {
        let state = PersistedState {
            version: STATE_SCHEMA_VERSION,
            updated_at_unix_ms: unix_ms(SystemTime::now()),
            active_key: Some("b".into()),
            keys: BTreeMap::from([(
                "a".into(),
                PersistedKeyState {
                    cooldown_until_unix_ms: Some(unix_ms(SystemTime::now()) - 1_000),
                    rate_limit_kind: Some(RateLimitKind::DailyQuota),
                    model: Some("z-ai/glm-5.3-flash".into()),
                    last_429_at_unix_ms: Some(unix_ms(SystemTime::now()) - 3_600_000),
                    last_success_at_unix_ms: None,
                },
            )]),
        };
        let pool = pool_named(&["a", "b"]);
        let summary = pool.restore(&state);
        assert_eq!(summary.expired_entries, 1);
        assert_eq!(summary.restored_cooldowns, 0);
        let snapshot = &pool.snapshots()[0];
        assert_ne!(snapshot.phase, KeyPhase::Cooling);
        assert_eq!(snapshot.cooldown_remaining, None);
        // Active key restored even though it was recorded alongside cooling.
        assert_eq!(pool.active_key_name().unwrap().as_ref(), "b");
    }

    #[test]
    fn restore_drops_unknown_names_and_follows_reordering() {
        let state = PersistedState {
            version: STATE_SCHEMA_VERSION,
            updated_at_unix_ms: unix_ms(SystemTime::now()),
            active_key: Some("brewlogic".into()),
            keys: BTreeMap::from([
                (
                    "removed".into(),
                    PersistedKeyState {
                        cooldown_until_unix_ms: Some(unix_ms(SystemTime::now()) + 60_000),
                        ..Default::default()
                    },
                ),
                (
                    "brewlogic".into(),
                    PersistedKeyState {
                        cooldown_until_unix_ms: Some(unix_ms(SystemTime::now()) + 120_000),
                        rate_limit_kind: Some(RateLimitKind::DailyQuota),
                        ..Default::default()
                    },
                ),
            ]),
        };
        // Config reordered and gained/lost keys relative to when state was
        // written; identity is the name, so state must follow the name.
        let pool = pool_named(&["zeta", "brewlogic", "alpha"]);
        let summary = pool.restore(&state);
        assert_eq!(summary.unknown_entries, 1);
        assert_eq!(summary.restored_active_key.as_deref(), Some("brewlogic"));
        let snapshots = pool.snapshots();
        assert_eq!(snapshots[0].phase, KeyPhase::Healthy);
        assert_eq!(snapshots[1].phase, KeyPhase::Cooling);
        assert_eq!(snapshots[2].phase, KeyPhase::Healthy);
        // Active pointer is restored by name (verified above); because
        // brewlogic itself is cooling, selection correctly advances to the
        // next eligible key instead of sending to a cooling key.
        assert_eq!(pool.active_key_name().unwrap().as_ref(), "brewlogic");
        assert_eq!(pool.select(&HashSet::new()).unwrap().name.as_ref(), "alpha");
    }

    #[test]
    fn restore_caps_implausibly_far_deadlines() {
        let state = PersistedState {
            version: STATE_SCHEMA_VERSION,
            updated_at_unix_ms: unix_ms(SystemTime::now()),
            active_key: None,
            keys: BTreeMap::from([(
                "a".into(),
                PersistedKeyState {
                    cooldown_until_unix_ms: Some(253_402_300_799_000), // year 9999
                    ..Default::default()
                },
            )]),
        };
        let pool = pool_named(&["a", "b"]);
        pool.restore(&state);
        let remaining = pool.snapshots()[0].cooldown_remaining.unwrap();
        assert!(remaining <= MAX_COOLDOWN);
    }

    #[test]
    fn persisted_state_never_contains_key_material() {
        let pool = KeyPool::new(&[ClineKeyConfig {
            name: "super-secret-key-name".into(),
            api_key: "sk-super-secret-value-123".into(),
            enabled: true,
        }]);
        pool.mark_http_429(
            0,
            Duration::from_secs(60),
            RateLimitKind::DailyQuota,
            Some("Bearer sk-super-secret-value-123 leaked".into()),
            Some("z-ai/glm-5.3-flash".into()),
        );
        let rendered = serde_json::to_string(&pool.persisted_state()).unwrap();
        assert!(!rendered.contains("sk-super-secret-value-123"));
        assert!(!rendered.contains("Bearer"));
        let value: serde_json::Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(
            value["keys"]["super-secret-key-name"]["rate_limit_kind"],
            json!("daily_quota")
        );
    }

    #[test]
    fn rapid_multi_key_429_burst_is_fully_represented() {
        let pool = pool(5);
        for index in 0..5 {
            pool.mark_http_429(
                index,
                Duration::from_secs(3_600 * (index as u64 + 1)),
                RateLimitKind::DailyQuota,
                None,
                None,
            );
        }
        let state = pool.persisted_state();
        for index in 0..5 {
            let entry = &state.keys[&format!("key-{index}")];
            assert!(entry.cooldown_until_unix_ms.is_some());
            assert_eq!(entry.rate_limit_kind, Some(RateLimitKind::DailyQuota));
        }
    }
}
