//! Concurrency-safe sticky sequential Cline API-key selection.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime};

use crate::config::ClineKeyConfig;

const MAX_COOLDOWN: Duration = Duration::from_secs(366 * 24 * 60 * 60);

#[derive(Clone)]
pub struct KeyPool {
    inner: Arc<Inner>,
}

struct Inner {
    keys: Vec<KeyEntry>,
    active: AtomicUsize,
}

struct KeyEntry {
    configured_index: usize,
    name: Arc<str>,
    api_key: Arc<str>,
    runtime: Mutex<KeyRuntime>,
}

#[derive(Default)]
struct KeyRuntime {
    cooldown_until: Option<Instant>,
    cooldown_until_wall: Option<SystemTime>,
    last_429_at: Option<SystemTime>,
    last_429_message: Option<String>,
    last_429_model: Option<String>,
}

#[derive(Clone)]
pub struct SelectedKey {
    pub index: usize,
    pub configured_index: usize,
    pub name: Arc<str>,
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
            .field("api_key", &"[REDACTED]")
            .finish()
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
    pub cooldown_remaining: Option<Duration>,
    pub cooldown_until: Option<SystemTime>,
    pub last_429_at: Option<SystemTime>,
    pub last_429_message: Option<String>,
    pub last_429_model: Option<String>,
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
            })
            .collect();
        Self {
            inner: Arc::new(Inner {
                keys,
                active: AtomicUsize::new(0),
            }),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.keys.is_empty()
    }

    /// Select the sticky active key, or the next non-cooling key in configured
    /// order. `attempted` makes the per-request attempt bound explicit.
    pub fn select(&self, attempted: &HashSet<usize>) -> Option<SelectedKey> {
        let count = self.len();
        if count == 0 {
            return None;
        }
        let active = self.inner.active.load(Ordering::Acquire) % count;
        let now = Instant::now();
        for offset in 0..count {
            let index = (active + offset) % count;
            if attempted.contains(&index) || self.is_cooling(index, now) {
                continue;
            }
            if index != active {
                let _ = self.inner.active.compare_exchange(
                    active,
                    index,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
            }
            let key = &self.inner.keys[index];
            return Some(SelectedKey {
                index,
                configured_index: key.configured_index,
                name: key.name.clone(),
                api_key: key.api_key.clone(),
            });
        }
        None
    }

    /// Mark a key cooling after a classified effective HTTP 429. No other call
    /// site may mutate cooldown or move the active key.
    pub fn mark_http_429(
        &self,
        index: usize,
        cooldown: Duration,
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
            runtime.last_429_at = Some(wall_now);
            runtime.last_429_message = message;
            runtime.last_429_model = model;
        }
        self.advance_if_active(index, now);
        RateLimitUpdate {
            cooldown,
            cooldown_until,
        }
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
                KeyStateSnapshot {
                    index,
                    configured_index: key.configured_index,
                    name: key.name.clone(),
                    cooldown_remaining: runtime
                        .cooldown_until
                        .and_then(|until| until.checked_duration_since(now)),
                    cooldown_until: runtime.cooldown_until_wall,
                    last_429_at: runtime.last_429_at,
                    last_429_message: runtime.last_429_message.clone(),
                    last_429_model: runtime.last_429_model.clone(),
                }
            })
            .collect()
    }

    fn advance_if_active(&self, rejected: usize, now: Instant) {
        let count = self.len();
        if count == 0 || self.inner.active.load(Ordering::Acquire) % count != rejected {
            return;
        }
        for offset in 1..count {
            let candidate = (rejected + offset) % count;
            if !self.is_cooling(candidate, now) {
                let _ = self.inner.active.compare_exchange(
                    rejected,
                    candidate,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                return;
            }
        }
    }

    fn is_cooling(&self, index: usize, now: Instant) -> bool {
        let Some(key) = self.inner.keys.get(index) else {
            return true;
        };
        let mut runtime = lock(&key.runtime);
        match runtime.cooldown_until {
            Some(until) if until > now => true,
            Some(_) => {
                runtime.cooldown_until = None;
                runtime.cooldown_until_wall = None;
                false
            }
            None => false,
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn normal_selection_is_sticky_and_429_advances() {
        let pool = pool(3);
        let attempted = HashSet::new();
        assert_eq!(pool.select(&attempted).unwrap().index, 0);
        assert_eq!(pool.select(&attempted).unwrap().index, 0);
        pool.mark_http_429(0, Duration::from_secs(60), None, None);
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

    #[tokio::test]
    async fn expired_key_can_become_eligible_again() {
        let pool = pool(2);
        pool.mark_http_429(0, Duration::from_millis(5), None, None);
        assert_eq!(pool.select(&HashSet::new()).unwrap().index, 1);
        tokio::time::sleep(Duration::from_millis(10)).await;
        pool.mark_http_429(1, Duration::from_secs(60), None, None);
        assert_eq!(pool.select(&HashSet::new()).unwrap().index, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_429_updates_keep_state_valid() {
        let pool = pool(4);
        let mut tasks = Vec::new();
        for _ in 0..32 {
            let pool = pool.clone();
            tasks.push(tokio::spawn(async move {
                pool.mark_http_429(0, Duration::from_secs(1), None, None);
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
}
