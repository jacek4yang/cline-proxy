//! Bounded, memory-only reasoning shadow store (issue #10).
//!
//! When the client did NOT request thinking, GLM's reasoning for a tool
//! loop is consumed in-turn and never surfaced. That keeps Claude Code
//! from storing and replaying internal reasoning (the anti-amplification
//! gate), but it also discards the reasoning continuity GLM benefits from
//! across a multi-step tool loop: every tool round-trip re-reasons from
//! scratch.
//!
//! The shadow store bridges that gap inside the proxy: when a response
//! carries reasoning + tool calls, the reasoning is kept here keyed by
//! (session fingerprint, model, tool-call id). The next request in the
//! SAME reasoning epoch that supplies a matching `tool` result gets the
//! reasoning restored onto its assistant message as `reasoning_content`.
//!
//! Hard resource rules (ADR 0006):
//! - memory-only; never persisted, never logged (content or sizes beyond
//!   aggregate counters);
//! - disabled entirely when no stable session identity exists — a missing
//!   identity must never degrade into "whatever request arrived last";
//! - bounded by session count, total bytes, per-entry bytes, and TTL;
//!   oversized reasoning is skipped (never truncated — a truncated
//!   reasoning blob would be wrong context presented as real);
//! - entries are evicted on final answer, new human turn, TTL, or LRU
//!   pressure; a restart loses everything by design.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// One shadowed reasoning blob, shared by every tool call of the same
/// assistant turn (`Arc` — a 3-call turn stores the reasoning once).
struct ShadowEntry {
    reasoning: Arc<str>,
    created: Instant,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ShadowKey {
    session: [u8; 8],
    /// Low bits of the tool-call id hash; ids are upstream-generated
    /// (e.g. `call_...`/`toolu_...`) and unique within a conversation.
    tool_call: u64,
}

/// Aggregate counters for observability — sizes and counts only.
#[derive(Default)]
pub struct ShadowMetrics {
    pub stores: AtomicU64,
    pub restores: AtomicU64,
    pub misses: AtomicU64,
    pub evicted_entries: AtomicU64,
    pub dropped_oversized: AtomicU64,
    pub disabled_no_session: AtomicU64,
    pub active_entries: AtomicU64,
}

impl ShadowMetrics {
    fn snapshot(&self) -> (u64, u64, u64, u64, u64, u64, u64) {
        (
            self.stores.load(Ordering::Relaxed),
            self.restores.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.evicted_entries.load(Ordering::Relaxed),
            self.dropped_oversized.load(Ordering::Relaxed),
            self.disabled_no_session.load(Ordering::Relaxed),
            self.active_entries.load(Ordering::Relaxed),
        )
    }
}

/// Resource limits for the store (issue #10: every long-lived collection
/// must have explicit bounds).
#[derive(Debug, Clone, Copy)]
pub struct ShadowLimits {
    pub max_sessions: usize,
    pub max_total_bytes: usize,
    pub max_entry_bytes: usize,
    pub ttl: Duration,
}

impl Default for ShadowLimits {
    fn default() -> Self {
        Self {
            max_sessions: 256,
            // 64 MiB total; generous for reasoning text (hundreds of KB per
            // turn at most) while hard-bounding memory on any workload.
            max_total_bytes: 64 * 1024 * 1024,
            // A single reasoning blob beyond 1 MiB is runaway output, not
            // useful continuity.
            max_entry_bytes: 1024 * 1024,
            ttl: Duration::from_secs(600),
        }
    }
}

/// Session-scoped shadow state: tool-call entries.
#[derive(Default)]
struct SessionState {
    entries: HashMap<ShadowKey, ShadowEntry>,
    /// Approximate total reasoning bytes held by this session.
    bytes: usize,
}

pub struct ReasoningShadowStore {
    inner: Mutex<ShadowInner>,
    limits: ShadowLimits,
    pub metrics: ShadowMetrics,
}

struct ShadowInner {
    sessions: HashMap<String, SessionState>,
    /// FIFO eviction order across all sessions (oldest touch first).
    lru: VecDeque<String>,
    total_bytes: usize,
}

impl ReasoningShadowStore {
    pub fn new(limits: ShadowLimits) -> Self {
        Self {
            inner: Mutex::new(ShadowInner {
                sessions: HashMap::new(),
                lru: VecDeque::new(),
                total_bytes: 0,
            }),
            limits,
            metrics: ShadowMetrics::default(),
        }
    }

    /// Derived HMAC fingerprint → compact 8-byte key component.
    fn session_key(session_fingerprint: &str) -> [u8; 8] {
        // The fingerprint is already an HMAC digest fragment (16 hex
        // chars); compress to 8 bytes for the key without re-keying.
        let mut key = [0u8; 8];
        for (index, chunk) in session_fingerprint.as_bytes().chunks(2).take(8).enumerate() {
            let hex_value = u8::from_str_radix(std::str::from_utf8(chunk).unwrap_or("00"), 16)
                .unwrap_or(index as u8);
            key[index] = hex_value;
        }
        key
    }

    /// Store reasoning for a completed assistant turn that issued tool
    /// calls. `tool_call_ids` are the ids of that turn's calls (one entry
    /// shared by the whole group).
    pub fn store(&self, session_fingerprint: &str, tool_call_ids: &[String], reasoning: &str) {
        if reasoning.is_empty() || tool_call_ids.is_empty() {
            return;
        }
        let bytes = reasoning.len();
        if bytes > self.limits.max_entry_bytes {
            self.metrics
                .dropped_oversized
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
        let key_session = Self::session_key(session_fingerprint);
        let reasoning: Arc<str> = Arc::from(reasoning);
        let now = Instant::now();
        let mut inner = self.inner.lock().unwrap();
        let entry_bytes_per_call = bytes;
        let is_new = !inner.sessions.contains_key(session_fingerprint);
        if is_new {
            inner.lru.push_back(session_fingerprint.to_owned());
        }
        let session = inner
            .sessions
            .entry(session_fingerprint.to_owned())
            .or_default();
        let mut added_bytes = 0usize;
        for id in tool_call_ids {
            let mut hasher = DefaultHasher::new();
            id.hash(&mut hasher);
            let key = ShadowKey {
                session: key_session,
                tool_call: hasher.finish(),
            };
            session.entries.insert(
                key,
                ShadowEntry {
                    reasoning: reasoning.clone(),
                    created: now,
                },
            );
            added_bytes = added_bytes.saturating_add(entry_bytes_per_call);
        }
        session.bytes = session.bytes.saturating_add(added_bytes);
        inner.total_bytes = inner.total_bytes.saturating_add(added_bytes);
        self.metrics.stores.fetch_add(1, Ordering::Relaxed);
        self.enforce_limits(&mut inner);
        self.metrics
            .active_entries
            .store(inner.total_entry_count(), Ordering::Relaxed);
    }

    /// Restore reasoning for a historical assistant tool-call turn, if the
    /// shadow holds an entry for one of its call ids that has not expired.
    pub fn restore(&self, session_fingerprint: &str, tool_call_ids: &[String]) -> Option<Arc<str>> {
        if tool_call_ids.is_empty() {
            return None;
        }
        let key_session = Self::session_key(session_fingerprint);
        let now = Instant::now();
        let mut inner = self.inner.lock().unwrap();
        let session = inner.sessions.get_mut(session_fingerprint)?;
        for id in tool_call_ids {
            let mut hasher = DefaultHasher::new();
            id.hash(&mut hasher);
            let key = ShadowKey {
                session: key_session,
                tool_call: hasher.finish(),
            };
            if let Some(entry) = session.entries.get(&key) {
                if now.duration_since(entry.created) > self.limits.ttl {
                    // Expired: drop it lazily.
                    session.entries.remove(&key);
                    self.metrics.misses.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                self.metrics.restores.fetch_add(1, Ordering::Relaxed);
                return Some(entry.reasoning.clone());
            }
        }
        self.metrics.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Drop every entry for a session: new human turn or shutdown.
    pub fn clear_session(&self, session_fingerprint: &str) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(session) = inner.sessions.remove(session_fingerprint) {
            inner.total_bytes = inner.total_bytes.saturating_sub(session.bytes);
            self.metrics
                .evicted_entries
                .fetch_add(session.entries.len() as u64, Ordering::Relaxed);
            inner.lru.retain(|name| name != session_fingerprint);
        }
    }

    /// Restore shadowed reasoning into an OpenAI Chat Completions body, in
    /// place: for every assistant message after the newest *human* user
    /// message that carries tool_calls but no `reasoning_content`, look up
    /// any of its call ids in the shadow and attach the reasoning. Misses
    /// are silent (the message simply keeps no reasoning). Must be called
    /// before the historical-thinking strip; the epoch boundary logic in
    /// `crate::optimize` guarantees restored messages are kept.
    pub fn restore_into(
        &self,
        body: &mut serde_json::Map<String, serde_json::Value>,
        session_fingerprint: &str,
    ) {
        use serde_json::Value;
        let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
            return;
        };
        // Epoch boundary: newest plain user message (mirrors optimize.rs).
        let epoch_boundary = messages
            .iter()
            .rposition(|message| message.get("role").and_then(Value::as_str) == Some("user"))
            .unwrap_or(messages.len());
        let mut restored_any = false;
        for message in messages.iter_mut().skip(epoch_boundary) {
            if message.get("role").and_then(Value::as_str) != Some("assistant") {
                continue;
            }
            if message.get("reasoning_content").is_some() {
                continue;
            }
            let Some(calls) = message.get("tool_calls").and_then(Value::as_array) else {
                continue;
            };
            let ids: Vec<String> = calls
                .iter()
                .filter_map(|call| {
                    call.get("id")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned)
                })
                .collect();
            if let Some(reasoning) = self.restore(session_fingerprint, &ids) {
                message["reasoning_content"] = Value::String((*reasoning).to_owned());
                restored_any = true;
            }
        }
        if restored_any {
            // Reasoning that was restored has served its purpose if the
            // request now carries it; the shadow copy is kept until the
            // epoch ends (clear_session on final answer / new human turn).
        }
    }

    /// Enforce session-count, total-byte, and TTL bounds. Called under
    /// lock; O(sessions) worst case on insertion pressure, amortized O(1).
    fn enforce_limits(&self, inner: &mut ShadowInner) {
        let now = Instant::now();
        // TTL sweep.
        let expired: Vec<String> = inner
            .sessions
            .iter()
            .filter(|(_, state)| {
                state
                    .entries
                    .iter()
                    .all(|(_, entry)| now.duration_since(entry.created) > self.limits.ttl)
            })
            .map(|(name, _)| name.clone())
            .collect();
        for name in expired {
            if let Some(state) = inner.sessions.remove(&name) {
                inner.total_bytes = inner.total_bytes.saturating_sub(state.bytes);
                self.metrics
                    .evicted_entries
                    .fetch_add(state.entries.len() as u64, Ordering::Relaxed);
                inner.lru.retain(|candidate| candidate != &name);
            }
        }
        // Total-byte pressure: evict least-recently-touched sessions first.
        while inner.total_bytes > self.limits.max_total_bytes {
            let Some(victim) = inner.lru.front().cloned() else {
                break;
            };
            if let Some(state) = inner.sessions.remove(&victim) {
                inner.total_bytes = inner.total_bytes.saturating_sub(state.bytes);
                self.metrics
                    .evicted_entries
                    .fetch_add(state.entries.len() as u64, Ordering::Relaxed);
            }
            inner.lru.pop_front();
        }
        // Session-count pressure: same order.
        while inner.sessions.len() > self.limits.max_sessions {
            let Some(victim) = inner.lru.front().cloned() else {
                break;
            };
            if let Some(state) = inner.sessions.remove(&victim) {
                inner.total_bytes = inner.total_bytes.saturating_sub(state.bytes);
                self.metrics
                    .evicted_entries
                    .fetch_add(state.entries.len() as u64, Ordering::Relaxed);
            }
            inner.lru.pop_front();
        }
    }

    /// Log aggregate counters (never content).
    pub fn log_metrics(&self, interval_secs: u64) {
        let (stores, restores, misses, evicted, oversized, _disabled, active) =
            self.metrics.snapshot();
        if stores + restores + misses + evicted + oversized == 0 {
            return;
        }
        let total_bytes = { self.inner.lock().unwrap().total_bytes };
        tracing::info!(
            interval_secs,
            shadow_stores = stores,
            shadow_restores = restores,
            shadow_misses = misses,
            shadow_evicted_entries = evicted,
            shadow_dropped_oversized = oversized,
            shadow_active_entries = active,
            shadow_total_bytes = total_bytes,
            "reasoning shadow store metrics"
        );
        // Reset interval counters so each report covers one window.
        self.metrics.stores.store(0, Ordering::Relaxed);
        self.metrics.restores.store(0, Ordering::Relaxed);
        self.metrics.misses.store(0, Ordering::Relaxed);
        self.metrics.evicted_entries.store(0, Ordering::Relaxed);
        self.metrics.dropped_oversized.store(0, Ordering::Relaxed);
    }
}

impl ShadowInner {
    fn total_entry_count(&self) -> u64 {
        self.sessions
            .values()
            .map(|state| state.entries.len() as u64)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> ReasoningShadowStore {
        ReasoningShadowStore::new(ShadowLimits::default())
    }

    #[test]
    fn store_and_restore_roundtrip_shares_one_blob_across_calls() {
        let shadow = store();
        shadow.store(
            "session_a",
            &["call_1".into(), "call_2".into()],
            "reasoning about the fix",
        );
        let restored_1 = shadow.restore("session_a", &["call_1".into()]).unwrap();
        let restored_2 = shadow.restore("session_a", &["call_2".into()]).unwrap();
        // Same underlying blob (shared Arc), not copies.
        assert!(Arc::ptr_eq(&restored_1, &restored_2));
        assert_eq!(&*restored_1, "reasoning about the fix");
    }

    #[test]
    fn sessions_are_isolated() {
        let shadow = store();
        shadow.store("agent_a", &["call_1".into()], "agent A reasoning");
        shadow.store("agent_b", &["call_1".into()], "agent B reasoning");
        // Same tool-call id, different sessions: no cross-talk.
        assert_eq!(
            &*shadow.restore("agent_a", &["call_1".into()]).unwrap(),
            "agent A reasoning"
        );
        assert_eq!(
            &*shadow.restore("agent_b", &["call_1".into()]).unwrap(),
            "agent B reasoning"
        );
        assert!(shadow.restore("agent_c", &["call_1".into()]).is_none());
    }

    #[test]
    fn clear_session_drops_everything() {
        let shadow = store();
        shadow.store("s", &["call_1".into()], "r");
        shadow.clear_session("s");
        assert!(shadow.restore("s", &["call_1".into()]).is_none());
    }

    #[test]
    fn oversized_reasoning_is_dropped_not_truncated() {
        let shadow = ReasoningShadowStore::new(ShadowLimits {
            max_entry_bytes: 1_000,
            ..ShadowLimits::default()
        });
        let big = "x".repeat(2_000);
        shadow.store("s", &["call_1".into()], &big);
        assert!(shadow.restore("s", &["call_1".into()]).is_none());
        assert_eq!(shadow.metrics.dropped_oversized.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn total_byte_limit_evicts_lru_sessions() {
        let shadow = ReasoningShadowStore::new(ShadowLimits {
            max_sessions: 16,
            // Small budget: ~1 reasoning blob per session only.
            max_total_bytes: 2_000,
            max_entry_bytes: 1_000,
            ttl: Duration::from_secs(600),
        });
        let blob = "x".repeat(900);
        shadow.store("first", &["call_1".into()], &blob);
        shadow.store("second", &["call_1".into()], &blob);
        // Storing the third must evict "first" (FIFO/LRU order) to fit.
        shadow.store("third", &["call_1".into()], &blob);
        assert!(shadow.restore("first", &["call_1".into()]).is_none());
        assert!(shadow.restore("third", &["call_1".into()]).is_some());
        assert!(shadow.metrics.evicted_entries.load(Ordering::Relaxed) >= 1);
    }

    #[test]
    fn session_count_limit_is_enforced() {
        let shadow = ReasoningShadowStore::new(ShadowLimits {
            max_sessions: 2,
            ..ShadowLimits::default()
        });
        shadow.store("s1", &["call_1".into()], "r1");
        shadow.store("s2", &["call_1".into()], "r2");
        shadow.store("s3", &["call_1".into()], "r3");
        assert!(shadow.restore("s1", &["call_1".into()]).is_none());
        assert!(shadow.restore("s3", &["call_1".into()]).is_some());
    }

    #[test]
    fn expired_entries_are_lazy_dropped() {
        let shadow = ReasoningShadowStore::new(ShadowLimits {
            ttl: Duration::from_millis(1),
            ..ShadowLimits::default()
        });
        shadow.store("s", &["call_1".into()], "r");
        std::thread::sleep(Duration::from_millis(5));
        assert!(shadow.restore("s", &["call_1".into()]).is_none());
    }

    #[test]
    fn empty_inputs_are_noops() {
        let shadow = store();
        shadow.store("s", &[], "reasoning");
        shadow.store("s", &["call_1".into()], "");
        assert!(shadow.restore("s", &["call_1".into()]).is_none());
    }

    /// 256 sessions × ~100 KB reasoning must stay under the 64 MiB budget
    /// (issue #10 stress shape).
    #[test]
    fn stress_256_sessions_stay_bounded() {
        let shadow = ReasoningShadowStore::new(ShadowLimits {
            max_sessions: 256,
            max_total_bytes: 64 * 1024 * 1024,
            max_entry_bytes: 1024 * 1024,
            ttl: Duration::from_secs(600),
        });
        let blob = "y".repeat(100_000);
        for index in 0..300 {
            shadow.store(&format!("session_{index}"), &["call_1".into()], &blob);
        }
        let total = shadow.inner.lock().unwrap().total_bytes;
        assert!(total <= 64 * 1024 * 1024);
        // Oldest sessions were evicted.
        assert!(shadow.restore("session_0", &["call_1".into()]).is_none());
        assert!(shadow.restore("session_299", &["call_1".into()]).is_some());
    }
}
