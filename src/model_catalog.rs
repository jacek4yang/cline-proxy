//! Dynamic Cline free-model catalog (issue #46).
//!
//! The free promotion list is served live by Cline
//! (`GET /ai/cline/recommended-models`, `free[].id`) and rotates over time;
//! the official extension reads it at runtime. This module caches that list
//! in-process with a TTL so `/v1/models` can advertise every currently free
//! model without a synchronous network dependency on the request path.
//!
//! Hard resource rules (AGENTS.md): bounded entries (64), bounded refresh
//! concurrency (one in-flight fetch), TTL-driven lazy refresh only — no
//! background polling task. A failed refresh keeps the previous list until
//! TTL expiry and logs a warning; the catalog is advisory only and never
//! fails a request.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;

/// Catalog bounds (issue #46 review: every long-lived collection needs
/// explicit bounds and a shutdown owner — this one lives on `AppState`, so
/// graceful shutdown drops it naturally).
#[derive(Debug, Clone, Copy)]
pub struct CatalogLimits {
    pub refresh_ttl: Duration,
    pub max_entries: usize,
}

impl Default for CatalogLimits {
    fn default() -> Self {
        Self {
            refresh_ttl: Duration::from_secs(6 * 60 * 60),
            max_entries: crate::upstream::MAX_FREE_MODEL_IDS,
        }
    }
}

#[derive(Debug, Default)]
struct Inner {
    ids: Vec<String>,
    fetched_at: Option<Instant>,
    /// Serializes refreshes: at most one in-flight network fetch.
    refresh_in_flight: bool,
}

/// Aggregate counters for observability — counts only, never content.
#[derive(Default)]
pub struct CatalogMetrics {
    pub refreshes: std::sync::atomic::AtomicU64,
    pub refresh_failures: std::sync::atomic::AtomicU64,
    pub served_from_cache: std::sync::atomic::AtomicU64,
}

pub struct ModelCatalog {
    inner: Mutex<Inner>,
    limits: CatalogLimits,
    pub metrics: Arc<CatalogMetrics>,
}

impl Default for ModelCatalog {
    fn default() -> Self {
        Self::new(CatalogLimits::default())
    }
}

impl ModelCatalog {
    pub fn new(limits: CatalogLimits) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            limits,
            metrics: Arc::new(CatalogMetrics::default()),
        }
    }

    /// Current free model IDs. Serves the cache within TTL; on expiry
    /// triggers one bounded refresh (awaiting it — the caller is the
    /// `/v1/models` handler, not the chat hot path). A failed refresh
    /// serves the stale list if any, else an empty list.
    pub async fn free_model_ids(&self, upstream: &crate::upstream::ClineUpstream) -> Vec<String> {
        let mut inner = self.inner.lock().await;
        let fresh = inner
            .fetched_at
            .is_some_and(|at| at.elapsed() < self.limits.refresh_ttl);
        if fresh || inner.refresh_in_flight {
            self.metrics
                .served_from_cache
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return inner.ids.clone();
        }
        inner.refresh_in_flight = true;
        // Fetch outside the intent of blocking others: the lock is held for
        // the whole refresh because callers are /v1/models (rare, cheap);
        // this guarantees at most one in-flight fetch and no stampede.
        let request_id = format!("models_{}", uuid::Uuid::new_v4().simple());
        let fetched = upstream.fetch_free_models(&request_id).await;
        inner.refresh_in_flight = false;
        self.metrics
            .refreshes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        match fetched {
            Some(ids) if !ids.is_empty() => {
                inner.ids = ids.into_iter().take(self.limits.max_entries).collect();
                inner.fetched_at = Some(Instant::now());
            }
            Some(_) => {
                // Empty list: a legitimate upstream answer (no promotions).
                // Cache it briefly so we do not hammer the endpoint, but a
                // short TTL keeps recovery fast.
                inner.ids.clear();
                inner.fetched_at = Some(
                    Instant::now()
                        - self
                            .limits
                            .refresh_ttl
                            .saturating_sub(Duration::from_secs(15 * 60)),
                );
            }
            None => {
                self.metrics
                    .refresh_failures
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // Keep the stale list (if any) and mark it freshly evaluated
                // so a flapping endpoint cannot turn /v1/models into a
                // per-request retry loop; the next check happens after a
                // full TTL.
                inner.fetched_at = Some(Instant::now());
            }
        }
        inner.ids.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> ModelCatalog {
        ModelCatalog::new(CatalogLimits {
            refresh_ttl: Duration::from_secs(3600),
            max_entries: 64,
        })
    }

    #[tokio::test]
    async fn parse_and_cache_via_upstream_failure_path() {
        // With an unreachable upstream the fetch fails; the catalog must
        // return an empty list (never panic, never retry per call).
        let catalog = catalog();
        let mut config = crate::config::Config::default();
        config.server.api_key = "test-secret".into();
        config.upstream.base_url = "http://127.0.0.1:1/api/v1".into();
        config.cline_api_keys = vec![crate::config::ClineKeyConfig {
            name: "k1".into(),
            api_key: "test-key".into(),
            enabled: true,
        }];
        config.runtime.state_file = None;
        let upstream = crate::upstream::ClineUpstream::new(
            &config,
            crate::pool::KeyPool::new(&config.cline_api_keys),
        )
        .unwrap();
        let ids = catalog.free_model_ids(&upstream).await;
        assert!(ids.is_empty());
        // Second call inside TTL: served without another failed fetch.
        let ids = catalog.free_model_ids(&upstream).await;
        assert!(ids.is_empty());
        assert_eq!(
            catalog
                .metrics
                .refresh_failures
                .load(std::sync::atomic::Ordering::Relaxed),
            1,
            "failed refresh must not repeat within TTL"
        );
    }

    #[test]
    fn limits_are_bounded() {
        let limits = CatalogLimits::default();
        assert_eq!(limits.max_entries, 64);
        assert!(limits.refresh_ttl >= Duration::from_secs(60));
    }
}
