# Performance & Resource Model

Request-path ownership, async/blocking boundaries, and the resource
budget of every subsystem. Evidence: `cargo bench --bench
request_optimization` (release), obs unit tests, production E2E
(docs/DEVELOPMENT_STATE.md).

## Request ownership (one pass, no duplicate representations)

```text
network Bytes
  → parse once (serde_json Value)
  → system normalize + shadow restore + policy + canonicalize (mutations in place)
  → serialize once (upstream body)
  → intermediates drop at scope end
```

Measured single-request hot path (release, 1.3 MB, 24-turn Claude
Code-shaped fixture):

| stage | 1 KB | 100 KB | 1.3 MB |
|---|---:|---:|---:|
| convert | 0.1 ms | 0.2 ms | 0.9 ms |
| optimize (policy+strip+compaction) | 0.0 ms | 0.1 ms | 1.2 ms |
| cache pass (billing strip + canonical JSON + prefix hash) | 0.0 ms | 0.1 ms | 1.2 ms |
| serialize | 0.0 ms | 0.0 ms | 0.5 ms |

## Async / blocking boundaries

| work | where | bound |
|---|---|---|
| CPU-bound exact tokenizer | `spawn_blocking` + `Semaphore(1)` (`glm53.telemetry.max_concurrent_token_counts`) | busy → **skip** telemetry (logged), never queued |
| log JSON serialization + disk IO | dedicated `cline-log-writer` std::thread | bounded queue 8192, `try_send` or drop |
| request/protocol/SSE work | Tokio workers | no disk IO, no blocking calls, no per-chunk allocations beyond protocol needs |
| runtime state persistence | debounced writer task (150 ms coalesce) + final flush on shutdown | atomic tmp+rename, off request path |

## Subsystem budgets

| subsystem | memory bound | notes |
|---|---|---|
| reasoning shadow store | 256 sessions / 64 MiB / 1 MiB per entry / 10 min TTL | oversized reasoning skipped, never truncated |
| non-stream aggregation | 16 MiB (`MAX_AGGREGATED_RESPONSE_BYTES`) | text/args accumulate via `push_str` (linear; no raw SSE retention) |
| log queue | 8192 × summary (~8 MiB worst case) | full → drop + counter |
| flight recorder | 32 events per request | request-local; dropped with the request unless anomalous |
| SSE hot path | O(1) per chunk | counters + first-event timestamps only |
| log disk | `max_total_size_mb` quota (default 1 GB) | 85% cleanup watermark, 64 MB rotation |

## SSE discipline

Per chunk: `counter += 1`, `bytes += len`, first-event timestamp.
No JSON encoding, channel sends, locks, or timestamp formatting per
chunk. Detailed emission happens once at stream close.

## Intentionally not done

- No allocator swap (jemalloc/mimalloc) — no fragmentation evidence.
- No unsafe, no io_uring, no mmap, no SIMD JSON — hot path is already
  single-digit ms; these would be complexity without measurable return.
- No concurrent load tests yet: real multi-agent production traffic is
  the intended evidence source; deterministic correctness tests run in
  CI.
