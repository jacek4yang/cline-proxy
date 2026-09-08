# ADR 0001: Persistent key runtime state (name-keyed, wall-clock, atomic)

Date: 2026-09-08 · Status: Accepted

## Context

`KeyPool` cooldown state was process-memory only. With 9 configured keys and
daily-quota cooldowns of 8–12 h, every restart re-discovered exhausted keys by
sending real requests: 5 failed 429 probes cost ~3.7 s of TTFT (~42%) in the
observed production trace, and burned no quota back.

## Decision

1. Persist runtime key state to a small local JSON file
   (`runtime.state_file`, default `./runtime-state.json`).
2. Persisted identity is the **configured key name** (validated unique),
   never the `Vec` index, so config reordering cannot cool the wrong key.
3. Deadlines are persisted as **Unix wall-clock milliseconds**. `Instant` is
   never serialized; on startup the remaining duration is reconstructed and
   runtime checks stay monotonic.
4. Writes are atomic (tmp file + rename), debounced and coalesced off the
   request hot path; healthy requests never touch the disk.
5. Corrupt/missing/incompatible state is **non-fatal**: sanitized warning,
   start with an empty state. State is advisory operational cache, not truth.
6. The state file never contains key material, auth headers, or raw upstream
   error text — only names, deadlines, rate-limit kind, model, timestamps.
7. Expired persisted deadlines are restored as HalfOpen (probe-eligible),
   not as active cooldowns; deadlines in the far future are capped at
   `MAX_COOLDOWN` so a corrupt clock cannot disable a key permanently.

## Consequences

- Restart with intact state: first request goes straight to the healthy key
  (`attempts=1, failed_probes=0`).
- Single-writer debounced task; a final synchronous flush runs during
  graceful shutdown.
- Future hot reload must re-match state by name (retain, drop stale,
  initialize new keys).

# ADR 0002: Strict sticky sequential routing + single-flight HalfOpen

Date: 2026-09-08 · Status: Accepted

## Context

Claude Code sends long, highly-prefix-repetitive requests. Keeping every
consecutive request on the same Cline credential preserves maximum possible
upstream routing/cache locality and consumes one quota pool at a time. The
business goal is *use a key until its quota is confirmed exhausted*, not
balanced consumption.

## Decision

1. The active key is global sticky state. Selection is
   "active if eligible, else next eligible in configured order" — never
   latency/inflight/random-based.
2. Success, 5xx, timeout, reset, TLS/DNS errors, malformed responses, stream
   interruptions, and client cancels **never** rotate the key. Only a
   classified effective HTTP 429 does.
3. An old key whose cooldown expires becomes HalfOpen (probe-eligible) but
   must not steal active back from a healthy key. It is reconsidered only
   when the current active key itself confirms a 429.
4. HalfOpen probing is single-flight: at most one in-flight probe per key;
   other requests prefer known-healthy keys. If no alternative exists they
   may use the HalfOpen key (never worse than previous behavior).
5. A successful probe is a usable 2xx upstream response (headers received);
   a non-429 probe failure releases the probe lease without touching
   cooldown (429-only invariant).
6. Every active-key change is logged with old key, new key, and a
   `KeySwitchReason`; `success` is never a switch reason.
7. Stickiness survives restart via the persisted `active_key` name
   (ADR 0001).

## Consequences

- We do not claim Cline's prompt cache is keyed per credential (unverified);
  stickiness is justified by locality preservation, not by that assumption.
- Adaptive/latency-aware routing, if ever added, must be config opt-in and
  must not change the default sticky behavior.
