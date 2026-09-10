# GROK_TASK.md — cline-proxy network latency + production deployment

## Goal

Repository: `D:\Workspace\cline-proxy` (`https://github.com/jacek4yang/cline-proxy`)
Production runtime: `D:\Workspace\cline-proxy-bin`
Target proxy: `socks5://127.0.0.1:10888`

Use Grok 4.6 High. Read `AGENTS.md` and `docs/DEVELOPMENT_STATE.md`, verify latest `main`, then complete implementation, tests, PR/CI/merge, release build, and safe production deployment.

Scope:
1. reduce outbound Cline TCP/TLS/HTTP latency where measurements justify it;
2. add first-class SOCKS5 support;
3. safely configure production to use `socks5://127.0.0.1:10888`;
4. keep JSONL logging correct;
5. upgrade `D:\Workspace\cline-proxy-bin` without breaking production.

Do not redesign reasoning/cache/shadow/tool semantics unless a concrete regression is proven.

## Secret safety

`D:\Workspace\cline-proxy-bin\config.json` contains real credentials.

You may let the local binary parse/use it and may modify it structurally, but NEVER print/read its raw contents into Grok context or terminal output. Never expose API keys, Authorization headers, tokens, proxy credentials, or secret-bearing config. Never commit production config/state/logs/temp secret configs.

Before editing production config:
- make a timestamped backup outside Git;
- edit JSON structurally;
- preserve unrelated settings;
- validate the result;
- never print either config file.

## Work

1. Start from latest `main`; run baseline fmt/check/clippy/tests and audit `Cargo.toml`, `src/upstream.rs`, `src/config.rs`, `src/main.rs`, `src/obs.rs`, `config.example.json`, README.

2. Add optional upstream proxy config:

```json
"upstream": {
  "proxy": "socks5://127.0.0.1:10888"
}
```

`null` = deterministic direct mode. Support `socks5h://` if cleanly supported; document DNS semantics.

Requirements:
- enable minimal Reqwest SOCKS support;
- keep exactly one shared `reqwest::Client`;
- proxy only outbound Cline traffic;
- preserve rustls, HTTP/2, connection pooling, keepalive, gzip, and existing timeout behavior;
- validate proxy URL at startup;
- direct mode must not silently inherit unwanted environment proxies;
- SOCKS/network errors do not rotate API keys;
- only effective HTTP 429 may fail over;
- never replay a committed stream;
- no automatic direct retry after an uncertain proxied POST.

3. Audit network latency before changing defaults:
- client reuse;
- TCP/TLS connection reuse;
- HTTP/2 reuse;
- pool idle timeout;
- TCP/HTTP2 keepalive;
- DNS path;
- unnecessary connection recreation;
- unnecessary body copies.

Keep only safe, measurable improvements. Never disable TLS verification or add unsafe/network hacks.

4. Add safe route telemetry:
`route=direct|socks5|socks5h`, `upstream_headers_ms`, `first_sse_ms`, `first_semantic_ms`, `duration_ms`, transport error class.
Never log secrets.

5. Run controlled sequential A/B using the real config through an isolated proxy:
`direct` vs `socks5://127.0.0.1:10888`.

Use comparable model/key/request class and multiple warm samples. Report median/spread for headers, first SSE, first semantic, total duration, and transport errors. Separate network latency from model generation time; do not claim SOCKS5 is faster without evidence.

6. If SOCKS5 validates successfully, update `D:\Workspace\cline-proxy-bin\config.json` to `socks5://127.0.0.1:10888`:
- timestamped config backup first;
- structurally change only `upstream.proxy`;
- preserve every other setting;
- validate with the real parser;
- never print config contents.
If 10888 is unreachable or invalid, leave production config unchanged.

7. Keep logging architecture:
- stderr = compact human summaries;
- `logs\events-*.jsonl` = bounded structured records.
Verify no `writer closed`, no ANSI in redirected output, no verbose INFO duplication, no prompt/reasoning/tool/API/proxy secrets, and logging never blocks inference.

8. Build `cargo build --release`. Validate new binary first on isolated bind/state/log paths using the real config:
- stream=true;
- stream=false;
- harmless tool call;
- SOCKS5 route;
- JSONL;
- restart logging;
- redirected output contains no ANSI.

Then safely deploy:
- stop current production proxy cleanly;
- back up current `cline-proxy.exe`;
- preserve config/state/historical logs;
- copy final `target\release\cline-proxy.exe` to `D:\Workspace\cline-proxy-bin\cline-proxy.exe`;
- start from production directory;
- verify SOCKS5 route;
- run one real Claude Code smoke request;
- verify JSONL;
- if validation fails, immediately restore previous exe/config backup.

Do not overwrite a running executable or delete production evidence.

9. Add regression tests for direct/socks5/socks5h config, invalid proxy URLs, secret redaction, route logging, transport failure not rotating keys, and existing stream/no-replay behavior.

Final gates:

```powershell
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo test --release --workspace --all-targets --all-features
```

Workflow:
`issue -> branch -> implementation -> tests -> A/B -> PR -> CI -> self-review -> merge -> final main build -> safe production deployment`

No GitHub Release/tag.

## Final report

Return only:

```text
NETWORK + PRODUCTION REPORT

Final main SHA:
PR/issue:

SOCKS5
implemented:
production route:
direct median headers/semantic:
SOCKS5 median headers/semantic:
measured improvement:

NETWORK CHANGES
...

PRODUCTION
config updated:
config backup:
binary updated:
binary backup:
startup:
Claude Code smoke:
JSONL:
ANSI:
rollback needed:

TESTS
fmt:
check:
clippy:
debug:
release:
CI:

REMAINING LATENCY
what is still upstream/model-side:

SECRETS
raw secrets printed: NO
secret files committed: NO
```

Do not fabricate measurements. Do not stop after analysis; continue until all safe, feasible work is complete.
