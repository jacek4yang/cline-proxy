# WebSearch for Claude Code

`cline-proxy` implements the Anthropic `web_search` server-tool contract and
executes searches against Cline's first-party endpoint. Claude Code does not
run the search locally.

## Supported versions

| Declaration | Status |
|---|---|
| `web_search_20250305` | Full support (direct search) |
| `web_search_20260209` | Direct search only when `allowed_callers` is `["direct"]` |
| `web_search_20260318` | Same as `20260209`; `response_inclusion` other than `"full"` is rejected |
| unknown `web_search_*` | `invalid_request_error` — never forwarded as a client tool |

Newer versions default to code-execution dynamic filtering. This gateway
cannot run that path. Set `allowed_callers: ["direct"]` for basic search.

## Behavior

- The Anthropic declaration is converted to a deterministic OpenAI function
  named `web_search` with a required `query` argument.
- Domain filters (`allowed_domains` **or** `blocked_domains`) come from the
  declaration and are enforced by the gateway. The model cannot weaken them.
- `max_uses` is honored and hard-capped at 5 searches per logical request.
- Stream and non-stream `/v1/messages` both work. Internal continuations
  keep one downstream Anthropic response open (no `message_stop` between
  search rounds).
- History replay accepts `server_tool_use` + `web_search_tool_result` and
  reconstructs matching OpenAI `tool_calls` / `role=tool` pairs.
- Budget exhaustion returns a `web_search_tool_result` error
  (`max_uses_exceeded`) and `stop_reason: pause_turn`.
- Search failures become Anthropic error blocks (`too_many_requests`,
  `invalid_tool_input`, `query_too_long`, `request_too_large`,
  `unavailable`). They do **not** rotate Cline chat keys.

Cline endpoint:

```text
POST {upstream.base_url}/search/websearch
```

Default: `POST https://api.cline.bot/api/v1/search/websearch`.

The search reuses the chat HTTP client, static Cline headers, selected
credential, request id, X-Task-ID, and SOCKS/direct route.

## Not supported

- WebFetch / fetching result URLs
- Dynamic filtering and `allowed_callers` other than `"direct"`
- Encrypted Anthropic `encrypted_content` / `encrypted_index` (never fabricated)
- `user_location` localization
- `response_inclusion: excluded`

## Observability

JSONL/console may record `web_searches`, error/round/result counts, and
search latency. Query text, snippets, page bodies, and full URLs are not
logged at INFO. API keys are never logged.
