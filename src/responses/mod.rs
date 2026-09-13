//! OpenAI Responses API frontend (primary client: Grok Build).
//!
//! Independent protocol frontend that shares the upstream generation
//! infrastructure with the Anthropic Messages and OpenAI chat frontends:
//!
//! ```text
//! Responses request → responses::request (normalize)
//!                   → OpenAI Chat Completions body
//!                   → GLM policy / Cline upstream (shared invariants)
//!                   → OpenAI Chat Completions SSE
//!                   → responses::stream (Responses SSE events)
//! ```
//!
//! Event shapes follow the current Responses wire schema as consumed by
//! Grok Build (typed `type`-tagged `ResponseStreamEvent` frames): every
//! event must carry a known `type`, `sequence_number`, and the exact field
//! names the client deserializes. Unknown extra events would fail the
//! client's typed deserialization, so none are ever emitted.

pub mod request;
pub mod stream;
pub mod types;
