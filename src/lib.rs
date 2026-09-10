//! cline-proxy library surface: exposed so benches, integration tests, and
//! tooling can exercise the exact modules the production binary uses.

pub mod anthropic;
pub mod cache;
pub mod config;
pub mod console;
pub mod context_guard;
pub mod glm53;
pub mod obs;
pub mod optimize;
pub mod pool;
pub mod rate_limit;
pub mod reasoning_shadow;
pub mod redaction;
pub mod server;
pub mod state;
pub mod stream_watch;
pub mod upstream;
