mod anthropic;
mod config;
mod pool;
mod rate_limit;
mod redaction;
mod server;
mod upstream;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use crate::config::{default_config_path, Config, LogFormat};
use crate::server::AppState;

#[derive(Parser)]
#[command(
    name = "cline-proxy",
    version,
    about = "Sticky multi-key Cline gateway for OpenAI and Anthropic clients"
)]
struct Cli {
    /// JSON configuration path (or CLINE_PROXY_CONFIG).
    #[arg(long)]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config_path = cli.config.unwrap_or_else(default_config_path);
    let config = Config::load(&config_path)?;
    init_tracing(&config);
    let enabled_keys = config
        .cline_api_keys
        .iter()
        .filter(|key| key.enabled)
        .count();
    tracing::info!(
        config_path = %config_path.display(),
        bind = %config.server.bind,
        upstream = %config.upstream.base_url,
        enabled_keys,
        "configuration loaded"
    );
    server::serve(AppState::new(config)?).await
}

fn init_tracing(config: &Config) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&config.runtime.log_level));
    match config.runtime.log_format {
        LogFormat::Pretty => {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_target(false)
                .compact()
                .try_init();
        }
        LogFormat::Json => {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_target(false)
                .json()
                .try_init();
        }
    }
}
