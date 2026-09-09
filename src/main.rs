use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use cline_proxy::config::{default_config_path, Config, LogFormat};
use cline_proxy::server;

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
    // Adaptive bounded file logging (issue #16): a dedicated writer thread
    // owns all disk IO; requests only try_send typed summaries.
    let (log_sink, _writer_guard) = if config.logging.enabled() {
        let directory = config.logging.directory_path().unwrap();
        match cline_proxy::obs::spawn_writer(cline_proxy::obs::WriterConfig {
            directory: directory.clone(),
            max_file_size_mb: config.logging.max_file_size_mb,
            max_total_size_mb: config.logging.max_total_size_mb,
            cleanup_target_percent: config.logging.cleanup_target_percent,
            flush_interval_ms: config.logging.flush_interval_ms,
        }) {
            Ok((sink, handle)) => {
                tracing::info!(
                    directory = %directory.display(),
                    max_total_size_mb = config.logging.max_total_size_mb,
                    rotation_mb = config.logging.max_file_size_mb,
                    "adaptive JSONL logging enabled"
                );
                (Some(sink), Some(handle))
            }
            Err(error) => {
                // Fail open: console logging still works.
                tracing::warn!(
                    error = %error,
                    "file logging disabled (writer startup failed); proxy continues"
                );
                (None, None)
            }
        }
    } else {
        (None, None)
    };
    server::serve(server::AppState::with_log_sink(config, log_sink)?).await
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
