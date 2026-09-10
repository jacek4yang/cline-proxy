use std::io::IsTerminal;
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Parser;

use cline_proxy::config::{default_config_path, Config, LogFormat};
use cline_proxy::console::{enable_ansi, ColorMode};
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
    /// Override `server.bind` without copying the config.
    #[arg(long)]
    bind: Option<String>,
    /// Override `runtime.state_file`.
    #[arg(long)]
    state_file: Option<PathBuf>,
    /// Override `logging.directory`.
    #[arg(long)]
    log_directory: Option<PathBuf>,
    /// Disable ANSI color (same as `--color never`).
    #[arg(long, conflicts_with = "color")]
    no_color: bool,
    /// ANSI color policy: auto (default), always, never.
    #[arg(long, value_name = "MODE")]
    color: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config_path = cli.config.clone().unwrap_or_else(default_config_path);
    let mut config = Config::load(&config_path)?;
    apply_cli_overrides(&mut config, &cli)?;
    config.validate()?;
    init_tracing(&config);
    let enabled_keys = config
        .cline_api_keys
        .iter()
        .filter(|key| key.enabled)
        .count();
    let logs = if config.logging.enabled() {
        "jsonl"
    } else {
        "console"
    };
    tracing::info!(
        "listening={} keys={enabled_keys} model={} logs={logs}",
        config.server.bind,
        config.models.default,
    );
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
                    "jsonl writer ready"
                );
                (Some(sink), Some(handle))
            }
            Err(error) => {
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

fn apply_cli_overrides(config: &mut Config, cli: &Cli) -> Result<()> {
    if let Some(bind) = &cli.bind {
        config.server.bind = bind.clone();
    }
    if let Some(path) = &cli.state_file {
        config.runtime.state_file = Some(path.display().to_string());
    }
    if let Some(path) = &cli.log_directory {
        config.logging.directory = Some(path.display().to_string());
    }
    if cli.no_color {
        config.runtime.log_color = ColorMode::Never;
    } else if let Some(value) = &cli.color {
        let Some(mode) = ColorMode::parse(value) else {
            bail!("--color must be auto, always, or never");
        };
        config.runtime.log_color = mode;
    }
    Ok(())
}

fn init_tracing(config: &Config) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&config.runtime.log_level));
    let ansi = enable_ansi(
        config.runtime.log_color,
        std::io::stderr().is_terminal(),
        cline_proxy::console::no_color_set(),
    );
    match config.runtime.log_format {
        LogFormat::Pretty => {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_writer(std::io::stderr)
                .with_ansi(ansi)
                .with_target(false)
                .compact()
                .try_init();
        }
        LogFormat::Json => {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_writer(std::io::stderr)
                .with_ansi(false)
                .with_target(false)
                .json()
                .try_init();
        }
    }
}
