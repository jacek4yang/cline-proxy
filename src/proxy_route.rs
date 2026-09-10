//! Outbound Cline route: deterministic direct, or SOCKS5 / SOCKS5h.
//!
//! Direct mode never inherits environment/system HTTP(S)_PROXY. SOCKS
//! credentials, if present, are kept for the client builder and never
//! appear in Display/Debug/logs.

use anyhow::{bail, Context, Result};
use reqwest::Url;

/// How the shared Cline HTTP client reaches `upstream.base_url`.
#[derive(Clone, PartialEq, Eq, Default)]
pub enum ProxyRoute {
    #[default]
    Direct,
    Socks5(ProxyTarget),
    Socks5h(ProxyTarget),
}

#[derive(Clone, PartialEq, Eq)]
pub struct ProxyTarget {
    /// Wire URL including optional credentials (never logged).
    wire: String,
    /// `scheme://host:port` with credentials stripped.
    display: String,
}

impl ProxyTarget {
    fn from_url(url: &Url) -> Self {
        let mut display = url.clone();
        let _ = display.set_username("");
        let _ = display.set_password(None);
        Self {
            wire: url.as_str().to_owned(),
            display: display.as_str().trim_end_matches('/').to_owned(),
        }
    }

    pub fn display(&self) -> &str {
        &self.display
    }

    pub fn wire(&self) -> &str {
        &self.wire
    }

    pub fn has_credentials(&self) -> bool {
        self.wire != self.display
    }
}

impl std::fmt::Debug for ProxyTarget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProxyTarget")
            .field("display", &self.display)
            .field("credentials", &self.has_credentials())
            .finish()
    }
}

impl ProxyRoute {
    /// `None` / empty / `direct` / `none` → Direct. Otherwise socks5 or socks5h.
    pub fn parse(raw: Option<&str>) -> Result<Self> {
        let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(Self::Direct);
        };
        if raw.eq_ignore_ascii_case("direct") || raw.eq_ignore_ascii_case("none") {
            return Ok(Self::Direct);
        }
        let url = Url::parse(raw).context("upstream.proxy must be a valid URL")?;
        if url.query().is_some() || url.fragment().is_some() {
            bail!("upstream.proxy must not contain a query or fragment");
        }
        if url.host_str().is_none() {
            bail!("upstream.proxy must include a host");
        }
        let target = ProxyTarget::from_url(&url);
        match url.scheme() {
            "socks5" => Ok(Self::Socks5(target)),
            "socks5h" => Ok(Self::Socks5h(target)),
            other => bail!(
                "upstream.proxy scheme {other:?} is not supported; use socks5:// or socks5h:// (or omit for direct)"
            ),
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Socks5(_) => "socks5",
            Self::Socks5h(_) => "socks5h",
        }
    }

    /// Safe for logs: no userinfo.
    pub fn display(&self) -> &str {
        match self {
            Self::Direct => "direct",
            Self::Socks5(target) | Self::Socks5h(target) => target.display(),
        }
    }

    pub fn credential_secret(&self) -> Option<&str> {
        match self {
            Self::Direct => None,
            Self::Socks5(target) | Self::Socks5h(target) if target.has_credentials() => {
                Some(target.wire())
            }
            Self::Socks5(_) | Self::Socks5h(_) => None,
        }
    }

    pub fn apply(&self, builder: reqwest::ClientBuilder) -> Result<reqwest::ClientBuilder> {
        match self {
            Self::Direct => Ok(builder.no_proxy()),
            Self::Socks5(target) | Self::Socks5h(target) => {
                let proxy = reqwest::Proxy::all(target.wire())
                    .context("building SOCKS proxy for upstream Cline client")?;
                Ok(builder.proxy(proxy))
            }
        }
    }
}

impl std::fmt::Debug for ProxyRoute {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Direct => formatter.write_str("Direct"),
            Self::Socks5(target) => formatter.debug_tuple("Socks5").field(target).finish(),
            Self::Socks5h(target) => formatter.debug_tuple("Socks5h").field(target).finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_direct_are_direct() {
        assert_eq!(ProxyRoute::parse(None).unwrap(), ProxyRoute::Direct);
        assert_eq!(ProxyRoute::parse(Some("")).unwrap(), ProxyRoute::Direct);
        assert_eq!(ProxyRoute::parse(Some("  ")).unwrap(), ProxyRoute::Direct);
        assert_eq!(
            ProxyRoute::parse(Some("direct")).unwrap(),
            ProxyRoute::Direct
        );
        assert_eq!(ProxyRoute::parse(Some("none")).unwrap(), ProxyRoute::Direct);
    }

    #[test]
    fn socks5_and_socks5h_are_accepted() {
        let socks = ProxyRoute::parse(Some("socks5://127.0.0.1:10888")).unwrap();
        assert_eq!(socks.kind(), "socks5");
        assert_eq!(socks.display(), "socks5://127.0.0.1:10888");
        let socks_h = ProxyRoute::parse(Some("socks5h://127.0.0.1:10888")).unwrap();
        assert_eq!(socks_h.kind(), "socks5h");
        assert_eq!(socks_h.display(), "socks5h://127.0.0.1:10888");
    }

    #[test]
    fn http_and_https_proxies_are_rejected() {
        assert!(ProxyRoute::parse(Some("http://127.0.0.1:8080")).is_err());
        assert!(ProxyRoute::parse(Some("https://127.0.0.1:8080")).is_err());
        assert!(ProxyRoute::parse(Some("socks4://127.0.0.1:1080")).is_err());
    }

    #[test]
    fn credentials_are_stripped_from_display_and_debug() {
        let route = ProxyRoute::parse(Some(
            "socks5://user:super-secret-proxy-pass@127.0.0.1:10888",
        ))
        .unwrap();
        let display = route.display().to_owned();
        let debug = format!("{route:?}");
        assert_eq!(display, "socks5://127.0.0.1:10888");
        assert!(!display.contains("super-secret-proxy-pass"));
        assert!(!debug.contains("super-secret-proxy-pass"));
        assert!(route.credential_secret().is_some());
    }

    #[test]
    fn missing_host_and_query_are_rejected() {
        assert!(ProxyRoute::parse(Some("socks5://")).is_err());
        assert!(ProxyRoute::parse(Some("socks5://127.0.0.1:10888?x=1")).is_err());
    }
}
