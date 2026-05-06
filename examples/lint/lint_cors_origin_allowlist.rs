//! # Lint CORS Origin Allowlist Validator
//!
//! CORS misconfig is a top OWASP API risk. Rules: never allow `*` with
//! credentialed requests; explicit allowlist must use full origin
//! (scheme + host + port); wildcard subdomains permitted via
//! `*.example.com` syntax. This recipe builds the validator.
//!
//! Demonstrates the **LINT.55** recipe for PMAT-131 (lint coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: OWASP API Security Top 10 (API8:2023).
//!
//! Run with: cargo run --example lint_cors_origin_allowlist
//!
//! Added by PMAT-131 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CorsVerdict {
    Ok,
    StarWithCredentials,
    InvalidOrigin { value: String },
    DuplicateOrigin { value: String },
    EmptyAllowlist,
}

pub fn validate(allowlist: &[&str], allow_credentials: bool) -> CorsVerdict {
    if allowlist.is_empty() {
        return CorsVerdict::EmptyAllowlist;
    }
    let mut seen = std::collections::HashSet::new();
    for origin in allowlist {
        if *origin == "*" {
            if allow_credentials {
                return CorsVerdict::StarWithCredentials;
            }
            // Wildcard alone (no credentials) is allowed.
            continue;
        }
        if !is_valid_origin(origin) {
            return CorsVerdict::InvalidOrigin {
                value: (*origin).to_string(),
            };
        }
        if !seen.insert(*origin) {
            return CorsVerdict::DuplicateOrigin {
                value: (*origin).to_string(),
            };
        }
    }
    CorsVerdict::Ok
}

fn is_valid_origin(s: &str) -> bool {
    if !s.starts_with("http://") && !s.starts_with("https://") {
        return false;
    }
    let scheme_split = s.split_once("://").map_or("", |(_, rest)| rest);
    let host = scheme_split.split('/').next().unwrap_or("");
    if host.is_empty() {
        return false;
    }
    // Allow wildcard subdomain: "*.example.com"
    if let Some(domain) = host.strip_prefix("*.") {
        return is_valid_host_chars(domain);
    }
    // Otherwise: standard host:port
    let host_only = host.split(':').next().unwrap_or("");
    is_valid_host_chars(host_only)
}

fn is_valid_host_chars(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.')
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("lint_cors_origin_allowlist")?;

    for (al, cred) in [
        (&["https://example.com"][..], false),
        (&["*"][..], false),
        (&["*"][..], true),
        (&["https://example.com", "https://example.com"][..], false),
        (&["bad-no-scheme"][..], false),
        (&[][..], false),
    ] {
        println!("{al:?} cred={cred}  →  {:?}", validate(al, cred));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_https_origin_passes() {
        assert_eq!(validate(&["https://example.com"], false), CorsVerdict::Ok);
    }

    #[test]
    fn wildcard_subdomain_allowed() {
        assert_eq!(validate(&["https://*.example.com"], false), CorsVerdict::Ok);
    }

    #[test]
    fn star_with_credentials_rejected() {
        assert_eq!(validate(&["*"], true), CorsVerdict::StarWithCredentials);
    }

    #[test]
    fn star_without_credentials_passes() {
        assert_eq!(validate(&["*"], false), CorsVerdict::Ok);
    }

    #[test]
    fn missing_scheme_rejected() {
        let v = validate(&["example.com"], false);
        assert!(matches!(v, CorsVerdict::InvalidOrigin { .. }));
    }

    #[test]
    fn invalid_chars_rejected() {
        let v = validate(&["https://example com"], false);
        assert!(matches!(v, CorsVerdict::InvalidOrigin { .. }));
    }

    #[test]
    fn empty_allowlist_rejected() {
        assert_eq!(validate(&[], false), CorsVerdict::EmptyAllowlist);
    }

    #[test]
    fn duplicate_origin_rejected() {
        let v = validate(&["https://x.com", "https://x.com"], false);
        assert!(matches!(v, CorsVerdict::DuplicateOrigin { .. }));
    }

    #[test]
    fn port_in_origin_allowed() {
        assert_eq!(validate(&["http://localhost:8080"], false), CorsVerdict::Ok);
    }

    #[test]
    fn ftp_scheme_rejected() {
        let v = validate(&["ftp://example.com"], false);
        assert!(matches!(v, CorsVerdict::InvalidOrigin { .. }));
    }
}
