//! # Contracts-Macros YAML URL Validation
//!
//! Validate URLs in YAML config: must have `scheme://host[/path]`.
//! Returns invalid-URL list per RFC 3986 minimal check.
//!
//! Demonstrates the **CMM.131** recipe for PMAT-201 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: RFC 3986 (URI generic syntax); URL-validity heuristics.
//!
//! Run with: cargo run --example contracts_macros_yaml_url_validation
//!
//! Added by PMAT-201 (catalog 1432→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum UrlVerdict {
    Ok {
        invalid: Vec<String>,
        valid_count: u32,
    },
    InvalidConfig,
}

pub fn validate(urls: &[&str]) -> UrlVerdict {
    if urls.is_empty() {
        return UrlVerdict::InvalidConfig;
    }
    let mut invalid: Vec<String> = Vec::new();
    let mut valid_count = 0u32;
    for url in urls {
        if is_valid_url(url) {
            valid_count += 1;
        } else {
            invalid.push((*url).to_string());
        }
    }
    invalid.sort();
    invalid.dedup();
    UrlVerdict::Ok {
        invalid,
        valid_count,
    }
}

fn is_valid_url(url: &str) -> bool {
    let scheme_end = match url.find("://") {
        Some(p) if p > 0 => p,
        _ => return false,
    };
    let scheme = &url[..scheme_end];
    if !scheme
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
    {
        return false;
    }
    let after_scheme = &url[scheme_end + 3..];
    if after_scheme.is_empty() {
        return false;
    }
    // Host must not be empty.
    let host = after_scheme.split('/').next().unwrap_or("");
    !host.is_empty()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_url_validation")?;

    let urls = [
        "https://example.com/api",
        "http://localhost:8080",
        "not-a-url",
        "https://",
    ];
    println!("audit: {:?}", validate(&urls));
    println!("invalid: {:?}", validate(&[]));
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
    fn https_passes() {
        let v = validate(&["https://example.com"]);
        if let UrlVerdict::Ok { invalid, .. } = v {
            assert!(invalid.is_empty());
        }
    }

    #[test]
    fn missing_scheme_fails() {
        let v = validate(&["example.com"]);
        if let UrlVerdict::Ok { invalid, .. } = v {
            assert_eq!(invalid, vec!["example.com".to_string()]);
        }
    }

    #[test]
    fn missing_host_fails() {
        let v = validate(&["https://"]);
        if let UrlVerdict::Ok { invalid, .. } = v {
            assert_eq!(invalid, vec!["https://".to_string()]);
        }
    }

    #[test]
    fn http_with_port_passes() {
        let v = validate(&["http://localhost:8080"]);
        if let UrlVerdict::Ok { invalid, .. } = v {
            assert!(invalid.is_empty());
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(validate(&[]), UrlVerdict::InvalidConfig);
    }

    #[test]
    fn valid_count_correct() {
        let v = validate(&["https://x.com", "bad", "http://y.com"]);
        if let UrlVerdict::Ok { valid_count, .. } = v {
            assert_eq!(valid_count, 2);
        }
    }

    #[test]
    fn invalid_sorted() {
        let v = validate(&["zeta-bad", "alpha-bad"]);
        if let UrlVerdict::Ok { invalid, .. } = v {
            assert_eq!(
                invalid,
                vec!["alpha-bad".to_string(), "zeta-bad".to_string()]
            );
        }
    }

    #[test]
    fn deterministic() {
        let r1 = validate(&["https://x.com"]);
        let r2 = validate(&["https://x.com"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn ftp_scheme_passes() {
        let v = validate(&["ftp://files.example.com"]);
        if let UrlVerdict::Ok { invalid, .. } = v {
            assert!(invalid.is_empty());
        }
    }

    #[test]
    fn git_plus_ssh_scheme_passes() {
        let v = validate(&["git+ssh://github.com/foo/bar"]);
        if let UrlVerdict::Ok { invalid, .. } = v {
            assert!(invalid.is_empty());
        }
    }

    #[test]
    fn empty_string_fails() {
        let v = validate(&[""]);
        if let UrlVerdict::Ok { invalid, .. } = v {
            assert_eq!(invalid.len(), 1);
        }
    }

    #[test]
    fn many_urls_handled() {
        let urls: Vec<&str> = vec!["https://x.com"; 20];
        let v = validate(&urls);
        if let UrlVerdict::Ok { valid_count, .. } = v {
            assert_eq!(valid_count, 20);
        }
    }
}
