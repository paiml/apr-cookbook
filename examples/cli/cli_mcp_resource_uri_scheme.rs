//! # apr mcp — Resource URI Scheme Validator
//!
//! MCP resource URIs follow `<scheme>://<authority>/<path>`. Recognised
//! schemes: `file`, `http`, `https`, `apr` (model artifacts). Local
//! file URIs require absolute paths; remote schemes require a host
//! component. This recipe builds the validator.
//!
//! Demonstrates the **MCP.4** recipe for PMAT-120 (apr mcp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MCP-001 + RFC 3986 (URI generic syntax)
//!
//! Run with: cargo run --example cli_mcp_resource_uri_scheme
//!
//! Added by PMAT-120 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    File,
    Http,
    Https,
    Apr,
}

#[derive(Debug, PartialEq)]
pub enum UriVerdict {
    Ok { scheme: Scheme },
    UnknownScheme,
    MissingScheme,
    FileMustBeAbsolute,
    RemoteMissingHost,
}

pub fn validate(uri: &str) -> UriVerdict {
    let Some((scheme_str, rest)) = uri.split_once("://") else {
        return UriVerdict::MissingScheme;
    };
    let scheme = match scheme_str {
        "file" => Scheme::File,
        "http" => Scheme::Http,
        "https" => Scheme::Https,
        "apr" => Scheme::Apr,
        _ => return UriVerdict::UnknownScheme,
    };
    if scheme == Scheme::File {
        // file:///abs/path or file://localhost/abs/path. Path must start with /.
        let path = rest.split_once('/').map_or("", |(_, p)| p);
        if path.is_empty() {
            return UriVerdict::FileMustBeAbsolute;
        }
    } else {
        // Remote schemes require a non-empty authority before the first /.
        let host = rest.split('/').next().unwrap_or("");
        if host.is_empty() {
            return UriVerdict::RemoteMissingHost;
        }
    }
    UriVerdict::Ok { scheme }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_mcp_resource_uri_scheme")?;

    for u in [
        "file:///abs/path/model.apr",
        "http://api.example.com/models/v1",
        "apr://meta-llama/llama-3-8b",
        "ftp://wrong/scheme",
        "file://relative",
        "https:///missing-host",
    ] {
        println!("{u:<48}  →  {:?}", validate(u));
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
    fn file_with_absolute_path_passes() {
        assert_eq!(
            validate("file:///abs/path/model.apr"),
            UriVerdict::Ok {
                scheme: Scheme::File
            }
        );
    }

    #[test]
    fn http_with_host_passes() {
        assert_eq!(
            validate("http://api.example.com/x"),
            UriVerdict::Ok {
                scheme: Scheme::Http
            }
        );
    }

    #[test]
    fn apr_with_owner_path_passes() {
        assert_eq!(
            validate("apr://meta-llama/llama-3-8b"),
            UriVerdict::Ok {
                scheme: Scheme::Apr
            }
        );
    }

    #[test]
    fn unknown_scheme_rejected() {
        assert_eq!(validate("ftp://x/y"), UriVerdict::UnknownScheme);
    }

    #[test]
    fn missing_scheme_rejected() {
        assert_eq!(validate("no-scheme-here"), UriVerdict::MissingScheme);
    }

    #[test]
    fn file_relative_rejected() {
        // file://relative has no leading / after the authority.
        assert_eq!(validate("file://relative"), UriVerdict::FileMustBeAbsolute);
    }

    #[test]
    fn https_missing_host_rejected() {
        assert_eq!(
            validate("https:///missing-host"),
            UriVerdict::RemoteMissingHost
        );
    }

    #[test]
    fn file_with_localhost_authority_passes() {
        assert_eq!(
            validate("file://localhost/abs/path"),
            UriVerdict::Ok {
                scheme: Scheme::File
            }
        );
    }
}
