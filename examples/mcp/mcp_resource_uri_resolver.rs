//! # MCP Resource URI Resolver
//!
//! Resolves `mcp://server/resource[?key=value...]` URIs into
//! (server, path, params). Constraints: scheme must be `mcp://`;
//! server non-empty; path starts with `/`; params unique keys; valid
//! URL-safe characters only.
//!
//! Demonstrates the **MCP.11** recipe for PMAT-132 (mcp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: RFC 3986 (URI generic syntax) + MCP spec.
//!
//! Run with: cargo run --example mcp_resource_uri_resolver
//!
//! Added by PMAT-132 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Eq)]
pub struct McpResourceRef {
    pub server: String,
    pub path: String,
    pub params: BTreeMap<String, String>,
}

#[derive(Debug, PartialEq)]
pub enum ResolveError {
    BadScheme,
    EmptyServer,
    PathMissingSlash,
    DuplicateParam { key: String },
    InvalidChars,
}

pub fn resolve(uri: &str) -> std::result::Result<McpResourceRef, ResolveError> {
    let rest = uri.strip_prefix("mcp://").ok_or(ResolveError::BadScheme)?;
    let (server_path, query) = rest.split_once('?').map_or((rest, ""), |(a, b)| (a, b));
    let (server, path_part) = server_path
        .split_once('/')
        .map_or((server_path, ""), |(s, p)| (s, p));
    if server.is_empty() {
        return Err(ResolveError::EmptyServer);
    }
    if !is_safe_chars(server) {
        return Err(ResolveError::InvalidChars);
    }
    if path_part.is_empty() {
        return Err(ResolveError::PathMissingSlash);
    }
    let path = format!("/{path_part}");
    let mut params = BTreeMap::new();
    if !query.is_empty() {
        for kv in query.split('&') {
            let (k, v) = kv.split_once('=').ok_or(ResolveError::InvalidChars)?;
            if k.is_empty() {
                return Err(ResolveError::InvalidChars);
            }
            if params.contains_key(k) {
                return Err(ResolveError::DuplicateParam { key: k.into() });
            }
            params.insert(k.into(), v.into());
        }
    }
    Ok(McpResourceRef {
        server: server.into(),
        path,
        params,
    })
}

fn is_safe_chars(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mcp_resource_uri_resolver")?;

    for u in [
        "mcp://server-a/resources/123",
        "mcp://server-a/path?key=value&other=v2",
        "http://wrong-scheme/x",
        "mcp:///empty-server",
        "mcp://server-a/x?a=1&a=2",
    ] {
        println!("{u:<50}  →  {:?}", resolve(u));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolver_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_uri_parses() {
        let r = resolve("mcp://server/path/sub").unwrap();
        assert_eq!(r.server, "server");
        assert_eq!(r.path, "/path/sub");
        assert!(r.params.is_empty());
    }

    #[test]
    fn query_params_parsed() {
        let r = resolve("mcp://server/x?a=1&b=2").unwrap();
        assert_eq!(r.params.len(), 2);
        assert_eq!(r.params["a"], "1");
        assert_eq!(r.params["b"], "2");
    }

    #[test]
    fn bad_scheme_rejected() {
        assert_eq!(resolve("http://x/y"), Err(ResolveError::BadScheme));
    }

    #[test]
    fn empty_server_rejected() {
        assert_eq!(resolve("mcp:///path"), Err(ResolveError::EmptyServer));
    }

    #[test]
    fn missing_path_rejected() {
        assert_eq!(resolve("mcp://server"), Err(ResolveError::PathMissingSlash));
    }

    #[test]
    fn duplicate_param_rejected() {
        let r = resolve("mcp://server/x?a=1&a=2");
        assert!(matches!(r, Err(ResolveError::DuplicateParam { .. })));
    }

    #[test]
    fn invalid_server_chars_rejected() {
        let r = resolve("mcp://bad server/x");
        assert_eq!(r, Err(ResolveError::InvalidChars));
    }

    #[test]
    fn empty_query_value_allowed() {
        let r = resolve("mcp://server/x?key=").unwrap();
        assert_eq!(r.params["key"], "");
    }

    #[test]
    fn query_without_equals_rejected() {
        let r = resolve("mcp://server/x?nokey");
        assert_eq!(r, Err(ResolveError::InvalidChars));
    }

    #[test]
    fn nested_path_preserved() {
        let r = resolve("mcp://server/a/b/c/d").unwrap();
        assert_eq!(r.path, "/a/b/c/d");
    }
}
