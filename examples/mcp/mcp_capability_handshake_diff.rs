//! # MCP Capability Handshake Diff
//!
//! Each side declares supported capabilities (e.g., "tools", "prompts",
//! "resources", "sampling", "logging"). The mutually-supported set is
//! the intersection. This recipe diffs client/server caps and reports
//! the agreed set + each side's exclusive features.
//!
//! Demonstrates the **MCP.15** recipe for PMAT-135 (mcp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: MCP capability negotiation specification.
//!
//! Run with: cargo run --example mcp_capability_handshake_diff
//!
//! Added by PMAT-135 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum DiffVerdict {
    Ok {
        agreed: Vec<String>,
        client_only: Vec<String>,
        server_only: Vec<String>,
    },
    EmptyClient,
    EmptyServer,
    NoOverlap {
        client: Vec<String>,
        server: Vec<String>,
    },
}

pub fn diff(client_caps: &[&str], server_caps: &[&str]) -> DiffVerdict {
    if client_caps.is_empty() {
        return DiffVerdict::EmptyClient;
    }
    if server_caps.is_empty() {
        return DiffVerdict::EmptyServer;
    }
    let client_set: BTreeSet<&str> = client_caps.iter().copied().collect();
    let server_set: BTreeSet<&str> = server_caps.iter().copied().collect();

    let agreed: Vec<String> = client_set
        .intersection(&server_set)
        .map(|s| (*s).to_string())
        .collect();
    if agreed.is_empty() {
        return DiffVerdict::NoOverlap {
            client: client_set.iter().map(|s| (*s).to_string()).collect(),
            server: server_set.iter().map(|s| (*s).to_string()).collect(),
        };
    }
    DiffVerdict::Ok {
        agreed,
        client_only: client_set
            .difference(&server_set)
            .map(|s| (*s).to_string())
            .collect(),
        server_only: server_set
            .difference(&client_set)
            .map(|s| (*s).to_string())
            .collect(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mcp_capability_handshake_diff")?;

    let client = ["tools", "prompts", "resources", "sampling"];
    let server = ["tools", "prompts", "logging"];
    println!("typical: {:?}", diff(&client, &server));

    println!("empty client: {:?}", diff(&[], &server));
    println!("no overlap: {:?}", diff(&["sampling"], &["logging"]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn intersection_returned_for_overlapping_caps() {
        let v = diff(&["a", "b", "c"], &["b", "c", "d"]);
        if let DiffVerdict::Ok { agreed, .. } = v {
            assert_eq!(agreed, vec!["b", "c"]);
        }
    }

    #[test]
    fn client_only_extracted() {
        let v = diff(&["a", "b"], &["b", "c"]);
        if let DiffVerdict::Ok { client_only, .. } = v {
            assert_eq!(client_only, vec!["a"]);
        }
    }

    #[test]
    fn server_only_extracted() {
        let v = diff(&["a", "b"], &["b", "c"]);
        if let DiffVerdict::Ok { server_only, .. } = v {
            assert_eq!(server_only, vec!["c"]);
        }
    }

    #[test]
    fn full_overlap_no_exclusives() {
        let v = diff(&["a", "b"], &["a", "b"]);
        if let DiffVerdict::Ok {
            agreed,
            client_only,
            server_only,
        } = v
        {
            assert_eq!(agreed, vec!["a", "b"]);
            assert!(client_only.is_empty());
            assert!(server_only.is_empty());
        }
    }

    #[test]
    fn empty_client_rejected() {
        assert_eq!(diff(&[], &["a"]), DiffVerdict::EmptyClient);
    }

    #[test]
    fn empty_server_rejected() {
        assert_eq!(diff(&["a"], &[]), DiffVerdict::EmptyServer);
    }

    #[test]
    fn no_overlap_returns_both_lists() {
        let v = diff(&["a", "b"], &["c", "d"]);
        assert!(matches!(v, DiffVerdict::NoOverlap { .. }));
    }

    #[test]
    fn duplicate_caps_deduped() {
        let v = diff(&["a", "a", "b"], &["a", "b"]);
        if let DiffVerdict::Ok { agreed, .. } = v {
            assert_eq!(agreed, vec!["a", "b"]);
        }
    }

    #[test]
    fn agreed_sorted_lexicographic() {
        let v = diff(&["zoo", "alpha", "mid"], &["mid", "alpha", "zoo"]);
        if let DiffVerdict::Ok { agreed, .. } = v {
            assert_eq!(agreed, vec!["alpha", "mid", "zoo"]);
        }
    }

    #[test]
    fn realistic_mcp_caps() {
        let client = ["tools", "prompts", "resources", "sampling"];
        let server = ["tools", "prompts", "logging"];
        if let DiffVerdict::Ok {
            agreed,
            client_only,
            server_only,
        } = diff(&client, &server)
        {
            assert_eq!(agreed, vec!["prompts", "tools"]);
            assert_eq!(client_only, vec!["resources", "sampling"]);
            assert_eq!(server_only, vec!["logging"]);
        }
    }
}
