//! # apr publish — Parent Chain Termination (FALSIFY-PM-006)
//!
//! Every published model must declare a `provenance.parent` reference. The
//! publish manifest contract enforces that the parent chain TERMINATES at
//! some upstream root (e.g. `Qwen/Qwen2.5-Coder-7B-Instruct` from
//! HuggingFace) within a bounded number of hops — no cycles, no infinite
//! ancestry.
//!
//! This recipe walks a synthetic publish chain (child → parent → grandparent
//! → root) and asserts termination within the configured max-depth, plus
//! detects cycles.
//!
//! Demonstrates the **CLI+.5** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: publish-manifest-v1.yaml v1.1.0 FALSIFY-PM-006 (parent chain termination)
//!
//! Run with: cargo run --example cli_publish_parent_chain_termination
//!
//! Added by PMAT-076 (expand-cookbooks: apr publish end-to-end).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::{HashMap, HashSet};

const MAX_CHAIN_DEPTH: usize = 16;

/// Walk the parent chain starting from `start` through the lookup table.
/// Returns the chain (terminating with the root that has no parent) or an
/// error if it exceeds MAX_CHAIN_DEPTH or contains a cycle.
fn walk_parent_chain(parents: &HashMap<&str, &str>, start: &str) -> Result<Vec<String>> {
    let mut chain = vec![start.to_string()];
    let mut seen = HashSet::new();
    seen.insert(start.to_string());

    let mut current = start;
    loop {
        match parents.get(current) {
            Some(parent) => {
                if seen.contains(*parent) {
                    return Err(apr_cookbook::CookbookError::Validation(format!(
                        "FALSIFY-PM-006: cycle detected at {parent} (chain so far: {chain:?})"
                    )));
                }
                chain.push((*parent).to_string());
                if chain.len() > MAX_CHAIN_DEPTH {
                    return Err(apr_cookbook::CookbookError::Validation(format!(
                        "FALSIFY-PM-006: parent chain exceeds max depth {MAX_CHAIN_DEPTH}"
                    )));
                }
                seen.insert((*parent).to_string());
                current = parent;
            }
            None => return Ok(chain), // reached root
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_publish_parent_chain_termination")?;

    // Synthetic chain: distilled_v3 → distilled_v2 → distilled_v1 → upstream_root
    let parents: HashMap<&str, &str> = HashMap::from([
        ("paiml/coder-distilled-v3", "paiml/coder-distilled-v2"),
        ("paiml/coder-distilled-v2", "paiml/coder-distilled-v1"),
        ("paiml/coder-distilled-v1", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    ]);

    let chain = walk_parent_chain(&parents, "paiml/coder-distilled-v3")?;
    println!(
        "FALSIFY-PM-006: parent chain terminates at root after {} hops:",
        chain.len()
    );
    for (i, model) in chain.iter().enumerate() {
        println!("  {i}: {model}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn straight_chain_terminates() {
        let parents = HashMap::from([
            ("a", "b"),
            ("b", "c"),
            ("c", "d"), // d has no parent → root
        ]);
        let chain = walk_parent_chain(&parents, "a").unwrap();
        assert_eq!(chain, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn cycle_rejected() {
        let parents = HashMap::from([
            ("a", "b"),
            ("b", "c"),
            ("c", "a"), // cycle
        ]);
        let err = walk_parent_chain(&parents, "a");
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("cycle detected"), "unexpected error: {msg}");
    }

    #[test]
    fn excessive_depth_rejected() {
        // Build a 20-deep chain (> MAX_CHAIN_DEPTH=16).
        let chain: Vec<String> = (0..20).map(|i| format!("model-{i}")).collect();
        let parents: HashMap<&str, &str> = chain
            .windows(2)
            .map(|w| (w[0].as_str(), w[1].as_str()))
            .collect();
        let err = walk_parent_chain(&parents, &chain[0]);
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("max depth"), "unexpected error: {msg}");
    }

    #[test]
    fn root_with_no_parent_returns_single_element_chain() {
        let parents: HashMap<&str, &str> = HashMap::new();
        let chain = walk_parent_chain(&parents, "lonely-root").unwrap();
        assert_eq!(chain, vec!["lonely-root"]);
    }
}
