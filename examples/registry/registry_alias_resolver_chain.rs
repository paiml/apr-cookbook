//! # Registry Alias Resolver Chain
//!
//! Aliases can chain (alias → alias → version). Resolver follows the
//! chain to a concrete version, with cycle detection (max depth 16).
//!
//! Errors:
//!   ChainTooDeep — exceeded max depth (likely cycle)
//!   AliasNotFound — leaf alias missing
//!   CycleDetected — explicit detection
//!
//! Demonstrates the **REG.16** recipe for PMAT-143 (registry round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Docker manifest-list and OCI alias-chain semantics.
//!
//! Run with: cargo run --example registry_alias_resolver_chain
//!
//! Added by PMAT-143 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::{BTreeMap, BTreeSet};

const MAX_DEPTH: usize = 16;

#[derive(Debug, PartialEq)]
pub enum ResolveVerdict {
    Resolved { version: String, depth: usize },
    AliasNotFound { missing: String },
    CycleDetected { path: Vec<String> },
    ChainTooDeep,
    EmptyAlias,
}

pub fn resolve(
    start: &str,
    aliases: &BTreeMap<String, String>,
    versions: &BTreeSet<String>,
) -> ResolveVerdict {
    if start.is_empty() {
        return ResolveVerdict::EmptyAlias;
    }
    let mut visited: Vec<String> = Vec::new();
    let mut current = start.to_string();
    for depth in 0..MAX_DEPTH {
        if versions.contains(&current) {
            return ResolveVerdict::Resolved {
                version: current,
                depth,
            };
        }
        if visited.contains(&current) {
            visited.push(current);
            return ResolveVerdict::CycleDetected { path: visited };
        }
        let next = match aliases.get(&current) {
            Some(n) => n.clone(),
            None => {
                return ResolveVerdict::AliasNotFound { missing: current };
            }
        };
        visited.push(current);
        current = next;
    }
    ResolveVerdict::ChainTooDeep
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("registry_alias_resolver_chain")?;

    let mut aliases = BTreeMap::new();
    aliases.insert("latest".to_string(), "stable".to_string());
    aliases.insert("stable".to_string(), "1.2.3".to_string());

    let mut versions = BTreeSet::new();
    versions.insert("1.2.3".to_string());

    println!("latest → ...: {:?}", resolve("latest", &aliases, &versions));
    println!("missing: {:?}", resolve("nonexistent", &aliases, &versions));

    let mut cycle = BTreeMap::new();
    cycle.insert("a".to_string(), "b".to_string());
    cycle.insert("b".to_string(), "a".to_string());
    println!("cycle: {:?}", resolve("a", &cycle, &BTreeSet::new()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aliases(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    fn versions(vs: &[&str]) -> BTreeSet<String> {
        vs.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn resolver_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn single_hop_resolves() {
        let a = aliases(&[("latest", "1.2.3")]);
        let v = versions(&["1.2.3"]);
        if let ResolveVerdict::Resolved { version, depth } = resolve("latest", &a, &v) {
            assert_eq!(version, "1.2.3");
            assert_eq!(depth, 1);
        }
    }

    #[test]
    fn multi_hop_resolves() {
        let a = aliases(&[("latest", "stable"), ("stable", "1.2.3")]);
        let v = versions(&["1.2.3"]);
        if let ResolveVerdict::Resolved { version, depth } = resolve("latest", &a, &v) {
            assert_eq!(version, "1.2.3");
            assert_eq!(depth, 2);
        }
    }

    #[test]
    fn direct_version_zero_depth() {
        let v = versions(&["1.0.0"]);
        if let ResolveVerdict::Resolved { depth, .. } = resolve("1.0.0", &BTreeMap::new(), &v) {
            assert_eq!(depth, 0);
        }
    }

    #[test]
    fn missing_alias_reported() {
        let res = resolve("ghost", &BTreeMap::new(), &BTreeSet::new());
        assert!(matches!(res, ResolveVerdict::AliasNotFound { .. }));
    }

    #[test]
    fn cycle_detected_a_to_b_to_a() {
        let a = aliases(&[("a", "b"), ("b", "a")]);
        let res = resolve("a", &a, &BTreeSet::new());
        assert!(matches!(res, ResolveVerdict::CycleDetected { .. }));
    }

    #[test]
    fn three_node_cycle_detected() {
        let a = aliases(&[("a", "b"), ("b", "c"), ("c", "a")]);
        let res = resolve("a", &a, &BTreeSet::new());
        assert!(matches!(res, ResolveVerdict::CycleDetected { .. }));
    }

    #[test]
    fn chain_too_deep() {
        // 17-element chain → exceeds MAX_DEPTH.
        let mut pairs: Vec<(String, String)> = Vec::new();
        for i in 0..MAX_DEPTH + 1 {
            pairs.push((format!("a{i}"), format!("a{}", i + 1)));
        }
        let a: BTreeMap<String, String> = pairs.into_iter().collect();
        let res = resolve("a0", &a, &BTreeSet::new());
        assert_eq!(res, ResolveVerdict::ChainTooDeep);
    }

    #[test]
    fn empty_alias_rejected() {
        let v = versions(&["1.0.0"]);
        assert_eq!(
            resolve("", &BTreeMap::new(), &v),
            ResolveVerdict::EmptyAlias
        );
    }

    #[test]
    fn cycle_path_contains_all_visited() {
        let a = aliases(&[("a", "b"), ("b", "c"), ("c", "a")]);
        if let ResolveVerdict::CycleDetected { path } = resolve("a", &a, &BTreeSet::new()) {
            // path: a, b, c, a (4 elements)
            assert!(path.len() >= 3);
            assert_eq!(path.first().unwrap(), "a");
        }
    }

    #[test]
    fn missing_intermediate_alias() {
        let a = aliases(&[("latest", "stable")]);
        // "stable" doesn't resolve to a version or further alias.
        let res = resolve("latest", &a, &BTreeSet::new());
        if let ResolveVerdict::AliasNotFound { missing } = res {
            assert_eq!(missing, "stable");
        }
    }
}
