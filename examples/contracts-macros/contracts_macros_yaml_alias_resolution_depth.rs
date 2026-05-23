//! # Contracts-Macros YAML Alias Resolution Depth
//!
//! Measure the maximum depth of YAML anchor/alias chains. A depth
//! greater than `max_safe_depth` is flagged for refactoring.
//!
//! Demonstrates the **CMM.98** recipe for PMAT-190 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 spec §6.9.2 (alias node); compiler-style
//!  reference-graph depth measurement.
//!
//! Run with: cargo run --example contracts_macros_yaml_alias_resolution_depth
//!
//! Added by PMAT-190 (catalog 1333→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum DepthVerdict {
    Ok {
        max_depth: u32,
        flagged: Vec<String>,
    },
    InvalidConfig,
}

pub fn measure(aliases: &[(&str, &str)], max_safe_depth: u32) -> DepthVerdict {
    if aliases.is_empty() || max_safe_depth == 0 {
        return DepthVerdict::InvalidConfig;
    }
    let mut graph: BTreeMap<String, String> = BTreeMap::new();
    for (alias, target) in aliases {
        graph.insert((*alias).to_string(), (*target).to_string());
    }
    let mut max_depth = 0u32;
    let mut flagged: Vec<String> = Vec::new();
    for alias in graph.keys() {
        let depth = chain_depth(alias, &graph, max_safe_depth + 5);
        if depth > max_depth {
            max_depth = depth;
        }
        if depth > max_safe_depth {
            flagged.push(alias.clone());
        }
    }
    flagged.sort();
    flagged.dedup();
    DepthVerdict::Ok { max_depth, flagged }
}

fn chain_depth(start: &str, graph: &BTreeMap<String, String>, cap: u32) -> u32 {
    let mut depth = 0u32;
    let mut current = start.to_string();
    while depth < cap {
        if let Some(next) = graph.get(&current) {
            depth += 1;
            current = next.clone();
        } else {
            break;
        }
    }
    depth
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_alias_resolution_depth")?;

    let aliases = [("a", "b"), ("b", "c"), ("c", "d")];
    println!("audit: {:?}", measure(&aliases, 2));
    println!("invalid: {:?}", measure(&[], 5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measurer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn shallow_chain_no_flags() {
        let aliases = [("a", "b")];
        let v = measure(&aliases, 5);
        if let DepthVerdict::Ok { flagged, .. } = v {
            assert!(flagged.is_empty());
        }
    }

    #[test]
    fn deep_chain_flagged() {
        let aliases = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")];
        let v = measure(&aliases, 2);
        if let DepthVerdict::Ok { flagged, .. } = v {
            assert!(!flagged.is_empty());
        }
    }

    #[test]
    fn empty_aliases_rejected() {
        assert_eq!(measure(&[], 5), DepthVerdict::InvalidConfig);
    }

    #[test]
    fn zero_max_safe_rejected() {
        let aliases = [("a", "b")];
        assert_eq!(measure(&aliases, 0), DepthVerdict::InvalidConfig);
    }

    #[test]
    fn max_depth_correct() {
        let aliases = [("a", "b"), ("b", "c")];
        let v = measure(&aliases, 5);
        if let DepthVerdict::Ok { max_depth, .. } = v {
            // Chain a→b→c; from "a" depth=2, from "b" depth=1.
            assert_eq!(max_depth, 2);
        }
    }

    #[test]
    fn deterministic() {
        let aliases = [("a", "b")];
        let r1 = measure(&aliases, 5);
        let r2 = measure(&aliases, 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn flagged_sorted() {
        let aliases = [("zeta", "x"), ("zeta-2", "zeta"), ("alpha", "zeta-2")];
        let v = measure(&aliases, 1);
        if let DepthVerdict::Ok { flagged, .. } = v {
            assert_eq!(flagged[0], "alpha");
        }
    }

    #[test]
    fn boundary_at_max_safe_no_flag() {
        let aliases = [("a", "b"), ("b", "c")];
        let v = measure(&aliases, 2);
        if let DepthVerdict::Ok { flagged, .. } = v {
            // depth from "a" = 2 → at limit but not over.
            assert!(flagged.is_empty());
        }
    }

    #[test]
    fn one_over_limit_flagged() {
        let aliases = [("a", "b"), ("b", "c"), ("c", "d")];
        let v = measure(&aliases, 2);
        if let DepthVerdict::Ok { flagged, .. } = v {
            assert!(flagged.contains(&"a".to_string()));
        }
    }

    #[test]
    fn cycle_capped_by_cap() {
        let aliases = [("a", "b"), ("b", "a")];
        let v = measure(&aliases, 5);
        if let DepthVerdict::Ok { max_depth, .. } = v {
            // Chain caps at max_safe_depth + 5 = 10.
            assert!(max_depth <= 10);
        }
    }

    #[test]
    fn single_alias_depth_one() {
        let aliases = [("a", "b")];
        let v = measure(&aliases, 5);
        if let DepthVerdict::Ok { max_depth, .. } = v {
            assert_eq!(max_depth, 1);
        }
    }
}
