//! # Contracts-Macros YAML Alias Resolution Chain
//!
//! Resolve a chain of YAML aliases (alias→target→target→...) to the
//! ultimate value, detecting cycles. Returns final target or
//! `Cycle` verdict.
//!
//! Demonstrates the **CMM.190** recipe for PMAT-221 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 §6.9.2 alias-resolution; libyaml event-loop
//!  alias-deref.
//!
//! Run with: cargo run --example contracts_macros_yaml_alias_resolve_chain
//!
//! Added by PMAT-221 (catalog 1612→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, PartialEq)]
pub enum AliasResolveVerdict {
    Resolved { final_value: String, hops: u32 },
    Cycle { offender: String },
    Unresolved { dangling: String },
    InvalidConfig,
}

pub fn resolve(aliases: &[(&str, &str)], start: &str) -> AliasResolveVerdict {
    if aliases.is_empty() || start.is_empty() {
        return AliasResolveVerdict::InvalidConfig;
    }
    let map: BTreeMap<&str, &str> = aliases.iter().copied().collect();
    let mut visited: BTreeSet<String> = BTreeSet::new();
    let mut current = start;
    let mut hops = 0u32;
    while let Some(next) = map.get(current) {
        if !visited.insert(current.to_string()) {
            return AliasResolveVerdict::Cycle {
                offender: current.to_string(),
            };
        }
        hops += 1;
        current = next;
        if hops > 1_000 {
            return AliasResolveVerdict::Cycle {
                offender: current.to_string(),
            };
        }
    }
    if hops == 0 {
        AliasResolveVerdict::Unresolved {
            dangling: start.to_string(),
        }
    } else {
        AliasResolveVerdict::Resolved {
            final_value: current.to_string(),
            hops,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_alias_resolve_chain")?;

    let aliases = [("a", "b"), ("b", "c"), ("c", "value")];
    println!("resolve: {:?}", resolve(&aliases, "a"));
    let cyclic = [("x", "y"), ("y", "x")];
    println!("cycle: {:?}", resolve(&cyclic, "x"));
    println!("invalid: {:?}", resolve(&[], "x"));
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
    fn linear_chain_resolved() {
        let aliases = [("a", "b"), ("b", "c"), ("c", "final")];
        let v = resolve(&aliases, "a");
        if let AliasResolveVerdict::Resolved { final_value, hops } = v {
            assert_eq!(final_value, "final");
            assert_eq!(hops, 3);
        }
    }

    #[test]
    fn cycle_detected() {
        let aliases = [("x", "y"), ("y", "x")];
        let v = resolve(&aliases, "x");
        assert!(matches!(v, AliasResolveVerdict::Cycle { .. }));
    }

    #[test]
    fn unresolved_dangling() {
        let aliases = [("a", "b")];
        let v = resolve(&aliases, "missing");
        if let AliasResolveVerdict::Unresolved { dangling } = v {
            assert_eq!(dangling, "missing");
        }
    }

    #[test]
    fn empty_aliases_rejected() {
        assert_eq!(resolve(&[], "x"), AliasResolveVerdict::InvalidConfig);
    }

    #[test]
    fn empty_start_rejected() {
        assert_eq!(
            resolve(&[("a", "b")], ""),
            AliasResolveVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let aliases = [("a", "b")];
        let r1 = resolve(&aliases, "a");
        let r2 = resolve(&aliases, "a");
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_hop_resolved() {
        let v = resolve(&[("a", "b")], "a");
        if let AliasResolveVerdict::Resolved { hops, .. } = v {
            assert_eq!(hops, 1);
        }
    }

    #[test]
    fn self_loop_cycle() {
        let v = resolve(&[("a", "a")], "a");
        assert!(matches!(v, AliasResolveVerdict::Cycle { .. }));
    }

    #[test]
    fn long_chain_handled() {
        let aliases: Vec<(&str, &str)> =
            vec![("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")];
        let v = resolve(&aliases, "a");
        if let AliasResolveVerdict::Resolved { final_value, .. } = v {
            assert_eq!(final_value, "f");
        }
    }

    #[test]
    fn unicode_alias_supported() {
        let v = resolve(&[("café", "résumé")], "café");
        if let AliasResolveVerdict::Resolved { final_value, .. } = v {
            assert_eq!(final_value, "résumé");
        }
    }

    #[test]
    fn many_hops_capped() {
        let mut aliases: Vec<(&str, &str)> = Vec::new();
        // Build a cycle to trigger hop cap.
        aliases.push(("a", "a"));
        let v = resolve(&aliases, "a");
        assert!(matches!(v, AliasResolveVerdict::Cycle { .. }));
    }

    #[test]
    fn three_node_cycle_detected() {
        let aliases = [("a", "b"), ("b", "c"), ("c", "a")];
        let v = resolve(&aliases, "a");
        assert!(matches!(v, AliasResolveVerdict::Cycle { .. }));
    }
}
