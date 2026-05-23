//! # apr qualify — Tier Progression (smoke → standard → full)
//!
//! `apr qualify <FILE> --tier {smoke,standard,full}` runs an increasing
//! battery of cross-subcommand checks. Each tier is a strict superset of
//! the previous one — a model that passes `--tier full` MUST pass
//! `--tier smoke`. This recipe documents the gate ladder and asserts
//! the inclusion property as a contract test.
//!
//! Demonstrates the **QUALIFY.3** recipe for PMAT-094 (apr qualify coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender QUALIFY-001
//!
//! Run with: cargo run --example cli_qualify_tier_progression
//!
//! Added by PMAT-094 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Smoke,
    Standard,
    Full,
}

pub fn gates_for_tier(tier: Tier) -> Vec<&'static str> {
    let smoke = vec!["info", "tensors", "tree", "validate"];
    let standard_addons = vec!["check", "bench", "qa"];
    let full_addons = vec!["parity", "compare-hf", "probar", "playbook"];
    match tier {
        Tier::Smoke => smoke,
        Tier::Standard => {
            let mut g = smoke;
            g.extend(standard_addons);
            g
        }
        Tier::Full => {
            let mut g = smoke;
            g.extend(standard_addons);
            g.extend(full_addons);
            g
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_qualify_tier_progression")?;

    for tier in [Tier::Smoke, Tier::Standard, Tier::Full] {
        let gates = gates_for_tier(tier);
        println!("{tier:?}:  {} gates  →  {gates:?}", gates.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn smoke_is_smallest() {
        assert!(gates_for_tier(Tier::Smoke).len() < gates_for_tier(Tier::Standard).len());
    }

    #[test]
    fn full_is_largest() {
        assert!(gates_for_tier(Tier::Full).len() > gates_for_tier(Tier::Standard).len());
    }

    #[test]
    fn standard_is_superset_of_smoke() {
        let smoke = gates_for_tier(Tier::Smoke);
        let standard = gates_for_tier(Tier::Standard);
        for g in &smoke {
            assert!(
                standard.contains(g),
                "smoke gate {g:?} missing from standard"
            );
        }
    }

    #[test]
    fn full_is_superset_of_standard() {
        let standard = gates_for_tier(Tier::Standard);
        let full = gates_for_tier(Tier::Full);
        for g in &standard {
            assert!(full.contains(g), "standard gate {g:?} missing from full");
        }
    }

    #[test]
    fn each_gate_appears_at_most_once_per_tier() {
        // Idempotent gate set — duplicates would confuse CI dashboards.
        for tier in [Tier::Smoke, Tier::Standard, Tier::Full] {
            let gates = gates_for_tier(tier);
            let unique: std::collections::HashSet<_> = gates.iter().collect();
            assert_eq!(unique.len(), gates.len(), "duplicates in {tier:?}");
        }
    }
}
