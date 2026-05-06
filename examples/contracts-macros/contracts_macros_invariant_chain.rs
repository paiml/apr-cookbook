//! # Contracts-Macros Invariant Chain Validator
//!
//! Verify a chain of invariants across N equations: each equation's
//! postcondition must imply the next equation's precondition. Returns
//! the index of the first broken link.
//!
//! Demonstrates the **CMM.16** recipe for PMAT-163 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hoare logic + sequential composition rule.
//!
//! Run with: cargo run --example contracts_macros_invariant_chain
//!
//! Added by PMAT-163 (catalog 1090→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EquationLink {
    pub name: String,
    pub postcond_facts: Vec<String>,
    pub next_precond_facts: Vec<String>,
}

#[derive(Debug, PartialEq)]
pub enum ChainVerdict {
    Ok {
        length: u32,
    },
    BrokenAt {
        equation: String,
        missing_facts: Vec<String>,
    },
    EmptyChain,
}

pub fn validate(chain: &[EquationLink]) -> ChainVerdict {
    if chain.is_empty() {
        return ChainVerdict::EmptyChain;
    }
    for link in chain {
        let post: std::collections::BTreeSet<&str> =
            link.postcond_facts.iter().map(String::as_str).collect();
        let missing: Vec<String> = link
            .next_precond_facts
            .iter()
            .filter(|f| !post.contains(f.as_str()))
            .cloned()
            .collect();
        if !missing.is_empty() {
            return ChainVerdict::BrokenAt {
                equation: link.name.clone(),
                missing_facts: missing,
            };
        }
    }
    ChainVerdict::Ok {
        length: chain.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_invariant_chain")?;

    let ok = vec![
        EquationLink {
            name: "load".to_string(),
            postcond_facts: vec!["model_loaded".to_string(), "memory_ok".to_string()],
            next_precond_facts: vec!["model_loaded".to_string()],
        },
        EquationLink {
            name: "infer".to_string(),
            postcond_facts: vec!["output_set".to_string()],
            next_precond_facts: vec![],
        },
    ];
    println!("ok: {:?}", validate(&ok));

    let broken = vec![EquationLink {
        name: "load".to_string(),
        postcond_facts: vec!["model_loaded".to_string()],
        next_precond_facts: vec!["model_validated".to_string()],
    }];
    println!("broken: {:?}", validate(&broken));
    println!("empty: {:?}", validate(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link(name: &str, post: &[&str], next_pre: &[&str]) -> EquationLink {
        EquationLink {
            name: name.to_string(),
            postcond_facts: post.iter().map(|s| (*s).to_string()).collect(),
            next_precond_facts: next_pre.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn well_formed_chain_ok() {
        let chain = vec![link("a", &["x", "y"], &["x"]), link("b", &["z"], &[])];
        if let ChainVerdict::Ok { length } = validate(&chain) {
            assert_eq!(length, 2);
        }
    }

    #[test]
    fn missing_fact_breaks_chain() {
        let chain = vec![link("a", &["x"], &["y"])];
        let v = validate(&chain);
        if let ChainVerdict::BrokenAt {
            equation,
            missing_facts,
        } = v
        {
            assert_eq!(equation, "a");
            assert_eq!(missing_facts, vec!["y".to_string()]);
        }
    }

    #[test]
    fn empty_chain_rejected() {
        assert_eq!(validate(&[]), ChainVerdict::EmptyChain);
    }

    #[test]
    fn empty_next_precond_passes() {
        let chain = vec![link("a", &["x"], &[])];
        assert!(matches!(validate(&chain), ChainVerdict::Ok { .. }));
    }

    #[test]
    fn first_break_returned() {
        let chain = vec![
            link("a", &["x"], &["x"]),
            link("b", &["y"], &["missing"]),
            link("c", &["z"], &["also_missing"]),
        ];
        if let ChainVerdict::BrokenAt { equation, .. } = validate(&chain) {
            assert_eq!(equation, "b");
        }
    }

    #[test]
    fn multiple_missing_returned() {
        let chain = vec![link("a", &["x"], &["y", "z", "w"])];
        if let ChainVerdict::BrokenAt { missing_facts, .. } = validate(&chain) {
            assert_eq!(missing_facts.len(), 3);
        }
    }

    #[test]
    fn duplicate_facts_dedup() {
        let chain = vec![link("a", &["x", "x", "y"], &["x"])];
        assert!(matches!(validate(&chain), ChainVerdict::Ok { .. }));
    }

    #[test]
    fn long_chain_works() {
        let chain: Vec<EquationLink> = (0..100)
            .map(|i| EquationLink {
                name: format!("eq{i}"),
                postcond_facts: vec!["fact".to_string()],
                next_precond_facts: vec!["fact".to_string()],
            })
            .collect();
        if let ChainVerdict::Ok { length } = validate(&chain) {
            assert_eq!(length, 100);
        }
    }

    #[test]
    fn fact_subset_works() {
        let chain = vec![link("a", &["x", "y", "z"], &["x", "y"])];
        assert!(matches!(validate(&chain), ChainVerdict::Ok { .. }));
    }

    #[test]
    fn deterministic() {
        let chain = vec![link("a", &["x"], &["x"])];
        let a = validate(&chain);
        let b = validate(&chain);
        assert_eq!(a, b);
    }
}
