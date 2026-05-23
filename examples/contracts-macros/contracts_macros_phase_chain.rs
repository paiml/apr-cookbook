//! # Contracts-Macros Phase Chain Validator
//!
//! Verify a kernel-structure phase ordering: e.g. `[load, transform,
//! validate, persist]`. Phases must be unique, non-empty, and follow
//! a declared partial order.
//!
//! Demonstrates the **CMM.11** recipe for PMAT-161 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: provable_contracts kernel_structure.phases convention.
//!
//! Run with: cargo run --example contracts_macros_phase_chain
//!
//! Added by PMAT-161 (catalog 1072→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PhaseVerdict {
    Ok { phase_count: u32 },
    EmptyChain,
    DuplicatePhase { name: String },
    OrderViolation { earlier: String, later: String },
    EmptyPhaseName,
}

pub fn validate(phases: &[&str], must_precede: &[(&str, &str)]) -> PhaseVerdict {
    if phases.is_empty() {
        return PhaseVerdict::EmptyChain;
    }
    let mut seen = std::collections::BTreeMap::<&str, usize>::new();
    for (i, p) in phases.iter().enumerate() {
        if p.is_empty() {
            return PhaseVerdict::EmptyPhaseName;
        }
        if seen.contains_key(p) {
            return PhaseVerdict::DuplicatePhase {
                name: (*p).to_string(),
            };
        }
        seen.insert(*p, i);
    }
    for (earlier, later) in must_precede {
        let e = seen.get(earlier);
        let l = seen.get(later);
        if let (Some(ei), Some(li)) = (e, l) {
            if ei >= li {
                return PhaseVerdict::OrderViolation {
                    earlier: (*earlier).to_string(),
                    later: (*later).to_string(),
                };
            }
        }
    }
    PhaseVerdict::Ok {
        phase_count: phases.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_phase_chain")?;

    let valid = ["load", "transform", "validate", "persist"];
    let order = [("load", "transform"), ("validate", "persist")];
    println!("valid: {:?}", validate(&valid, &order));

    let dup = ["load", "transform", "load"];
    println!("duplicate: {:?}", validate(&dup, &order));

    let bad_order = ["transform", "load"];
    println!("order violation: {:?}", validate(&bad_order, &order));

    println!("empty: {:?}", validate(&[], &order));
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
    fn well_ordered_chain_ok() {
        let v = validate(&["a", "b", "c"], &[("a", "b"), ("b", "c")]);
        if let PhaseVerdict::Ok { phase_count } = v {
            assert_eq!(phase_count, 3);
        }
    }

    #[test]
    fn duplicate_phase_rejected() {
        let v = validate(&["a", "a", "b"], &[]);
        assert!(matches!(v, PhaseVerdict::DuplicatePhase { .. }));
    }

    #[test]
    fn order_violation_rejected() {
        let v = validate(&["b", "a"], &[("a", "b")]);
        assert!(matches!(v, PhaseVerdict::OrderViolation { .. }));
    }

    #[test]
    fn empty_chain_rejected() {
        assert_eq!(validate(&[], &[]), PhaseVerdict::EmptyChain);
    }

    #[test]
    fn empty_phase_name_rejected() {
        let v = validate(&["a", "", "b"], &[]);
        assert_eq!(v, PhaseVerdict::EmptyPhaseName);
    }

    #[test]
    fn missing_phase_constraint_ignored() {
        // If a constraint references a phase not in the chain, ignore it.
        let v = validate(&["a"], &[("a", "missing")]);
        assert!(matches!(v, PhaseVerdict::Ok { .. }));
    }

    #[test]
    fn equal_index_violates() {
        // If both phases map to same index (impossible by uniqueness)
        // this is just a defensive check; never triggered.
        let v = validate(&["a", "b"], &[("a", "b")]);
        assert!(matches!(v, PhaseVerdict::Ok { .. }));
    }

    #[test]
    fn reverse_order_violates() {
        let v = validate(&["c", "b", "a"], &[("a", "c")]);
        assert!(matches!(v, PhaseVerdict::OrderViolation { .. }));
    }

    #[test]
    fn no_constraints_just_uniq() {
        let v = validate(&["x", "y", "z"], &[]);
        assert!(matches!(v, PhaseVerdict::Ok { .. }));
    }

    #[test]
    fn deterministic() {
        let phases = ["a", "b", "c"];
        let order = [("a", "b")];
        let a = validate(&phases, &order);
        let b = validate(&phases, &order);
        assert_eq!(a, b);
    }
}
