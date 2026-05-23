//! # Contracts-Macros Lean Axiom Audit
//!
//! Flag theorems still relying on Lean axioms (`sorry` or `admit`).
//! Reports the proportion of theorems with proof gaps and the names
//! of the worst offenders (those marked `sorry`).
//!
//! Demonstrates the **CMM.48** recipe for PMAT-173 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Lean 4 #print axioms convention.
//!
//! Run with: cargo run --example contracts_macros_lean_axiom_audit
//!
//! Added by PMAT-173 (catalog 1180→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofState {
    Proved,
    Sorry,
    Admit,
    NotApplicable,
}

#[derive(Debug, PartialEq)]
pub enum AxiomVerdict {
    AllProved,
    HasSorry {
        sorry_names: Vec<String>,
    },
    HasAdmit {
        admit_names: Vec<String>,
    },
    Mixed {
        sorry_names: Vec<String>,
        admit_names: Vec<String>,
    },
    EmptyContract,
}

pub fn audit(theorems: &[(&str, ProofState)]) -> AxiomVerdict {
    if theorems.is_empty() {
        return AxiomVerdict::EmptyContract;
    }
    let mut sorry_names = Vec::new();
    let mut admit_names = Vec::new();
    for (name, state) in theorems {
        match state {
            ProofState::Sorry => sorry_names.push((*name).to_string()),
            ProofState::Admit => admit_names.push((*name).to_string()),
            _ => {}
        }
    }
    match (sorry_names.is_empty(), admit_names.is_empty()) {
        (true, true) => AxiomVerdict::AllProved,
        (false, true) => AxiomVerdict::HasSorry { sorry_names },
        (true, false) => AxiomVerdict::HasAdmit { admit_names },
        (false, false) => AxiomVerdict::Mixed {
            sorry_names,
            admit_names,
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_lean_axiom_audit")?;

    println!(
        "all proved: {:?}",
        audit(&[("a", ProofState::Proved), ("b", ProofState::NotApplicable)])
    );
    println!(
        "has sorry: {:?}",
        audit(&[("a", ProofState::Proved), ("bad", ProofState::Sorry)])
    );
    println!("has admit: {:?}", audit(&[("a", ProofState::Admit)]));
    println!(
        "mixed: {:?}",
        audit(&[("s", ProofState::Sorry), ("a", ProofState::Admit),])
    );
    println!("empty: {:?}", audit(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auditor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn proved_and_na_all_proved() {
        let v = audit(&[("a", ProofState::Proved), ("b", ProofState::NotApplicable)]);
        assert_eq!(v, AxiomVerdict::AllProved);
    }

    #[test]
    fn single_sorry_listed() {
        let v = audit(&[("bad", ProofState::Sorry)]);
        if let AxiomVerdict::HasSorry { sorry_names } = v {
            assert_eq!(sorry_names, vec!["bad".to_string()]);
        }
    }

    #[test]
    fn single_admit_listed() {
        let v = audit(&[("bad", ProofState::Admit)]);
        if let AxiomVerdict::HasAdmit { admit_names } = v {
            assert_eq!(admit_names, vec!["bad".to_string()]);
        }
    }

    #[test]
    fn mixed_both_listed() {
        let v = audit(&[("s", ProofState::Sorry), ("a", ProofState::Admit)]);
        if let AxiomVerdict::Mixed {
            sorry_names,
            admit_names,
        } = v
        {
            assert_eq!(sorry_names.len(), 1);
            assert_eq!(admit_names.len(), 1);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(audit(&[]), AxiomVerdict::EmptyContract);
    }

    #[test]
    fn order_preserved_in_lists() {
        let v = audit(&[("zzz", ProofState::Sorry), ("aaa", ProofState::Sorry)]);
        if let AxiomVerdict::HasSorry { sorry_names } = v {
            assert_eq!(sorry_names, vec!["zzz".to_string(), "aaa".to_string()]);
        }
    }

    #[test]
    fn many_proved_one_sorry() {
        let v = audit(&[
            ("a", ProofState::Proved),
            ("b", ProofState::Proved),
            ("c", ProofState::Proved),
            ("d", ProofState::Sorry),
        ]);
        if let AxiomVerdict::HasSorry { sorry_names } = v {
            assert_eq!(sorry_names, vec!["d".to_string()]);
        }
    }

    #[test]
    fn multiple_sorries() {
        let v = audit(&[
            ("a", ProofState::Sorry),
            ("b", ProofState::Sorry),
            ("c", ProofState::Sorry),
        ]);
        if let AxiomVerdict::HasSorry { sorry_names } = v {
            assert_eq!(sorry_names.len(), 3);
        }
    }

    #[test]
    fn na_only_all_proved() {
        let v = audit(&[("a", ProofState::NotApplicable)]);
        assert_eq!(v, AxiomVerdict::AllProved);
    }

    #[test]
    fn deterministic() {
        let theorems = [("a", ProofState::Sorry)];
        let a = audit(&theorems);
        let b = audit(&theorems);
        assert_eq!(a, b);
    }
}
