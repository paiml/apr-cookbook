//! # Contracts-Macros Lean Status Provenance
//!
//! Each obligation in a `.yaml` contract has a status: `proved`,
//! `wip`, `sorry`, or `not-applicable`. This recipe rolls a list of
//! per-obligation statuses up to a contract-level verdict.
//!
//! Demonstrates the **CMM.09** recipe for PMAT-160 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Lean 4 sorry-tracking + provable_contracts spec.
//!
//! Run with: cargo run --example contracts_macros_status_provenance
//!
//! Added by PMAT-160 (catalog 1063→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeanStatus {
    Proved,
    Wip,
    Sorry,
    NotApplicable,
}

#[derive(Debug, PartialEq)]
pub enum ProvenanceVerdict {
    AllProved,
    PartiallyProved {
        proved: u32,
        wip: u32,
        sorry: u32,
        na: u32,
    },
    HasSorry {
        sorry: u32,
    },
    EmptyContract,
}

pub fn classify(statuses: &[LeanStatus]) -> ProvenanceVerdict {
    if statuses.is_empty() {
        return ProvenanceVerdict::EmptyContract;
    }
    let mut proved = 0u32;
    let mut wip = 0u32;
    let mut sorry = 0u32;
    let mut na = 0u32;
    for s in statuses {
        match s {
            LeanStatus::Proved => proved += 1,
            LeanStatus::Wip => wip += 1,
            LeanStatus::Sorry => sorry += 1,
            LeanStatus::NotApplicable => na += 1,
        }
    }
    if sorry > 0 {
        return ProvenanceVerdict::HasSorry { sorry };
    }
    if wip == 0 && (proved + na) == statuses.len() as u32 {
        return ProvenanceVerdict::AllProved;
    }
    ProvenanceVerdict::PartiallyProved {
        proved,
        wip,
        sorry,
        na,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_status_provenance")?;

    println!(
        "all proved: {:?}",
        classify(&[
            LeanStatus::Proved,
            LeanStatus::Proved,
            LeanStatus::NotApplicable
        ])
    );
    println!(
        "wip mix: {:?}",
        classify(&[LeanStatus::Proved, LeanStatus::Wip, LeanStatus::Wip])
    );
    println!(
        "has sorry: {:?}",
        classify(&[LeanStatus::Proved, LeanStatus::Sorry])
    );
    println!("empty: {:?}", classify(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn all_proved_recognized() {
        let v = classify(&[
            LeanStatus::Proved,
            LeanStatus::Proved,
            LeanStatus::NotApplicable,
        ]);
        assert_eq!(v, ProvenanceVerdict::AllProved);
    }

    #[test]
    fn sorry_takes_precedence() {
        let v = classify(&[LeanStatus::Proved, LeanStatus::Sorry]);
        assert!(matches!(v, ProvenanceVerdict::HasSorry { .. }));
    }

    #[test]
    fn wip_returns_partial() {
        let v = classify(&[LeanStatus::Proved, LeanStatus::Wip]);
        assert!(matches!(v, ProvenanceVerdict::PartiallyProved { .. }));
    }

    #[test]
    fn empty_contract_rejected() {
        assert_eq!(classify(&[]), ProvenanceVerdict::EmptyContract);
    }

    #[test]
    fn only_na_all_proved() {
        let v = classify(&[LeanStatus::NotApplicable, LeanStatus::NotApplicable]);
        // No theorems exist; trivially "proved" since none open.
        assert_eq!(v, ProvenanceVerdict::AllProved);
    }

    #[test]
    fn count_correct() {
        let v = classify(&[
            LeanStatus::Proved,
            LeanStatus::Proved,
            LeanStatus::Wip,
            LeanStatus::NotApplicable,
        ]);
        if let ProvenanceVerdict::PartiallyProved {
            proved,
            wip,
            sorry,
            na,
        } = v
        {
            assert_eq!(proved, 2);
            assert_eq!(wip, 1);
            assert_eq!(sorry, 0);
            assert_eq!(na, 1);
        }
    }

    #[test]
    fn single_proved_all_proved() {
        let v = classify(&[LeanStatus::Proved]);
        assert_eq!(v, ProvenanceVerdict::AllProved);
    }

    #[test]
    fn single_wip_partial() {
        let v = classify(&[LeanStatus::Wip]);
        assert!(matches!(v, ProvenanceVerdict::PartiallyProved { .. }));
    }

    #[test]
    fn single_sorry_has_sorry() {
        let v = classify(&[LeanStatus::Sorry]);
        if let ProvenanceVerdict::HasSorry { sorry } = v {
            assert_eq!(sorry, 1);
        }
    }

    #[test]
    fn deterministic() {
        let s = vec![LeanStatus::Proved, LeanStatus::Wip];
        let a = classify(&s);
        let b = classify(&s);
        assert_eq!(a, b);
    }
}
