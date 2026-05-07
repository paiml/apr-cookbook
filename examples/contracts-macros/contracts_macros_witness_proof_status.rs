//! # Contracts-Macros Witness Proof Status
//!
//! Aggregate witness proof statuses (proved, sorry, wip,
//! not-applicable) and compute the proved-percent. Returns counts
//! per-status and proved-pct.
//!
//! Demonstrates the **CMM.204** recipe for PMAT-225 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Lean theorem-status enum; Coq proof-script-status
//!  reporting.
//!
//! Run with: cargo run --example contracts_macros_witness_proof_status
//!
//! Added by PMAT-225 (catalog 1648→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ProofStatusVerdict {
    Ok {
        proved: u32,
        sorry: u32,
        wip: u32,
        not_applicable: u32,
        proved_pct: u32,
    },
    InvalidConfig,
}

pub fn aggregate(statuses: &[&str]) -> ProofStatusVerdict {
    if statuses.is_empty() {
        return ProofStatusVerdict::InvalidConfig;
    }
    let mut proved = 0u32;
    let mut sorry = 0u32;
    let mut wip = 0u32;
    let mut na = 0u32;
    let mut unknown = 0u32;
    for s in statuses {
        match *s {
            "proved" => proved += 1,
            "sorry" => sorry += 1,
            "wip" => wip += 1,
            "not-applicable" => na += 1,
            _ => unknown += 1,
        }
    }
    if unknown > 0 {
        return ProofStatusVerdict::InvalidConfig;
    }
    let total = statuses.len() as u32;
    let pct = (proved as u64 * 100 / total as u64) as u32;
    ProofStatusVerdict::Ok {
        proved,
        sorry,
        wip,
        not_applicable: na,
        proved_pct: pct,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_witness_proof_status")?;

    println!(
        "mixed: {:?}",
        aggregate(&["proved", "wip", "proved", "sorry"])
    );
    println!("invalid: {:?}", aggregate(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(aggregate(&[]), ProofStatusVerdict::InvalidConfig);
    }

    #[test]
    fn unknown_status_rejected() {
        assert_eq!(aggregate(&["unknown"]), ProofStatusVerdict::InvalidConfig);
    }

    #[test]
    fn proved_count_correct() {
        let v = aggregate(&["proved", "proved", "wip"]);
        if let ProofStatusVerdict::Ok { proved, .. } = v {
            assert_eq!(proved, 2);
        }
    }

    #[test]
    fn sorry_count_correct() {
        let v = aggregate(&["sorry", "sorry"]);
        if let ProofStatusVerdict::Ok { sorry, .. } = v {
            assert_eq!(sorry, 2);
        }
    }

    #[test]
    fn wip_count_correct() {
        let v = aggregate(&["wip", "wip", "wip"]);
        if let ProofStatusVerdict::Ok { wip, .. } = v {
            assert_eq!(wip, 3);
        }
    }

    #[test]
    fn na_count_correct() {
        let v = aggregate(&["not-applicable"]);
        if let ProofStatusVerdict::Ok { not_applicable, .. } = v {
            assert_eq!(not_applicable, 1);
        }
    }

    #[test]
    fn proved_pct_correct() {
        // 1 proved out of 4 → 25%
        let v = aggregate(&["proved", "wip", "wip", "wip"]);
        if let ProofStatusVerdict::Ok { proved_pct, .. } = v {
            assert_eq!(proved_pct, 25);
        }
    }

    #[test]
    fn all_proved_100_pct() {
        let v = aggregate(&["proved", "proved", "proved"]);
        if let ProofStatusVerdict::Ok { proved_pct, .. } = v {
            assert_eq!(proved_pct, 100);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = aggregate(&["proved"]);
        let r2 = aggregate(&["proved"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn case_sensitive() {
        assert_eq!(aggregate(&["PROVED"]), ProofStatusVerdict::InvalidConfig);
    }

    #[test]
    fn many_statuses_handled() {
        let items: Vec<&str> = (0..30).map(|_| "proved").collect();
        let v = aggregate(&items);
        if let ProofStatusVerdict::Ok { proved, .. } = v {
            assert_eq!(proved, 30);
        }
    }

    #[test]
    fn na_pct_distinct_from_proved() {
        let v = aggregate(&["not-applicable", "not-applicable", "proved"]);
        if let ProofStatusVerdict::Ok {
            not_applicable,
            proved,
            proved_pct,
            ..
        } = v
        {
            assert_eq!(not_applicable, 2);
            assert_eq!(proved, 1);
            assert_eq!(proved_pct, 33);
        }
    }
}
