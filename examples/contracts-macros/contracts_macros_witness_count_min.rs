//! # Contracts-Macros Witness Count Min
//!
//! Verify each obligation has at least `min_witnesses` distinct
//! witness IDs. Returns sorted offending obligations and the median
//! witness count seen.
//!
//! Demonstrates the **CMM.159** recipe for PMAT-210 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SAT-solver UNSAT-core minimal-witness requirements;
//!  Coq/Lean proof-witness completeness rules.
//!
//! Run with: cargo run --example contracts_macros_witness_count_min
//!
//! Added by PMAT-210 (catalog 1513→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WitnessVerdict {
    Ok {
        offending_obligations: Vec<String>,
        median_witness_count: u32,
    },
    InvalidConfig,
}

pub fn check(obligations: &[(&str, u32)], min_witnesses: u32) -> WitnessVerdict {
    if obligations.is_empty() || min_witnesses == 0 {
        return WitnessVerdict::InvalidConfig;
    }
    let mut offenders: Vec<String> = obligations
        .iter()
        .filter(|(_, n)| *n < min_witnesses)
        .map(|(id, _)| (*id).to_string())
        .collect();
    offenders.sort();
    let mut counts: Vec<u32> = obligations.iter().map(|(_, n)| *n).collect();
    counts.sort_unstable();
    let median = counts[counts.len() / 2];
    WitnessVerdict::Ok {
        offending_obligations: offenders,
        median_witness_count: median,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_witness_count_min")?;

    let obligations = [("o1", 3), ("o2", 1), ("o3", 5)];
    println!("min-3: {:?}", check(&obligations, 3));
    println!("invalid: {:?}", check(&[], 3));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn meeting_min_no_offender() {
        let v = check(&[("o", 5)], 3);
        if let WitnessVerdict::Ok {
            offending_obligations,
            ..
        } = v
        {
            assert!(offending_obligations.is_empty());
        }
    }

    #[test]
    fn under_min_offender() {
        let v = check(&[("o", 1)], 3);
        if let WitnessVerdict::Ok {
            offending_obligations,
            ..
        } = v
        {
            assert_eq!(offending_obligations, vec!["o".to_string()]);
        }
    }

    #[test]
    fn at_min_no_offender() {
        let v = check(&[("o", 3)], 3);
        if let WitnessVerdict::Ok {
            offending_obligations,
            ..
        } = v
        {
            assert!(offending_obligations.is_empty());
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[], 3), WitnessVerdict::InvalidConfig);
    }

    #[test]
    fn zero_min_rejected() {
        assert_eq!(check(&[("o", 1)], 0), WitnessVerdict::InvalidConfig);
    }

    #[test]
    fn median_correct_odd() {
        let v = check(&[("a", 1), ("b", 5), ("c", 9)], 3);
        if let WitnessVerdict::Ok {
            median_witness_count,
            ..
        } = v
        {
            assert_eq!(median_witness_count, 5);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(&[("o", 5)], 3);
        let r2 = check(&[("o", 5)], 3);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offenders_sorted() {
        let v = check(&[("zeta", 1), ("alpha", 1)], 3);
        if let WitnessVerdict::Ok {
            offending_obligations,
            ..
        } = v
        {
            assert_eq!(
                offending_obligations,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn many_obligations_handled() {
        let obligations: Vec<(&str, u32)> = (0..30).map(|_| ("o", 1)).collect();
        let v = check(&obligations, 3);
        if let WitnessVerdict::Ok {
            offending_obligations,
            ..
        } = v
        {
            assert_eq!(offending_obligations.len(), 30);
        }
    }

    #[test]
    fn no_offenders_returns_empty() {
        let v = check(&[("a", 5), ("b", 7)], 3);
        if let WitnessVerdict::Ok {
            offending_obligations,
            ..
        } = v
        {
            assert!(offending_obligations.is_empty());
        }
    }

    #[test]
    fn unicode_id_supported() {
        let v = check(&[("café", 1)], 3);
        if let WitnessVerdict::Ok {
            offending_obligations,
            ..
        } = v
        {
            assert_eq!(offending_obligations, vec!["café".to_string()]);
        }
    }

    #[test]
    fn single_obligation_handled() {
        let v = check(&[("o", 5)], 3);
        if let WitnessVerdict::Ok {
            median_witness_count,
            ..
        } = v
        {
            assert_eq!(median_witness_count, 5);
        }
    }
}
