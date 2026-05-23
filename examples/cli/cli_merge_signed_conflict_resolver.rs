//! # apr merge — Signed-Conflict Resolver (TIES sign election)
//!
//! TIES merging requires per-parameter sign election: for each
//! parameter, vote for the sign that has the largest aggregate
//! magnitude across task vectors. This recipe builds the resolver.
//!
//! Demonstrates the **MERGE.6** recipe for PMAT-112 (apr merge coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MERGE-001 + Yadav et al. 2023 (TIES sign election)
//!
//! Run with: cargo run --example cli_merge_signed_conflict_resolver
//!
//! Added by PMAT-112 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ElectedSign {
    Positive,
    Negative,
    Tie,
    NoData,
}

pub fn elect_sign(task_vectors: &[f64]) -> ElectedSign {
    if task_vectors.is_empty() {
        return ElectedSign::NoData;
    }
    let pos_mag: f64 = task_vectors.iter().filter(|&&x| x > 0.0).sum();
    let neg_mag: f64 = task_vectors
        .iter()
        .filter(|&&x| x < 0.0)
        .map(|x| x.abs())
        .sum();
    if pos_mag > neg_mag {
        ElectedSign::Positive
    } else if neg_mag > pos_mag {
        ElectedSign::Negative
    } else {
        ElectedSign::Tie
    }
}

pub fn merge_with_sign_filter(task_vectors: &[f64]) -> f64 {
    match elect_sign(task_vectors) {
        ElectedSign::Positive => task_vectors.iter().filter(|&&x| x > 0.0).sum(),
        ElectedSign::Negative => task_vectors.iter().filter(|&&x| x < 0.0).sum(),
        ElectedSign::Tie | ElectedSign::NoData => 0.0,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_merge_signed_conflict_resolver")?;

    let cases: &[(&str, &[f64])] = &[
        ("all positive", &[0.1, 0.2, 0.3]),
        ("all negative", &[-0.1, -0.2, -0.3]),
        ("mixed positive-dominant", &[0.5, -0.1, 0.2]),
        ("mixed negative-dominant", &[0.1, -0.5, -0.2]),
        ("perfect tie", &[0.3, -0.3]),
        ("empty", &[]),
    ];
    for (label, tv) in cases {
        println!(
            "{label:>26}  →  {:?}   merged = {:+.3}",
            elect_sign(tv),
            merge_with_sign_filter(tv)
        );
    }
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
    fn all_positive_elect_positive() {
        assert_eq!(elect_sign(&[0.1, 0.2, 0.3]), ElectedSign::Positive);
    }

    #[test]
    fn all_negative_elect_negative() {
        assert_eq!(elect_sign(&[-0.1, -0.2, -0.3]), ElectedSign::Negative);
    }

    #[test]
    fn larger_aggregate_magnitude_wins() {
        // pos=0.5, neg=0.1+0.2=0.3 → positive wins.
        assert_eq!(elect_sign(&[0.5, -0.1, -0.2]), ElectedSign::Positive);
        // pos=0.1, neg=0.5+0.2=0.7 → negative wins.
        assert_eq!(elect_sign(&[0.1, -0.5, -0.2]), ElectedSign::Negative);
    }

    #[test]
    fn perfect_tie_yields_tie() {
        assert_eq!(elect_sign(&[0.3, -0.3]), ElectedSign::Tie);
    }

    #[test]
    fn empty_yields_no_data() {
        assert_eq!(elect_sign(&[]), ElectedSign::NoData);
    }

    #[test]
    fn merge_filters_by_elected_sign() {
        // pos elected → drop negatives, sum positives = 0.5.
        let result = merge_with_sign_filter(&[0.5, -0.1, -0.2]);
        assert!((result - 0.5).abs() < 1e-9);
    }

    #[test]
    fn merge_tie_yields_zero() {
        assert_eq!(merge_with_sign_filter(&[0.3, -0.3]), 0.0);
    }

    #[test]
    fn zero_vectors_treated_as_ties() {
        // No positive or negative magnitudes → tie.
        assert_eq!(elect_sign(&[0.0, 0.0]), ElectedSign::Tie);
    }
}
