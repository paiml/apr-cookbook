//! # apr explain --saliency — Top-K Token Importance Classifier
//!
//! Saliency maps assign each input token an importance score (e.g.,
//! grad×input). Top-K is the standard summary; rules: K ≥ 1, K ≤ N
//! tokens, abs-value sort by default, ties broken by position-first.
//! This recipe builds the ranker.
//!
//! Demonstrates the **EXP.4** recipe for PMAT-114 (apr explain coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender EXP-001 + Sundararajan et al. 2017 (Integrated Gradients)
//!
//! Run with: cargo run --example cli_explain_saliency_rank_classifier
//!
//! Added by PMAT-114 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RankVerdict {
    Ok(Vec<usize>),
    InvalidK,
    EmptyScores,
}

pub fn top_k_indices(scores: &[f64], k: usize) -> RankVerdict {
    if scores.is_empty() {
        return RankVerdict::EmptyScores;
    }
    if k == 0 || k > scores.len() {
        return RankVerdict::InvalidK;
    }
    let mut indexed: Vec<(usize, f64)> = scores
        .iter()
        .enumerate()
        .map(|(i, s)| (i, s.abs()))
        .collect();
    // Sort by abs desc, ties broken by smaller index first.
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    RankVerdict::Ok(indexed.into_iter().take(k).map(|(i, _)| i).collect())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_explain_saliency_rank_classifier")?;

    let scores = [0.1, -0.5, 0.3, -0.4, 0.2];
    println!("top-3: {:?}", top_k_indices(&scores, 3));
    println!("top-1: {:?}", top_k_indices(&scores, 1));
    println!("top-0 (bad): {:?}", top_k_indices(&scores, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn top_1_picks_max_abs() {
        let v = top_k_indices(&[0.1, -0.5, 0.3, -0.4, 0.2], 1);
        assert_eq!(v, RankVerdict::Ok(vec![1]));
    }

    #[test]
    fn top_k_returns_k_indices() {
        let v = top_k_indices(&[0.1, -0.5, 0.3, -0.4, 0.2], 3);
        if let RankVerdict::Ok(idxs) = v {
            assert_eq!(idxs.len(), 3);
        }
    }

    #[test]
    fn top_k_uses_abs_value() {
        // Negative big magnitude beats small positive.
        let v = top_k_indices(&[0.01, -0.99, 0.5], 2);
        if let RankVerdict::Ok(idxs) = v {
            assert!(idxs.contains(&1));
            assert!(idxs.contains(&2));
        }
    }

    #[test]
    fn ties_broken_by_position_first() {
        // Two tokens with identical magnitude: lower index first.
        let v = top_k_indices(&[0.5, 0.5, 0.5], 2);
        assert_eq!(v, RankVerdict::Ok(vec![0, 1]));
    }

    #[test]
    fn k_zero_rejected() {
        assert_eq!(top_k_indices(&[0.1], 0), RankVerdict::InvalidK);
    }

    #[test]
    fn k_larger_than_n_rejected() {
        assert_eq!(top_k_indices(&[0.1, 0.2], 5), RankVerdict::InvalidK);
    }

    #[test]
    fn empty_scores_rejected() {
        assert_eq!(top_k_indices(&[], 1), RankVerdict::EmptyScores);
    }

    #[test]
    fn k_equals_n_returns_all() {
        let v = top_k_indices(&[0.1, 0.2, 0.3], 3);
        if let RankVerdict::Ok(idxs) = v {
            assert_eq!(idxs.len(), 3);
            // All three indices present (in some order).
            let mut sorted = idxs;
            sorted.sort();
            assert_eq!(sorted, vec![0, 1, 2]);
        }
    }

    #[test]
    fn nan_scores_treat_as_zero_by_partial_cmp() {
        // NaN comparisons return Equal under unwrap_or; should not panic.
        let v = top_k_indices(&[f64::NAN, 0.5, 0.1], 2);
        assert!(matches!(v, RankVerdict::Ok(_)));
    }
}
