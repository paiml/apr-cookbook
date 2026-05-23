//! # apr prune --method wanda — Activation-Aware Score
//!
//! Wanda (Sun et al. 2024) scores each weight by |W| · ‖X‖₂ where X is
//! the input activation magnitude. Largest scores survive; smallest get
//! pruned. This recipe builds the scorer + top-K selector.
//!
//! Demonstrates the **PRUNE.4** recipe for PMAT-113 (apr prune coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PRUNE-001 + Sun et al. 2024 (Wanda)
//!
//! Run with: cargo run --example cli_prune_wanda_activation_scorer
//!
//! Added by PMAT-113 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ScoringVerdict {
    Ok(Vec<f64>),
    LengthMismatch { weights: usize, activations: usize },
    NegativeActivationNorm,
}

pub fn wanda_scores(weights: &[f64], activation_norms: &[f64]) -> ScoringVerdict {
    if weights.len() != activation_norms.len() {
        return ScoringVerdict::LengthMismatch {
            weights: weights.len(),
            activations: activation_norms.len(),
        };
    }
    if activation_norms.iter().any(|x| *x < 0.0) {
        return ScoringVerdict::NegativeActivationNorm;
    }
    let scores = weights
        .iter()
        .zip(activation_norms)
        .map(|(w, x)| w.abs() * x)
        .collect();
    ScoringVerdict::Ok(scores)
}

pub fn pruning_mask(scores: &[f64], sparsity: f64) -> Vec<bool> {
    if scores.is_empty() {
        return Vec::new();
    }
    if !(0.0..=1.0).contains(&sparsity) {
        return vec![true; scores.len()]; // keep all on bad input
    }
    let n_prune = (scores.len() as f64 * sparsity).round() as usize;
    let mut indexed: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut mask = vec![true; scores.len()];
    for (i, _) in indexed.iter().take(n_prune) {
        mask[*i] = false;
    }
    mask
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_prune_wanda_activation_scorer")?;

    let weights = [-0.5, 0.1, 0.8, -0.2, 0.05];
    let activations = [1.0, 2.0, 0.5, 3.0, 0.1];
    println!("scores: {:?}", wanda_scores(&weights, &activations));
    if let ScoringVerdict::Ok(s) = wanda_scores(&weights, &activations) {
        println!("mask @ 40%: {:?}", pruning_mask(&s, 0.4));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scorer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn product_of_abs_weight_and_activation() {
        let v = wanda_scores(&[-2.0, 3.0], &[4.0, 5.0]);
        if let ScoringVerdict::Ok(s) = v {
            assert_eq!(s, vec![8.0, 15.0]);
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn length_mismatch_rejected() {
        let v = wanda_scores(&[1.0], &[1.0, 2.0]);
        assert!(matches!(v, ScoringVerdict::LengthMismatch { .. }));
    }

    #[test]
    fn negative_activation_rejected() {
        // ‖X‖₂ is always non-negative; negative input is a contract violation.
        let v = wanda_scores(&[1.0], &[-0.1]);
        assert_eq!(v, ScoringVerdict::NegativeActivationNorm);
    }

    #[test]
    fn prune_lowest_scores() {
        let scores = [10.0, 1.0, 5.0, 0.5];
        let mask = pruning_mask(&scores, 0.5);
        // 50% of 4 = 2 pruned; lowest = 0.5 (idx 3) and 1.0 (idx 1).
        assert!(!mask[1]);
        assert!(!mask[3]);
        assert!(mask[0]);
        assert!(mask[2]);
    }

    #[test]
    fn sparsity_zero_keeps_all() {
        let mask = pruning_mask(&[1.0, 2.0, 3.0], 0.0);
        assert_eq!(mask, vec![true, true, true]);
    }

    #[test]
    fn sparsity_one_prunes_all() {
        let mask = pruning_mask(&[1.0, 2.0, 3.0], 1.0);
        assert_eq!(mask, vec![false, false, false]);
    }

    #[test]
    fn invalid_sparsity_keeps_all() {
        assert_eq!(pruning_mask(&[1.0, 2.0], -0.1), vec![true, true]);
        assert_eq!(pruning_mask(&[1.0, 2.0], 1.5), vec![true, true]);
    }

    #[test]
    fn empty_input_yields_empty_mask() {
        assert!(pruning_mask(&[], 0.5).is_empty());
    }

    #[test]
    fn zero_weight_always_pruned_first() {
        // Zero weight gives zero score regardless of activation.
        let scores = [10.0, 0.0, 5.0];
        let mask = pruning_mask(&scores, 0.34); // round to 1
        assert!(!mask[1]);
    }
}
