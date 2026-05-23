//! # Inference Top-p Nucleus Sampling
//!
//! Top-p (nucleus) sampling: keep the smallest set of tokens whose
//! cumulative probability ≥ p, then renormalize and sample. Common p:
//! 0.9, 0.95. p=1.0 = full distribution; p=0 = top-1 only. This recipe
//! builds the cutoff selector + renormalization.
//!
//! Demonstrates the **INF.14** recipe for PMAT-129 (inference coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Holtzman et al. (2019). The Curious Case of Neural Text Degeneration. arXiv:1904.09751.
//!
//! Run with: cargo run --example inference_top_p_nucleus
//!
//! Added by PMAT-129 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum NucleusVerdict {
    Ok {
        kept_indices: Vec<usize>,
        renormalized: Vec<f64>,
    },
    InvalidP,
    EmptyDistribution,
    NonNormalizedDistribution {
        sum: f64,
    },
}

pub fn select_nucleus(probs: &[f64], p: f64) -> NucleusVerdict {
    if probs.is_empty() {
        return NucleusVerdict::EmptyDistribution;
    }
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return NucleusVerdict::InvalidP;
    }
    if probs.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return NucleusVerdict::InvalidP;
    }
    let sum: f64 = probs.iter().sum();
    if (sum - 1.0).abs() > 1e-3 {
        return NucleusVerdict::NonNormalizedDistribution { sum };
    }
    let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept = Vec::new();
    let mut cum = 0.0;
    for (idx, prob) in &indexed {
        kept.push(*idx);
        cum += *prob;
        if cum >= p {
            break;
        }
    }
    let kept_probs: Vec<f64> = kept.iter().map(|&i| probs[i]).collect();
    let kept_sum: f64 = kept_probs.iter().sum();
    let renormalized: Vec<f64> = if kept_sum > 0.0 {
        kept_probs.iter().map(|x| x / kept_sum).collect()
    } else {
        kept_probs
    };
    NucleusVerdict::Ok {
        kept_indices: kept,
        renormalized,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("inference_top_p_nucleus")?;

    let probs = [0.4, 0.3, 0.15, 0.1, 0.05];
    for p in [0.1, 0.5, 0.9, 1.0] {
        println!("p={p}  →  {:?}", select_nucleus(&probs, p));
    }
    println!("invalid p: {:?}", select_nucleus(&probs, 1.5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nucleus_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn p_threshold_keeps_correct_tokens() {
        // [0.4, 0.3, 0.15, 0.1, 0.05]; p=0.7 → keep 0, 1 (cum 0.7).
        let v = select_nucleus(&[0.4, 0.3, 0.15, 0.1, 0.05], 0.7);
        if let NucleusVerdict::Ok { kept_indices, .. } = v {
            assert_eq!(kept_indices, vec![0, 1]);
        }
    }

    #[test]
    fn small_p_keeps_top_1() {
        let v = select_nucleus(&[0.5, 0.3, 0.2], 0.1);
        if let NucleusVerdict::Ok { kept_indices, .. } = v {
            // Cum 0.5 ≥ 0.1 → just top-1.
            assert_eq!(kept_indices, vec![0]);
        }
    }

    #[test]
    fn p_one_keeps_all() {
        let v = select_nucleus(&[0.4, 0.3, 0.2, 0.1], 1.0);
        if let NucleusVerdict::Ok { kept_indices, .. } = v {
            assert_eq!(kept_indices.len(), 4);
        }
    }

    #[test]
    fn renormalized_sums_to_one() {
        let v = select_nucleus(&[0.4, 0.3, 0.15, 0.1, 0.05], 0.7);
        if let NucleusVerdict::Ok { renormalized, .. } = v {
            let sum: f64 = renormalized.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_p_rejected() {
        assert_eq!(select_nucleus(&[0.5, 0.5], 1.5), NucleusVerdict::InvalidP);
        assert_eq!(select_nucleus(&[0.5, 0.5], -0.1), NucleusVerdict::InvalidP);
        assert_eq!(
            select_nucleus(&[0.5, 0.5], f64::NAN),
            NucleusVerdict::InvalidP
        );
    }

    #[test]
    fn empty_distribution_rejected() {
        assert_eq!(select_nucleus(&[], 0.9), NucleusVerdict::EmptyDistribution);
    }

    #[test]
    fn non_normalized_rejected() {
        let v = select_nucleus(&[0.5, 0.4], 0.9);
        assert!(matches!(
            v,
            NucleusVerdict::NonNormalizedDistribution { .. }
        ));
    }

    #[test]
    fn negative_prob_rejected() {
        assert_eq!(
            select_nucleus(&[0.6, -0.1, 0.5], 0.9),
            NucleusVerdict::InvalidP
        );
    }

    #[test]
    fn sorted_descending_within_tolerance() {
        // Verify the picker uses sorted-by-prob, not source order.
        let v = select_nucleus(&[0.05, 0.4, 0.3, 0.15, 0.1], 0.7);
        if let NucleusVerdict::Ok { kept_indices, .. } = v {
            // Top probs are at indices 1 (0.4) and 2 (0.3).
            assert!(kept_indices.contains(&1));
            assert!(kept_indices.contains(&2));
        }
    }

    #[test]
    fn tiny_distribution_preserved() {
        // Single-token distribution.
        let v = select_nucleus(&[1.0], 0.9);
        if let NucleusVerdict::Ok {
            kept_indices,
            renormalized,
        } = v
        {
            assert_eq!(kept_indices, vec![0]);
            assert!((renormalized[0] - 1.0).abs() < 1e-9);
        }
    }
}
