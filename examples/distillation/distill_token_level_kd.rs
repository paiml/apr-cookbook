//! # Distillation Token-Level KD Weighting
//!
//! In sequence distillation, weight per-token KL by token importance
//! (high-entropy tokens are harder, deserve more loss; low-entropy
//! tokens are trivial). Weight = clamp(entropy / max_entropy, 0.1, 1.0).
//!
//! Demonstrates the **DIST.30** recipe for PMAT-155 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Sanh et al. (2019) DistilBERT + token-importance weighting.
//!
//! Run with: cargo run --example distill_token_level_kd
//!
//! Added by PMAT-155 (catalog 1018→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WeightVerdict {
    Ok { weights: Vec<f64>, mean_weight: f64 },
    EmptyEntropies,
    InvalidEntropy,
}

pub fn weight(token_entropies: &[f64], max_entropy: f64) -> WeightVerdict {
    if token_entropies.is_empty() {
        return WeightVerdict::EmptyEntropies;
    }
    if !max_entropy.is_finite() || max_entropy <= 0.0 {
        return WeightVerdict::InvalidEntropy;
    }
    if token_entropies.iter().any(|e| !e.is_finite() || *e < 0.0) {
        return WeightVerdict::InvalidEntropy;
    }
    let weights: Vec<f64> = token_entropies
        .iter()
        .map(|&e| (e / max_entropy).clamp(0.1, 1.0))
        .collect();
    let mean_weight = weights.iter().sum::<f64>() / weights.len() as f64;
    WeightVerdict::Ok {
        weights,
        mean_weight,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_token_level_kd")?;

    let entropies = vec![0.1, 1.0, 2.5, 5.0];
    println!("typical: {:?}", weight(&entropies, 5.0));
    println!("empty: {:?}", weight(&[], 5.0));
    println!("invalid max: {:?}", weight(&[1.0], -1.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn high_entropy_high_weight() {
        let v = weight(&[5.0], 5.0);
        if let WeightVerdict::Ok { weights, .. } = v {
            assert!((weights[0] - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn low_entropy_floors_at_0_1() {
        let v = weight(&[0.0], 5.0);
        if let WeightVerdict::Ok { weights, .. } = v {
            assert!((weights[0] - 0.1).abs() < 1e-9);
        }
    }

    #[test]
    fn proportional_in_range() {
        let v = weight(&[2.5], 5.0);
        if let WeightVerdict::Ok { weights, .. } = v {
            // 2.5 / 5.0 = 0.5.
            assert!((weights[0] - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(weight(&[], 5.0), WeightVerdict::EmptyEntropies);
    }

    #[test]
    fn negative_max_rejected() {
        assert_eq!(weight(&[1.0], -1.0), WeightVerdict::InvalidEntropy);
    }

    #[test]
    fn nan_token_rejected() {
        assert_eq!(weight(&[f64::NAN], 5.0), WeightVerdict::InvalidEntropy);
    }

    #[test]
    fn negative_entropy_rejected() {
        assert_eq!(weight(&[-1.0], 5.0), WeightVerdict::InvalidEntropy);
    }

    #[test]
    fn over_max_clamped_at_1() {
        // Entropy > max → clamp to 1.0.
        let v = weight(&[10.0], 5.0);
        if let WeightVerdict::Ok { weights, .. } = v {
            assert!((weights[0] - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn mean_weight_correct() {
        let v = weight(&[5.0, 5.0, 5.0], 5.0);
        if let WeightVerdict::Ok { mean_weight, .. } = v {
            assert!((mean_weight - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn token_count_preserved() {
        let v = weight(&[0.1, 1.0, 2.5, 5.0], 5.0);
        if let WeightVerdict::Ok { weights, .. } = v {
            assert_eq!(weights.len(), 4);
        }
    }

    #[test]
    fn deterministic() {
        let a = weight(&[1.0, 2.0, 3.0], 5.0);
        let b = weight(&[1.0, 2.0, 3.0], 5.0);
        assert_eq!(a, b);
    }
}
