//! # Training Label Smoothing Epsilon Picker
//!
//! Smooth one-hot labels: target = (1-ε) for true class + ε/(V-1) for
//! others. Picker:
//!   small vocab (< 100): ε = 0.0 (no smoothing needed)
//!   medium (100-10k): ε = 0.05
//!   large (10k-100k): ε = 0.1 (helps reduce overconfidence)
//!   huge (≥ 100k): ε = 0.05 (already smooth via softmax)
//!
//! Demonstrates the **TRAIN.20** recipe for PMAT-151 (training round 6).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Szegedy et al. (2016). Inception-v3 label smoothing.
//!
//! Run with: cargo run --example training_label_smoothing
//!
//! Added by PMAT-151 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SmoothingVerdict {
    Ok {
        epsilon: f64,
        true_class_target: f64,
        other_class_target: f64,
    },
    InvalidVocabSize,
}

pub fn pick(vocab_size: u32) -> SmoothingVerdict {
    if vocab_size < 2 {
        return SmoothingVerdict::InvalidVocabSize;
    }
    let epsilon = if vocab_size < 100 {
        0.0
    } else if vocab_size < 10_000 {
        0.05
    } else if vocab_size < 100_000 {
        0.10
    } else {
        0.05
    };
    let true_class_target = 1.0 - epsilon;
    let other_class_target = epsilon / f64::from(vocab_size - 1);
    SmoothingVerdict::Ok {
        epsilon,
        true_class_target,
        other_class_target,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("training_label_smoothing")?;

    println!("small vocab 50: {:?}", pick(50));
    println!("medium 500: {:?}", pick(500));
    println!("large 50k: {:?}", pick(50_000));
    println!("huge 200k: {:?}", pick(200_000));
    println!("invalid: {:?}", pick(1));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_vocab_no_smoothing() {
        let v = pick(50);
        if let SmoothingVerdict::Ok { epsilon, .. } = v {
            assert!(epsilon.abs() < 1e-9);
        }
    }

    #[test]
    fn medium_vocab_05_smoothing() {
        let v = pick(500);
        if let SmoothingVerdict::Ok { epsilon, .. } = v {
            assert!((epsilon - 0.05).abs() < 1e-9);
        }
    }

    #[test]
    fn large_vocab_10_smoothing() {
        let v = pick(50_000);
        if let SmoothingVerdict::Ok { epsilon, .. } = v {
            assert!((epsilon - 0.10).abs() < 1e-9);
        }
    }

    #[test]
    fn huge_vocab_05_smoothing() {
        let v = pick(200_000);
        if let SmoothingVerdict::Ok { epsilon, .. } = v {
            assert!((epsilon - 0.05).abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_vocab_size_one() {
        assert_eq!(pick(1), SmoothingVerdict::InvalidVocabSize);
    }

    #[test]
    fn invalid_vocab_size_zero() {
        assert_eq!(pick(0), SmoothingVerdict::InvalidVocabSize);
    }

    #[test]
    fn true_class_target_one_minus_eps() {
        let v = pick(500);
        if let SmoothingVerdict::Ok {
            epsilon,
            true_class_target,
            ..
        } = v
        {
            assert!((true_class_target - (1.0 - epsilon)).abs() < 1e-9);
        }
    }

    #[test]
    fn other_class_target_distributed() {
        // vocab=10, eps=0.05 → other = 0.05 / 9.
        let v = pick(10_000);
        if let SmoothingVerdict::Ok {
            epsilon,
            other_class_target,
            ..
        } = v
        {
            let expected = epsilon / 9999.0;
            assert!((other_class_target - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn boundary_at_100_vocab() {
        let v = pick(100);
        if let SmoothingVerdict::Ok { epsilon, .. } = v {
            assert!((epsilon - 0.05).abs() < 1e-9);
        }
    }

    #[test]
    fn boundary_at_10k_vocab() {
        let v = pick(10_000);
        if let SmoothingVerdict::Ok { epsilon, .. } = v {
            assert!((epsilon - 0.10).abs() < 1e-9);
        }
    }

    #[test]
    fn smoothing_targets_sum_to_one() {
        // (1 - ε) + (V - 1) × ε / (V - 1) = (1 - ε) + ε = 1.
        let v = pick(500);
        if let SmoothingVerdict::Ok {
            true_class_target,
            other_class_target,
            ..
        } = v
        {
            let total = true_class_target + other_class_target * 499.0;
            assert!((total - 1.0).abs() < 1e-9);
        }
    }
}
