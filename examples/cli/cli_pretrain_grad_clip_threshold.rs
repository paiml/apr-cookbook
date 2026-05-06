//! # apr pretrain --grad-clip — Norm Threshold Validator
//!
//! Gradient clipping caps ‖∇‖₂ to prevent exploding gradients. Floor:
//! 0.1 (too small destroys signal); typical: 1.0 (most LLM recipes);
//! ceiling: 10.0 (essentially unclipped). This recipe builds the
//! validator + clip-fraction predictor.
//!
//! Demonstrates the **PRE.6** recipe for PMAT-117 (apr pretrain coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PRE-001 + Pascanu et al. 2013 (gradient clipping)
//!
//! Run with: cargo run --example cli_pretrain_grad_clip_threshold
//!
//! Added by PMAT-117 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ClipVerdict {
    Ok,
    BelowFloor { recommended: f64 },
    AboveCeiling { recommended: f64 },
    InvalidThreshold,
}

const MIN_THRESHOLD: f64 = 0.1;
const MAX_THRESHOLD: f64 = 10.0;
const TYPICAL: f64 = 1.0;

pub fn classify(threshold: f64) -> ClipVerdict {
    if !threshold.is_finite() || threshold <= 0.0 {
        return ClipVerdict::InvalidThreshold;
    }
    if threshold < MIN_THRESHOLD {
        return ClipVerdict::BelowFloor {
            recommended: TYPICAL,
        };
    }
    if threshold > MAX_THRESHOLD {
        return ClipVerdict::AboveCeiling {
            recommended: MAX_THRESHOLD,
        };
    }
    ClipVerdict::Ok
}

pub fn would_clip(observed_norm: f64, threshold: f64) -> bool {
    observed_norm.is_finite() && observed_norm > threshold
}

pub fn clip_fraction(norms: &[f64], threshold: f64) -> Option<f64> {
    if norms.is_empty() {
        return None;
    }
    let clipped = norms.iter().filter(|&&n| would_clip(n, threshold)).count();
    Some(clipped as f64 / norms.len() as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_pretrain_grad_clip_threshold")?;

    for t in [0.0, 0.05, 0.1, 1.0, 10.0, 50.0] {
        println!("threshold={t:>5.2}  →  {:?}", classify(t));
    }

    let norms = [0.5, 1.2, 0.8, 5.0, 0.9];
    println!("clip fraction @ 1.0: {:?}", clip_fraction(&norms, 1.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_threshold_passes() {
        assert_eq!(classify(TYPICAL), ClipVerdict::Ok);
    }

    #[test]
    fn at_floor_passes() {
        assert_eq!(classify(MIN_THRESHOLD), ClipVerdict::Ok);
    }

    #[test]
    fn at_ceiling_passes() {
        assert_eq!(classify(MAX_THRESHOLD), ClipVerdict::Ok);
    }

    #[test]
    fn below_floor_rejected() {
        let v = classify(0.05);
        assert!(matches!(v, ClipVerdict::BelowFloor { .. }));
    }

    #[test]
    fn above_ceiling_rejected() {
        let v = classify(50.0);
        assert!(matches!(v, ClipVerdict::AboveCeiling { .. }));
    }

    #[test]
    fn zero_or_negative_invalid() {
        assert_eq!(classify(0.0), ClipVerdict::InvalidThreshold);
        assert_eq!(classify(-0.5), ClipVerdict::InvalidThreshold);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(classify(f64::NAN), ClipVerdict::InvalidThreshold);
    }

    #[test]
    fn would_clip_above_threshold() {
        assert!(would_clip(2.0, 1.0));
        assert!(!would_clip(0.5, 1.0));
        assert!(!would_clip(1.0, 1.0)); // equal does not clip
    }

    #[test]
    fn clip_fraction_typical() {
        // 1 of 5 above 1.0 → 0.2.
        let norms = [0.5, 1.2, 0.8, 0.99, 0.9];
        let f = clip_fraction(&norms, 1.0).unwrap();
        assert!((f - 0.2).abs() < 1e-9);
    }

    #[test]
    fn clip_fraction_empty_yields_none() {
        assert!(clip_fraction(&[], 1.0).is_none());
    }
}
