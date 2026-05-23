//! # Training Gradient Clip Norm Calculator
//!
//! Compute global L2 norm of all parameter gradients; if norm > clip,
//! scale all gradients by clip / norm. This recipe builds the per-step
//! clip decision (NoClip / Clipped { scale }) + the post-clip norm.
//!
//! Demonstrates the **TRAIN.16** recipe for PMAT-132 (training coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Pascanu, Mikolov & Bengio (2013). On the difficulty of training RNNs.
//!
//! Run with: cargo run --example training_grad_clip_norm
//!
//! Added by PMAT-132 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ClipVerdict {
    NoClip {
        norm: f64,
    },
    Clipped {
        norm: f64,
        scale: f64,
        clipped_norm: f64,
    },
    InvalidGradients,
    InvalidClip,
}

pub fn global_l2_norm(grads: &[f64]) -> Option<f64> {
    if grads.iter().any(|g| !g.is_finite()) {
        return None;
    }
    let sum_sq: f64 = grads.iter().map(|g| g * g).sum();
    Some(sum_sq.sqrt())
}

pub fn clip_decision(grads: &[f64], clip: f64) -> ClipVerdict {
    if !clip.is_finite() || clip <= 0.0 {
        return ClipVerdict::InvalidClip;
    }
    let Some(norm) = global_l2_norm(grads) else {
        return ClipVerdict::InvalidGradients;
    };
    if norm <= clip {
        ClipVerdict::NoClip { norm }
    } else {
        let scale = clip / norm;
        ClipVerdict::Clipped {
            norm,
            scale,
            clipped_norm: clip,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("training_grad_clip_norm")?;

    let small = vec![0.1, 0.2, 0.3];
    let huge = vec![10.0, 20.0, 30.0];
    let with_nan = vec![0.5, f64::NAN];
    println!("small clip 1.0: {:?}", clip_decision(&small, 1.0));
    println!("huge clip 1.0: {:?}", clip_decision(&huge, 1.0));
    println!("nan: {:?}", clip_decision(&with_nan, 1.0));
    println!("invalid clip: {:?}", clip_decision(&small, 0.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calc_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn norm_basic_math() {
        // sqrt(3^2 + 4^2) = 5.
        let n = global_l2_norm(&[3.0, 4.0]).unwrap();
        assert!((n - 5.0).abs() < 1e-9);
    }

    #[test]
    fn norm_zero_for_zero_grads() {
        let n = global_l2_norm(&[0.0; 100]).unwrap();
        assert!(n.abs() < 1e-12);
    }

    #[test]
    fn nan_grad_rejected() {
        assert!(global_l2_norm(&[1.0, f64::NAN]).is_none());
    }

    #[test]
    fn small_grads_no_clip() {
        let v = clip_decision(&[0.1, 0.2], 1.0);
        assert!(matches!(v, ClipVerdict::NoClip { .. }));
    }

    #[test]
    fn large_grads_clipped() {
        let v = clip_decision(&[10.0, 20.0], 1.0);
        assert!(matches!(v, ClipVerdict::Clipped { .. }));
    }

    #[test]
    fn at_clip_boundary_no_clip() {
        // norm = exactly 1.0; clip = 1.0 → NoClip (≤ inclusive).
        let v = clip_decision(&[1.0], 1.0);
        assert!(matches!(v, ClipVerdict::NoClip { .. }));
    }

    #[test]
    fn clipped_norm_equals_clip_value() {
        let v = clip_decision(&[10.0, 20.0], 5.0);
        if let ClipVerdict::Clipped { clipped_norm, .. } = v {
            assert!((clipped_norm - 5.0).abs() < 1e-9);
        }
    }

    #[test]
    fn scale_proportional_to_overage() {
        // norm = 5, clip = 1 → scale = 0.2.
        let v = clip_decision(&[3.0, 4.0], 1.0);
        if let ClipVerdict::Clipped { scale, .. } = v {
            assert!((scale - 0.2).abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_clip_rejected() {
        assert_eq!(clip_decision(&[1.0, 2.0], 0.0), ClipVerdict::InvalidClip);
        assert_eq!(clip_decision(&[1.0, 2.0], -1.0), ClipVerdict::InvalidClip);
    }

    #[test]
    fn nan_grad_yields_invalid() {
        assert_eq!(
            clip_decision(&[1.0, f64::NAN], 1.0),
            ClipVerdict::InvalidGradients
        );
    }
}
