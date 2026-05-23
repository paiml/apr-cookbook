//! # Distillation Logit Smoothing
//!
//! Extreme teacher logits (very high or very low) destabilize KL
//! divergence because softmax saturates. Smoother clamps logits to
//! `[-cap, +cap]` before passing to the student loss.
//!
//! Demonstrates the **DIST.28** recipe for PMAT-155 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Müller et al. (2019). When does label smoothing help?
//!
//! Run with: cargo run --example distill_logit_smoothing
//!
//! Added by PMAT-155 (catalog 1018→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SmoothVerdict {
    Ok {
        smoothed: Vec<f64>,
        clamped_count: u32,
    },
    EmptyLogits,
    InvalidCap,
}

pub fn smooth(logits: &[f64], cap: f64) -> SmoothVerdict {
    if logits.is_empty() {
        return SmoothVerdict::EmptyLogits;
    }
    if !cap.is_finite() || cap <= 0.0 {
        return SmoothVerdict::InvalidCap;
    }
    let mut clamped_count = 0u32;
    let smoothed: Vec<f64> = logits
        .iter()
        .map(|&l| {
            if l > cap {
                clamped_count += 1;
                cap
            } else if l < -cap {
                clamped_count += 1;
                -cap
            } else {
                l
            }
        })
        .collect();
    SmoothVerdict::Ok {
        smoothed,
        clamped_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_logit_smoothing")?;

    println!(
        "typical: {:?}",
        smooth(&[1.0, 2.0, -3.0, 50.0, -50.0], 10.0)
    );
    println!("no clamping: {:?}", smooth(&[1.0, 2.0, 3.0], 10.0));
    println!("empty: {:?}", smooth(&[], 10.0));
    println!("invalid cap: {:?}", smooth(&[1.0], -1.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoother_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn high_logit_clamped() {
        let v = smooth(&[50.0], 10.0);
        if let SmoothVerdict::Ok {
            smoothed,
            clamped_count,
        } = v
        {
            assert!((smoothed[0] - 10.0).abs() < 1e-9);
            assert_eq!(clamped_count, 1);
        }
    }

    #[test]
    fn low_logit_clamped() {
        let v = smooth(&[-50.0], 10.0);
        if let SmoothVerdict::Ok { smoothed, .. } = v {
            assert!((smoothed[0] + 10.0).abs() < 1e-9);
        }
    }

    #[test]
    fn middle_logit_unchanged() {
        let v = smooth(&[5.0], 10.0);
        if let SmoothVerdict::Ok {
            smoothed,
            clamped_count,
        } = v
        {
            assert!((smoothed[0] - 5.0).abs() < 1e-9);
            assert_eq!(clamped_count, 0);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(smooth(&[], 10.0), SmoothVerdict::EmptyLogits);
    }

    #[test]
    fn negative_cap_rejected() {
        assert_eq!(smooth(&[1.0], -1.0), SmoothVerdict::InvalidCap);
    }

    #[test]
    fn zero_cap_rejected() {
        assert_eq!(smooth(&[1.0], 0.0), SmoothVerdict::InvalidCap);
    }

    #[test]
    fn nan_cap_rejected() {
        assert_eq!(smooth(&[1.0], f64::NAN), SmoothVerdict::InvalidCap);
    }

    #[test]
    fn boundary_at_cap_unchanged() {
        let v = smooth(&[10.0, -10.0], 10.0);
        if let SmoothVerdict::Ok { clamped_count, .. } = v {
            // Exactly at boundary → not clamped (only > or <).
            assert_eq!(clamped_count, 0);
        }
    }

    #[test]
    fn count_correct_for_mixed() {
        let v = smooth(&[1.0, 50.0, -50.0, 100.0], 10.0);
        if let SmoothVerdict::Ok { clamped_count, .. } = v {
            assert_eq!(clamped_count, 3);
        }
    }

    #[test]
    fn deterministic() {
        let a = smooth(&[1.0, 50.0, -50.0], 10.0);
        let b = smooth(&[1.0, 50.0, -50.0], 10.0);
        assert_eq!(a, b);
    }
}
