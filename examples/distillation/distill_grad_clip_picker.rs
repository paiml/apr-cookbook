//! # Distillation Gradient-Clip Picker
//!
//! Pick gradient-clipping threshold from the gradient-norm distribution.
//! Rule: `clip = max(median × k, hardcoded_min)` where k=2.0 catches
//! outliers but keeps most updates intact.
//!
//! Demonstrates the **DIST.31** recipe for PMAT-156 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Pascanu et al. (2013) "On the difficulty of training RNNs."
//!
//! Run with: cargo run --example distill_grad_clip_picker
//!
//! Added by PMAT-156 (catalog 1027→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ClipVerdict {
    Ok { clip_threshold: f64, median: f64 },
    EmptyNorms,
    InvalidNorms,
}

pub fn pick(grad_norms: &[f64], min_clip: f64, k: f64) -> ClipVerdict {
    if grad_norms.is_empty() {
        return ClipVerdict::EmptyNorms;
    }
    if grad_norms.iter().any(|n| !n.is_finite() || *n < 0.0) {
        return ClipVerdict::InvalidNorms;
    }
    if !min_clip.is_finite() || min_clip <= 0.0 || !k.is_finite() || k <= 0.0 {
        return ClipVerdict::InvalidNorms;
    }
    let mut sorted = grad_norms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let clip_threshold = (median * k).max(min_clip);
    ClipVerdict::Ok {
        clip_threshold,
        median,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_grad_clip_picker")?;

    println!("typical: {:?}", pick(&[1.0, 1.5, 2.0, 2.5, 3.0], 0.5, 2.0));
    println!(
        "with outlier: {:?}",
        pick(&[0.5, 1.0, 1.5, 2.0, 100.0], 0.5, 2.0)
    );
    println!(
        "low norms (floor): {:?}",
        pick(&[0.01, 0.02, 0.03], 1.0, 2.0)
    );
    println!("empty: {:?}", pick(&[], 0.5, 2.0));
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
    fn typical_uses_median_times_k() {
        let v = pick(&[1.0, 1.5, 2.0, 2.5, 3.0], 0.5, 2.0);
        if let ClipVerdict::Ok {
            clip_threshold,
            median,
        } = v
        {
            // Median = 2.0, clip = 4.0.
            assert!((median - 2.0).abs() < 1e-9);
            assert!((clip_threshold - 4.0).abs() < 1e-9);
        }
    }

    #[test]
    fn outlier_does_not_skew_median() {
        let v = pick(&[0.5, 1.0, 1.5, 2.0, 100.0], 0.5, 2.0);
        if let ClipVerdict::Ok { median, .. } = v {
            // Median is 1.5, robust to the 100.0 outlier.
            assert!((median - 1.5).abs() < 1e-9);
        }
    }

    #[test]
    fn min_clip_floor_applied() {
        let v = pick(&[0.01, 0.02, 0.03], 1.0, 2.0);
        if let ClipVerdict::Ok { clip_threshold, .. } = v {
            // median × k = 0.04 < min_clip = 1.0 → floor.
            assert!((clip_threshold - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(pick(&[], 0.5, 2.0), ClipVerdict::EmptyNorms);
    }

    #[test]
    fn nan_norm_rejected() {
        assert_eq!(pick(&[f64::NAN], 0.5, 2.0), ClipVerdict::InvalidNorms);
    }

    #[test]
    fn negative_norm_rejected() {
        assert_eq!(pick(&[-1.0, 1.0], 0.5, 2.0), ClipVerdict::InvalidNorms);
    }

    #[test]
    fn negative_min_clip_rejected() {
        assert_eq!(pick(&[1.0, 2.0], -1.0, 2.0), ClipVerdict::InvalidNorms);
    }

    #[test]
    fn negative_k_rejected() {
        assert_eq!(pick(&[1.0, 2.0], 0.5, -2.0), ClipVerdict::InvalidNorms);
    }

    #[test]
    fn single_norm_works() {
        let v = pick(&[5.0], 0.1, 2.0);
        if let ClipVerdict::Ok { clip_threshold, .. } = v {
            assert!((clip_threshold - 10.0).abs() < 1e-9);
        }
    }

    #[test]
    fn unsorted_input_sorts_internally() {
        let v_a = pick(&[3.0, 1.0, 2.0, 5.0, 4.0], 0.5, 2.0);
        let v_b = pick(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.5, 2.0);
        assert_eq!(v_a, v_b);
    }

    #[test]
    fn deterministic() {
        let a = pick(&[1.0, 2.0, 3.0], 0.5, 2.0);
        let b = pick(&[1.0, 2.0, 3.0], 0.5, 2.0);
        assert_eq!(a, b);
    }
}
