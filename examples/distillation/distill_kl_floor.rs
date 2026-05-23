//! # Distillation KL Floor for Stability
//!
//! When teacher is very confident, KL divergence is huge and dominates
//! gradient. Floor: clamp computed KL to a maximum value to prevent
//! the student from being pulled too hard on edge cases.
//!
//! Demonstrates the **DIST.42** recipe for PMAT-159 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hinton et al. (2015) on temperature/floor for KD stability.
//!
//! Run with: cargo run --example distill_kl_floor
//!
//! Added by PMAT-159 (catalog 1054→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FloorVerdict {
    Ok { clamped: f64, was_clamped: bool },
    InvalidInput,
}

pub fn apply(raw_kl: f64, max_kl: f64) -> FloorVerdict {
    if !raw_kl.is_finite() || !max_kl.is_finite() || raw_kl < 0.0 || max_kl <= 0.0 {
        return FloorVerdict::InvalidInput;
    }
    if raw_kl > max_kl {
        FloorVerdict::Ok {
            clamped: max_kl,
            was_clamped: true,
        }
    } else {
        FloorVerdict::Ok {
            clamped: raw_kl,
            was_clamped: false,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_kl_floor")?;

    println!("normal: {:?}", apply(0.5, 5.0));
    println!("clamped: {:?}", apply(50.0, 5.0));
    println!("at boundary: {:?}", apply(5.0, 5.0));
    println!("invalid: {:?}", apply(-1.0, 5.0));
    println!("nan: {:?}", apply(f64::NAN, 5.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_kl_unchanged() {
        let v = apply(0.5, 5.0);
        if let FloorVerdict::Ok {
            clamped,
            was_clamped,
        } = v
        {
            assert!((clamped - 0.5).abs() < 1e-9);
            assert!(!was_clamped);
        }
    }

    #[test]
    fn large_kl_clamped() {
        let v = apply(50.0, 5.0);
        if let FloorVerdict::Ok {
            clamped,
            was_clamped,
        } = v
        {
            assert!((clamped - 5.0).abs() < 1e-9);
            assert!(was_clamped);
        }
    }

    #[test]
    fn boundary_at_max_unchanged() {
        // raw == max → not clamped.
        let v = apply(5.0, 5.0);
        if let FloorVerdict::Ok { was_clamped, .. } = v {
            assert!(!was_clamped);
        }
    }

    #[test]
    fn negative_kl_rejected() {
        assert_eq!(apply(-1.0, 5.0), FloorVerdict::InvalidInput);
    }

    #[test]
    fn zero_max_rejected() {
        assert_eq!(apply(1.0, 0.0), FloorVerdict::InvalidInput);
    }

    #[test]
    fn nan_kl_rejected() {
        assert_eq!(apply(f64::NAN, 5.0), FloorVerdict::InvalidInput);
    }

    #[test]
    fn infinite_kl_rejected() {
        assert_eq!(apply(f64::INFINITY, 5.0), FloorVerdict::InvalidInput);
    }

    #[test]
    fn infinite_max_rejected() {
        assert_eq!(apply(1.0, f64::INFINITY), FloorVerdict::InvalidInput);
    }

    #[test]
    fn just_above_max_clamped() {
        let v = apply(5.001, 5.0);
        if let FloorVerdict::Ok { was_clamped, .. } = v {
            assert!(was_clamped);
        }
    }

    #[test]
    fn zero_kl_unchanged() {
        let v = apply(0.0, 5.0);
        if let FloorVerdict::Ok { clamped, .. } = v {
            assert!((clamped - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic() {
        let a = apply(50.0, 5.0);
        let b = apply(50.0, 5.0);
        assert_eq!(a, b);
    }
}
