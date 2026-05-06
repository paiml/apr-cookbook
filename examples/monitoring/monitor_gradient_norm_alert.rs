//! # Monitoring Gradient Norm Alert
//!
//! During training, gradient norms reveal pathologies:
//!   norm < 1e-7 → vanishing
//!   norm > 1e3 → exploding
//!   sudden spike (>10× moving avg) → instability
//!
//! Demonstrates the **MON.40** recipe for PMAT-156 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Pascanu et al. (2013) RNN gradient pathology analysis.
//!
//! Run with: cargo run --example monitor_gradient_norm_alert
//!
//! Added by PMAT-156 (catalog 1027→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GradVerdict {
    Healthy,
    Vanishing { current: f64 },
    Exploding { current: f64 },
    Spike { current: f64, ratio: f64 },
    InvalidNorm,
}

pub fn check(current_norm: f64, recent_avg: f64) -> GradVerdict {
    if !current_norm.is_finite()
        || !recent_avg.is_finite()
        || current_norm < 0.0
        || recent_avg < 0.0
    {
        return GradVerdict::InvalidNorm;
    }
    if current_norm < 1e-7 {
        return GradVerdict::Vanishing {
            current: current_norm,
        };
    }
    if current_norm > 1e3 {
        return GradVerdict::Exploding {
            current: current_norm,
        };
    }
    if recent_avg > 0.0 {
        let ratio = current_norm / recent_avg;
        if ratio > 10.0 {
            return GradVerdict::Spike {
                current: current_norm,
                ratio,
            };
        }
    }
    GradVerdict::Healthy
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_gradient_norm_alert")?;

    println!("healthy: {:?}", check(1.5, 1.4));
    println!("vanishing: {:?}", check(1e-9, 1.0));
    println!("exploding: {:?}", check(1e5, 1.0));
    println!("spike: {:?}", check(20.0, 1.0));
    println!("invalid: {:?}", check(-1.0, 1.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn healthy_normal_range() {
        assert_eq!(check(1.5, 1.4), GradVerdict::Healthy);
    }

    #[test]
    fn vanishing_detected() {
        let v = check(1e-9, 1.0);
        assert!(matches!(v, GradVerdict::Vanishing { .. }));
    }

    #[test]
    fn exploding_detected() {
        let v = check(1e5, 1.0);
        assert!(matches!(v, GradVerdict::Exploding { .. }));
    }

    #[test]
    fn spike_detected() {
        let v = check(20.0, 1.0);
        assert!(matches!(v, GradVerdict::Spike { .. }));
    }

    #[test]
    fn nan_rejected() {
        assert_eq!(check(f64::NAN, 1.0), GradVerdict::InvalidNorm);
    }

    #[test]
    fn negative_rejected() {
        assert_eq!(check(-1.0, 1.0), GradVerdict::InvalidNorm);
    }

    #[test]
    fn boundary_at_vanishing_floor() {
        // Just above 1e-7 → healthy.
        let v = check(1e-6, 1.0);
        assert_eq!(v, GradVerdict::Healthy);
    }

    #[test]
    fn boundary_at_exploding_ceiling() {
        // Just below 1e3 → healthy.
        let v = check(999.0, 100.0);
        assert_eq!(v, GradVerdict::Healthy);
    }

    #[test]
    fn boundary_at_10x_spike_just_under() {
        let v = check(10.0, 1.0);
        // Exactly 10× → not spike (only > 10).
        assert_eq!(v, GradVerdict::Healthy);
    }

    #[test]
    fn no_recent_avg_no_spike_check() {
        // recent_avg == 0 → skip spike check.
        let v = check(50.0, 0.0);
        assert_eq!(v, GradVerdict::Healthy);
    }

    #[test]
    fn deterministic() {
        let a = check(20.0, 1.0);
        let b = check(20.0, 1.0);
        assert_eq!(a, b);
    }
}
