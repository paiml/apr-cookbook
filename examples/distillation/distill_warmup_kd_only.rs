//! # Distillation KD-Only Warmup
//!
//! Recipe: warm up training with pure KD (alpha=1) for first N steps,
//! then linearly anneal to target alpha (mix of KD + hard-label CE).
//! Removes label noise during early student exploration.
//!
//! Demonstrates the **DIST.38** recipe for PMAT-158 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Romero et al. (2014) FitNets — staged distillation.
//!
//! Run with: cargo run --example distill_warmup_kd_only
//!
//! Added by PMAT-158 (catalog 1045→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AlphaVerdict {
    Ok { alpha: f64 },
    InvalidConfig,
}

pub fn alpha_at_step(
    step: u32,
    warmup_steps: u32,
    anneal_end_step: u32,
    target_alpha: f64,
) -> AlphaVerdict {
    if !target_alpha.is_finite() || !(0.0..=1.0).contains(&target_alpha) {
        return AlphaVerdict::InvalidConfig;
    }
    if anneal_end_step <= warmup_steps {
        return AlphaVerdict::InvalidConfig;
    }
    if step <= warmup_steps {
        return AlphaVerdict::Ok { alpha: 1.0 };
    }
    if step >= anneal_end_step {
        return AlphaVerdict::Ok {
            alpha: target_alpha,
        };
    }
    let progress = f64::from(step - warmup_steps) / f64::from(anneal_end_step - warmup_steps);
    let alpha = 1.0 + (target_alpha - 1.0) * progress;
    AlphaVerdict::Ok { alpha }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_warmup_kd_only")?;

    println!("warmup: {:?}", alpha_at_step(50, 100, 1000, 0.5));
    println!("annealing: {:?}", alpha_at_step(550, 100, 1000, 0.5));
    println!("after anneal: {:?}", alpha_at_step(1500, 100, 1000, 0.5));
    println!("invalid: {:?}", alpha_at_step(50, 1000, 100, 0.5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn warmup_phase_alpha_one() {
        let v = alpha_at_step(50, 100, 1000, 0.5);
        if let AlphaVerdict::Ok { alpha } = v {
            assert!((alpha - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn after_anneal_target() {
        let v = alpha_at_step(1500, 100, 1000, 0.5);
        if let AlphaVerdict::Ok { alpha } = v {
            assert!((alpha - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn middle_anneal_interpolated() {
        // (550 - 100)/(1000 - 100) = 0.5; alpha = 1 + (0.5-1)*0.5 = 0.75.
        let v = alpha_at_step(550, 100, 1000, 0.5);
        if let AlphaVerdict::Ok { alpha } = v {
            assert!((alpha - 0.75).abs() < 1e-6);
        }
    }

    #[test]
    fn invalid_window_rejected() {
        assert_eq!(
            alpha_at_step(50, 1000, 100, 0.5),
            AlphaVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_target_rejected() {
        assert_eq!(
            alpha_at_step(50, 100, 1000, 1.5),
            AlphaVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_target_rejected() {
        assert_eq!(
            alpha_at_step(50, 100, 1000, f64::NAN),
            AlphaVerdict::InvalidConfig
        );
    }

    #[test]
    fn boundary_at_warmup_alpha_one() {
        // Exactly at warmup_steps → still alpha=1.
        let v = alpha_at_step(100, 100, 1000, 0.5);
        if let AlphaVerdict::Ok { alpha } = v {
            assert!((alpha - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn boundary_at_anneal_end() {
        let v = alpha_at_step(1000, 100, 1000, 0.5);
        if let AlphaVerdict::Ok { alpha } = v {
            assert!((alpha - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn target_zero_works() {
        let v = alpha_at_step(1500, 100, 1000, 0.0);
        if let AlphaVerdict::Ok { alpha } = v {
            assert!((alpha - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn alpha_monotone_decreasing() {
        let mut last = 1.0;
        for step in [101, 200, 400, 600, 800, 1000] {
            let v = alpha_at_step(step, 100, 1000, 0.5);
            if let AlphaVerdict::Ok { alpha } = v {
                assert!(alpha <= last + 1e-9);
                last = alpha;
            }
        }
    }

    #[test]
    fn deterministic() {
        let a = alpha_at_step(550, 100, 1000, 0.5);
        let b = alpha_at_step(550, 100, 1000, 0.5);
        assert_eq!(a, b);
    }
}
