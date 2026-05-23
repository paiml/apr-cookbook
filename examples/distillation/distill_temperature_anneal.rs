//! # Distillation Temperature Annealing Schedule
//!
//! Anneal T from `T_start` (high, soft labels) to `T_end` (low, sharp
//! labels) over training. Strategy:
//!   linear: T(s) = T_start + (T_end - T_start) × (s/total)
//!   exponential: T(s) = T_start × (T_end/T_start)^(s/total)
//!
//! Linear is simpler; exponential decays faster early.
//!
//! Demonstrates the **DIST.18** recipe for PMAT-145 (distillation round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Distillation temperature schedule (DistilBERT § 4.2).
//!
//! Run with: cargo run --example distill_temperature_anneal
//!
//! Added by PMAT-145 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnealStrategy {
    Linear,
    Exponential,
}

#[derive(Debug, PartialEq)]
pub enum AnnealVerdict {
    Ok { temperature: f64 },
    InvalidStartTemp,
    InvalidEndTemp,
    InvalidStep,
}

pub fn pick(
    strategy: AnnealStrategy,
    t_start: f64,
    t_end: f64,
    step: u32,
    total_steps: u32,
) -> AnnealVerdict {
    if !t_start.is_finite() || t_start <= 0.0 {
        return AnnealVerdict::InvalidStartTemp;
    }
    if !t_end.is_finite() || t_end <= 0.0 {
        return AnnealVerdict::InvalidEndTemp;
    }
    if total_steps == 0 {
        return AnnealVerdict::InvalidStep;
    }
    let progress = (f64::from(step) / f64::from(total_steps)).clamp(0.0, 1.0);
    let temperature = match strategy {
        AnnealStrategy::Linear => t_start + (t_end - t_start) * progress,
        AnnealStrategy::Exponential => t_start * (t_end / t_start).powf(progress),
    };
    AnnealVerdict::Ok { temperature }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_temperature_anneal")?;

    println!(
        "linear, step 0: {:?}",
        pick(AnnealStrategy::Linear, 10.0, 1.0, 0, 1000)
    );
    println!(
        "linear, step 500: {:?}",
        pick(AnnealStrategy::Linear, 10.0, 1.0, 500, 1000)
    );
    println!(
        "linear, step 1000: {:?}",
        pick(AnnealStrategy::Linear, 10.0, 1.0, 1000, 1000)
    );
    println!(
        "exp, step 500: {:?}",
        pick(AnnealStrategy::Exponential, 10.0, 1.0, 500, 1000)
    );
    println!(
        "invalid: {:?}",
        pick(AnnealStrategy::Linear, 0.0, 1.0, 0, 1000)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t_at(strategy: AnnealStrategy, step: u32) -> f64 {
        if let AnnealVerdict::Ok { temperature } = pick(strategy, 10.0, 1.0, step, 1000) {
            temperature
        } else {
            f64::NAN
        }
    }

    #[test]
    fn anneal_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn linear_start_at_t_start() {
        assert!((t_at(AnnealStrategy::Linear, 0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn linear_end_at_t_end() {
        assert!((t_at(AnnealStrategy::Linear, 1000) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn linear_mid_average() {
        assert!((t_at(AnnealStrategy::Linear, 500) - 5.5).abs() < 1e-9);
    }

    #[test]
    fn exponential_start_at_t_start() {
        assert!((t_at(AnnealStrategy::Exponential, 0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn exponential_end_at_t_end() {
        assert!((t_at(AnnealStrategy::Exponential, 1000) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn exponential_decays_faster_initially() {
        // At step 250, exp should be lower than linear.
        let lin = t_at(AnnealStrategy::Linear, 250);
        let exp = t_at(AnnealStrategy::Exponential, 250);
        assert!(exp < lin);
    }

    #[test]
    fn invalid_zero_total_rejected() {
        assert_eq!(
            pick(AnnealStrategy::Linear, 10.0, 1.0, 0, 0),
            AnnealVerdict::InvalidStep
        );
    }

    #[test]
    fn invalid_zero_start_rejected() {
        assert_eq!(
            pick(AnnealStrategy::Linear, 0.0, 1.0, 0, 100),
            AnnealVerdict::InvalidStartTemp
        );
    }

    #[test]
    fn invalid_zero_end_rejected() {
        assert_eq!(
            pick(AnnealStrategy::Linear, 10.0, 0.0, 0, 100),
            AnnealVerdict::InvalidEndTemp
        );
    }

    #[test]
    fn nan_inputs_rejected() {
        assert_eq!(
            pick(AnnealStrategy::Linear, f64::NAN, 1.0, 0, 100),
            AnnealVerdict::InvalidStartTemp
        );
    }

    #[test]
    fn step_above_total_clamped() {
        let v = pick(AnnealStrategy::Linear, 10.0, 1.0, 5000, 1000);
        if let AnnealVerdict::Ok { temperature } = v {
            assert!((temperature - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn linear_monotone_decreasing() {
        let t1 = t_at(AnnealStrategy::Linear, 100);
        let t2 = t_at(AnnealStrategy::Linear, 500);
        let t3 = t_at(AnnealStrategy::Linear, 900);
        assert!(t1 > t2);
        assert!(t2 > t3);
    }
}
