//! # apr prune --gradual — Sparsity Ramp Validator
//!
//! Gradual pruning increases sparsity over training steps via a
//! polynomial schedule (Zhu & Gupta 2017): s(t) = s_f + (s_i − s_f) ·
//! (1 − (t − t_0) / Δt)³. This recipe validates the schedule envelope
//! and computes per-step sparsity.
//!
//! Demonstrates the **PRUNE.5** recipe for PMAT-113 (apr prune coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PRUNE-001 + Zhu & Gupta 2017 (gradual magnitude pruning)
//!
//! Run with: cargo run --example cli_prune_sparsity_ramp_validator
//!
//! Added by PMAT-113 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RampVerdict {
    Ok,
    InitialAboveFinal,
    SparsityOutOfRange,
    InvalidStepRange,
}

#[derive(Debug, Clone, Copy)]
pub struct RampConfig {
    pub initial_sparsity: f64,
    pub final_sparsity: f64,
    pub start_step: u32,
    pub end_step: u32,
}

pub fn validate(config: RampConfig) -> RampVerdict {
    for s in [config.initial_sparsity, config.final_sparsity] {
        if !s.is_finite() || !(0.0..=1.0).contains(&s) {
            return RampVerdict::SparsityOutOfRange;
        }
    }
    if config.initial_sparsity > config.final_sparsity {
        return RampVerdict::InitialAboveFinal;
    }
    if config.end_step <= config.start_step {
        return RampVerdict::InvalidStepRange;
    }
    RampVerdict::Ok
}

pub fn sparsity_at_step(config: RampConfig, step: u32) -> Option<f64> {
    if !matches!(validate(config), RampVerdict::Ok) {
        return None;
    }
    if step <= config.start_step {
        return Some(config.initial_sparsity);
    }
    if step >= config.end_step {
        return Some(config.final_sparsity);
    }
    let progress =
        f64::from(step - config.start_step) / f64::from(config.end_step - config.start_step);
    let factor = (1.0 - progress).powi(3);
    Some(config.final_sparsity + (config.initial_sparsity - config.final_sparsity) * factor)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_prune_sparsity_ramp_validator")?;

    let cfg = RampConfig {
        initial_sparsity: 0.0,
        final_sparsity: 0.9,
        start_step: 0,
        end_step: 1000,
    };
    println!("validate: {:?}", validate(cfg));
    for step in [0u32, 100, 500, 900, 1000, 5000] {
        println!(
            "step={step:>4} → s={:.4}",
            sparsity_at_step(cfg, step).unwrap()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> RampConfig {
        RampConfig {
            initial_sparsity: 0.0,
            final_sparsity: 0.9,
            start_step: 0,
            end_step: 1000,
        }
    }

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_config_passes() {
        assert_eq!(validate(cfg()), RampVerdict::Ok);
    }

    #[test]
    fn initial_above_final_rejected() {
        let mut c = cfg();
        c.initial_sparsity = 0.95;
        assert_eq!(validate(c), RampVerdict::InitialAboveFinal);
    }

    #[test]
    fn out_of_range_sparsity_rejected() {
        let mut c = cfg();
        c.final_sparsity = 1.5;
        assert_eq!(validate(c), RampVerdict::SparsityOutOfRange);
        c.final_sparsity = -0.1;
        assert_eq!(validate(c), RampVerdict::SparsityOutOfRange);
    }

    #[test]
    fn end_before_start_rejected() {
        let mut c = cfg();
        c.start_step = 1000;
        c.end_step = 500;
        assert_eq!(validate(c), RampVerdict::InvalidStepRange);
    }

    #[test]
    fn step_at_or_before_start_returns_initial() {
        assert_eq!(sparsity_at_step(cfg(), 0), Some(0.0));
    }

    #[test]
    fn step_at_or_after_end_returns_final() {
        assert_eq!(sparsity_at_step(cfg(), 1000), Some(0.9));
        assert_eq!(sparsity_at_step(cfg(), 5000), Some(0.9));
    }

    #[test]
    fn schedule_monotonically_increases() {
        let c = cfg();
        let s100 = sparsity_at_step(c, 100).unwrap();
        let s500 = sparsity_at_step(c, 500).unwrap();
        let s900 = sparsity_at_step(c, 900).unwrap();
        assert!(s100 < s500);
        assert!(s500 < s900);
    }

    #[test]
    fn invalid_config_yields_none() {
        let mut c = cfg();
        c.final_sparsity = 1.5;
        assert!(sparsity_at_step(c, 100).is_none());
    }

    #[test]
    fn cubic_schedule_steeper_at_end() {
        // Cubic decay from initial to final: most of the ramp happens early
        // (s grows fast at first). Verify mid-point is closer to final than midline.
        let c = cfg();
        let s_mid = sparsity_at_step(c, 500).unwrap();
        let linear_mid = (c.initial_sparsity + c.final_sparsity) / 2.0;
        // (1 - 0.5)^3 = 0.125 → s(0.5) = 0.9 + (0 - 0.9)·0.125 = 0.7875
        assert!(s_mid > linear_mid, "cubic should be above linear midpoint");
    }
}
