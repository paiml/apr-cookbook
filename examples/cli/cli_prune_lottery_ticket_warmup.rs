//! # apr prune --lottery-ticket — Iterative Magnitude Warmup
//!
//! Lottery Ticket Hypothesis (Frankle & Carbin 2018): train k steps,
//! prune p% by magnitude, rewind to step k, repeat. This recipe builds
//! the warmup-step validator + cycle planner.
//!
//! Demonstrates the **PRUNE.6** recipe for PMAT-113 (apr prune coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PRUNE-001 + Frankle & Carbin 2018 (Lottery Ticket)
//!
//! Run with: cargo run --example cli_prune_lottery_ticket_warmup
//!
//! Added by PMAT-113 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WarmupVerdict {
    Ok,
    WarmupTooShort,
    WarmupExceedsTotal,
    InvalidPruneFraction,
    InvalidIterations,
}

const MIN_WARMUP_STEPS: u32 = 100;

pub fn validate(
    warmup_steps: u32,
    total_steps: u32,
    prune_fraction: f64,
    iterations: u32,
) -> WarmupVerdict {
    if warmup_steps < MIN_WARMUP_STEPS {
        return WarmupVerdict::WarmupTooShort;
    }
    if warmup_steps >= total_steps {
        return WarmupVerdict::WarmupExceedsTotal;
    }
    if !prune_fraction.is_finite() || !(0.0..1.0).contains(&prune_fraction) {
        return WarmupVerdict::InvalidPruneFraction;
    }
    if iterations == 0 {
        return WarmupVerdict::InvalidIterations;
    }
    WarmupVerdict::Ok
}

pub fn final_density(prune_fraction: f64, iterations: u32) -> Option<f64> {
    if !prune_fraction.is_finite() || !(0.0..1.0).contains(&prune_fraction) {
        return None;
    }
    let keep = 1.0 - prune_fraction;
    Some(keep.powi(iterations as i32))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_prune_lottery_ticket_warmup")?;

    let cases = [
        (200u32, 5000, 0.2, 5),
        (50, 5000, 0.2, 5),
        (200, 100, 0.2, 5),
        (200, 5000, 1.5, 5),
        (200, 5000, 0.2, 0),
    ];
    for (w, t, p, i) in cases {
        println!(
            "w={w} t={t} p={p} i={i} → {:?}  density={:?}",
            validate(w, t, p, i),
            final_density(p, i)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_config_passes() {
        assert_eq!(validate(200, 5000, 0.2, 5), WarmupVerdict::Ok);
    }

    #[test]
    fn warmup_too_short_rejected() {
        // < 100 steps gives unreliable rewind point.
        assert_eq!(validate(50, 5000, 0.2, 5), WarmupVerdict::WarmupTooShort);
    }

    #[test]
    fn warmup_at_or_above_total_rejected() {
        assert_eq!(
            validate(5000, 5000, 0.2, 5),
            WarmupVerdict::WarmupExceedsTotal
        );
        assert_eq!(
            validate(6000, 5000, 0.2, 5),
            WarmupVerdict::WarmupExceedsTotal
        );
    }

    #[test]
    fn prune_fraction_at_one_rejected() {
        // p=1.0 prunes everything → no model left.
        assert_eq!(
            validate(200, 5000, 1.0, 5),
            WarmupVerdict::InvalidPruneFraction
        );
    }

    #[test]
    fn negative_prune_fraction_rejected() {
        assert_eq!(
            validate(200, 5000, -0.1, 5),
            WarmupVerdict::InvalidPruneFraction
        );
    }

    #[test]
    fn zero_iterations_rejected() {
        assert_eq!(
            validate(200, 5000, 0.2, 0),
            WarmupVerdict::InvalidIterations
        );
    }

    #[test]
    fn final_density_compounds() {
        // 20% prune × 5 iterations: 0.8^5 ≈ 0.328.
        let d = final_density(0.2, 5).unwrap();
        assert!((d - 0.32768).abs() < 1e-6);
    }

    #[test]
    fn final_density_zero_iterations_is_one() {
        assert_eq!(final_density(0.5, 0), Some(1.0));
    }

    #[test]
    fn final_density_invalid_prune_yields_none() {
        assert!(final_density(1.5, 3).is_none());
        assert!(final_density(-0.1, 3).is_none());
    }
}
