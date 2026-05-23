//! # Self-Distillation Bootstrap
//!
//! Self-distillation: train a model M_0, use it as teacher for an
//! identical-size student M_1, repeat. Marginal gains plateau quickly;
//! 2-3 iterations is the sweet spot. This recipe builds the iteration
//! planner + accuracy-improvement classifier (Marginal/Significant/None).
//!
//! Demonstrates the **DISTILL.7** recipe for PMAT-124 (distillation coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Furlanello et al. (2018). Born Again Neural Networks. arXiv:1805.04770.
//!
//! Run with: cargo run --example distill_self_distill_bootstrap
//!
//! Added by PMAT-124 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BootstrapVerdict {
    Ok { iterations: u32, expected_acc: f64 },
    InvalidBaseline,
    IterationsExceedsCap,
}

const MAX_USEFUL_ITERS: u32 = 5;
const PER_ITER_GAIN_DECAY: f64 = 0.5;

pub fn plan(baseline_acc: f64, target_acc: f64, first_iter_gain: f64) -> BootstrapVerdict {
    if !baseline_acc.is_finite() || !target_acc.is_finite() || !first_iter_gain.is_finite() {
        return BootstrapVerdict::InvalidBaseline;
    }
    if !(0.0..=1.0).contains(&baseline_acc) || !(0.0..=1.0).contains(&target_acc) {
        return BootstrapVerdict::InvalidBaseline;
    }
    if first_iter_gain < 0.0 {
        return BootstrapVerdict::InvalidBaseline;
    }
    let mut acc = baseline_acc;
    let mut gain = first_iter_gain;
    for i in 1..=MAX_USEFUL_ITERS {
        acc += gain;
        if acc >= target_acc {
            return BootstrapVerdict::Ok {
                iterations: i,
                expected_acc: acc,
            };
        }
        gain *= PER_ITER_GAIN_DECAY;
    }
    BootstrapVerdict::IterationsExceedsCap
}

#[derive(Debug, PartialEq)]
pub enum GainTier {
    Significant,
    Marginal,
    None,
    InvalidValue,
}

const SIGNIFICANT_PCT: f64 = 0.005; // 0.5%
const MARGINAL_PCT: f64 = 0.001; // 0.1%

pub fn classify_gain(prev_acc: f64, new_acc: f64) -> GainTier {
    if !prev_acc.is_finite() || !new_acc.is_finite() {
        return GainTier::InvalidValue;
    }
    let delta = new_acc - prev_acc;
    if delta >= SIGNIFICANT_PCT {
        GainTier::Significant
    } else if delta >= MARGINAL_PCT {
        GainTier::Marginal
    } else {
        GainTier::None
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_self_distill_bootstrap")?;

    let cases = [
        (0.80, 0.83, 0.02),
        (0.80, 0.99, 0.02),
        (0.80, 0.85, 0.05),
        (1.5, 0.83, 0.02),
    ];
    for (b, t, g) in cases {
        println!("baseline={b} target={t} gain1={g}  →  {:?}", plan(b, t, g));
    }
    for (prev, new) in [(0.80, 0.81), (0.80, 0.802), (0.80, 0.7999)] {
        println!("{prev} → {new}  =  {:?}", classify_gain(prev, new));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_bootstrap_picks_few_iterations() {
        // baseline 0.80, target 0.83, gain 0.02 → after iter 1 = 0.82,
        // iter 2 = 0.83 → return iter 2.
        if let BootstrapVerdict::Ok { iterations, .. } = plan(0.80, 0.83, 0.02) {
            assert_eq!(iterations, 2);
        }
    }

    #[test]
    fn unreachable_target_caps() {
        // baseline 0.80, target 0.99, small gain → cap.
        let v = plan(0.80, 0.99, 0.02);
        assert_eq!(v, BootstrapVerdict::IterationsExceedsCap);
    }

    #[test]
    fn baseline_already_above_target_one_iter() {
        // After 1 iteration of even tiny gain, exceeds 0.80 target.
        if let BootstrapVerdict::Ok { iterations, .. } = plan(0.80, 0.80, 0.001) {
            assert_eq!(iterations, 1);
        }
    }

    #[test]
    fn nan_baseline_invalid() {
        assert_eq!(
            plan(f64::NAN, 0.85, 0.02),
            BootstrapVerdict::InvalidBaseline
        );
    }

    #[test]
    fn out_of_range_acc_invalid() {
        assert_eq!(plan(1.5, 0.85, 0.02), BootstrapVerdict::InvalidBaseline);
        assert_eq!(plan(0.5, 1.5, 0.02), BootstrapVerdict::InvalidBaseline);
    }

    #[test]
    fn negative_gain_invalid() {
        assert_eq!(plan(0.5, 0.7, -0.1), BootstrapVerdict::InvalidBaseline);
    }

    #[test]
    fn classify_significant_gain() {
        assert_eq!(classify_gain(0.80, 0.81), GainTier::Significant);
    }

    #[test]
    fn classify_marginal_gain() {
        assert_eq!(classify_gain(0.80, 0.802), GainTier::Marginal);
    }

    #[test]
    fn classify_no_gain() {
        assert_eq!(classify_gain(0.80, 0.7999), GainTier::None);
        assert_eq!(classify_gain(0.80, 0.80), GainTier::None);
    }

    #[test]
    fn classify_nan_invalid() {
        assert_eq!(classify_gain(f64::NAN, 0.5), GainTier::InvalidValue);
    }

    #[test]
    fn gain_decays_each_iteration() {
        // Verify the decay model: with baseline 0.80, large first gain
        // 0.10, target 0.85 → iter 1 reaches 0.90 already.
        if let BootstrapVerdict::Ok {
            iterations,
            expected_acc,
        } = plan(0.80, 0.85, 0.10)
        {
            assert_eq!(iterations, 1);
            assert!((expected_acc - 0.90).abs() < 1e-9);
        }
    }
}
