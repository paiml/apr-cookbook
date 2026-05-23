//! # Training LR Warmup-Decay Combo Scheduler
//!
//! Composite schedulers chain warmup with one of {linear, cosine,
//! polynomial, constant} decay. This recipe builds the picker that
//! returns the recommended composite given total steps + warmup
//! fraction + decay style.
//!
//! Demonstrates the **TRAIN.14** recipe for PMAT-132 (training coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Goyal et al. (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.
//!
//! Run with: cargo run --example training_lr_combo_scheduler
//!
//! Added by PMAT-132 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecayStyle {
    Linear,
    Cosine,
    Polynomial,
    Constant,
}

#[derive(Debug, PartialEq)]
pub enum ComboVerdict {
    Ok {
        warmup_steps: u32,
        decay_style: DecayStyle,
    },
    InvalidTotal,
    WarmupExceedsTotal,
    WarmupFractionOutOfRange,
}

pub const DEFAULT_WARMUP_FRACTION: f64 = 0.05;

pub fn pick(total_steps: u32, warmup_fraction: f64, decay_style: DecayStyle) -> ComboVerdict {
    if total_steps == 0 {
        return ComboVerdict::InvalidTotal;
    }
    if !warmup_fraction.is_finite() || !(0.0..=0.5).contains(&warmup_fraction) {
        return ComboVerdict::WarmupFractionOutOfRange;
    }
    let warmup_steps = (f64::from(total_steps) * warmup_fraction).round() as u32;
    if warmup_steps >= total_steps {
        return ComboVerdict::WarmupExceedsTotal;
    }
    ComboVerdict::Ok {
        warmup_steps,
        decay_style,
    }
}

pub fn auto_pick_decay(num_epochs: u32) -> DecayStyle {
    if num_epochs <= 1 {
        DecayStyle::Constant
    } else if num_epochs <= 10 {
        DecayStyle::Linear
    } else {
        DecayStyle::Cosine
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("training_lr_combo_scheduler")?;

    let cases = [
        (10_000u32, 0.05, DecayStyle::Cosine),
        (10_000, 0.6, DecayStyle::Cosine),
        (0, 0.05, DecayStyle::Linear),
    ];
    for (total, frac, style) in cases {
        println!(
            "total={total} frac={frac} style={style:?} → {:?}",
            pick(total, frac, style)
        );
    }

    for ep in [1u32, 5, 100] {
        println!("epochs={ep} → auto_decay={:?}", auto_pick_decay(ep));
    }
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
    fn typical_pick_succeeds() {
        let v = pick(10_000, DEFAULT_WARMUP_FRACTION, DecayStyle::Cosine);
        assert!(matches!(
            v,
            ComboVerdict::Ok {
                warmup_steps: 500,
                ..
            }
        ));
    }

    #[test]
    fn warmup_rounds_correctly() {
        // 0.05 × 1234 = 61.7 → rounds to 62.
        let v = pick(1234, 0.05, DecayStyle::Cosine);
        if let ComboVerdict::Ok { warmup_steps, .. } = v {
            assert_eq!(warmup_steps, 62);
        }
    }

    #[test]
    fn zero_total_invalid() {
        assert_eq!(
            pick(0, 0.05, DecayStyle::Linear),
            ComboVerdict::InvalidTotal
        );
    }

    #[test]
    fn fraction_over_half_rejected() {
        let v = pick(10_000, 0.6, DecayStyle::Cosine);
        assert_eq!(v, ComboVerdict::WarmupFractionOutOfRange);
    }

    #[test]
    fn negative_fraction_rejected() {
        let v = pick(10_000, -0.1, DecayStyle::Cosine);
        assert_eq!(v, ComboVerdict::WarmupFractionOutOfRange);
    }

    #[test]
    fn warmup_at_50_pct_rejected_when_rounding_meets_total() {
        // 0.5 × 1 = 0.5 → rounds to 1 = total → rejected by WarmupExceedsTotal.
        let v = pick(1, 0.5, DecayStyle::Cosine);
        assert!(matches!(v, ComboVerdict::WarmupExceedsTotal));
    }

    #[test]
    fn auto_pick_short_runs_constant() {
        assert_eq!(auto_pick_decay(1), DecayStyle::Constant);
    }

    #[test]
    fn auto_pick_medium_runs_linear() {
        assert_eq!(auto_pick_decay(5), DecayStyle::Linear);
    }

    #[test]
    fn auto_pick_long_runs_cosine() {
        assert_eq!(auto_pick_decay(100), DecayStyle::Cosine);
    }

    #[test]
    fn boundary_at_10_epochs_picks_linear() {
        assert_eq!(auto_pick_decay(10), DecayStyle::Linear);
        assert_eq!(auto_pick_decay(11), DecayStyle::Cosine);
    }
}
