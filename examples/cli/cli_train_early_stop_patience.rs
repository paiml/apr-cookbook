//! # apr train --early-stop-patience — Patience Window Validator
//!
//! Early stopping monitors a validation metric for K consecutive
//! epochs without improvement. Patience too low: stops on noise; too
//! high: wastes compute past the optimum. Floor: 3 (statistical
//! minimum); ceiling: 50 (signal/noise floor for most schedules);
//! default: 10. This recipe codifies the envelope + the should-stop
//! decision.
//!
//! Demonstrates the **TRAIN.4** recipe for PMAT-116 (apr train coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender TRAIN-001 + Prechelt 1998 (early stopping)
//!
//! Run with: cargo run --example cli_train_early_stop_patience
//!
//! Added by PMAT-116 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PatienceVerdict {
    Ok,
    BelowFloor { recommended: u32 },
    AboveCeiling { recommended: u32 },
    InvalidZero,
}

const MIN_PATIENCE: u32 = 3;
const MAX_PATIENCE: u32 = 50;
const DEFAULT_PATIENCE: u32 = 10;

pub fn validate(patience: u32) -> PatienceVerdict {
    if patience == 0 {
        return PatienceVerdict::InvalidZero;
    }
    if patience < MIN_PATIENCE {
        return PatienceVerdict::BelowFloor {
            recommended: DEFAULT_PATIENCE,
        };
    }
    if patience > MAX_PATIENCE {
        return PatienceVerdict::AboveCeiling {
            recommended: MAX_PATIENCE,
        };
    }
    PatienceVerdict::Ok
}

pub fn should_stop(val_history: &[f64], patience: u32, lower_is_better: bool) -> bool {
    if val_history.len() < (patience as usize + 1) {
        return false;
    }
    let cmp = |a: f64, b: f64| if lower_is_better { a < b } else { a > b };
    let baseline_idx = val_history.len() - 1 - patience as usize;
    let baseline = val_history[baseline_idx];
    !val_history
        .iter()
        .skip(baseline_idx + 1)
        .any(|&v| cmp(v, baseline))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_train_early_stop_patience")?;

    for p in [0u32, 2, 3, 10, 50, 100] {
        println!("patience={p:>3}  →  {:?}", validate(p));
    }

    // Loss decreasing → don't stop.
    let dec = [1.0, 0.9, 0.8, 0.7, 0.6];
    println!(
        "decreasing should_stop(p=3, lower)? {}",
        should_stop(&dec, 3, true)
    );
    // Loss plateaued → stop after patience.
    let flat = [1.0, 0.5, 0.5, 0.5, 0.5];
    println!(
        "flat should_stop(p=3, lower)? {}",
        should_stop(&flat, 3, true)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn zero_invalid() {
        assert_eq!(validate(0), PatienceVerdict::InvalidZero);
    }

    #[test]
    fn under_floor_rejected() {
        let v = validate(2);
        assert!(matches!(v, PatienceVerdict::BelowFloor { .. }));
    }

    #[test]
    fn at_floor_passes() {
        assert_eq!(validate(MIN_PATIENCE), PatienceVerdict::Ok);
    }

    #[test]
    fn default_passes() {
        assert_eq!(validate(DEFAULT_PATIENCE), PatienceVerdict::Ok);
    }

    #[test]
    fn at_ceiling_passes() {
        assert_eq!(validate(MAX_PATIENCE), PatienceVerdict::Ok);
    }

    #[test]
    fn above_ceiling_rejected() {
        let v = validate(100);
        assert!(matches!(v, PatienceVerdict::AboveCeiling { .. }));
    }

    #[test]
    fn decreasing_loss_does_not_stop() {
        let dec = [1.0, 0.9, 0.8, 0.7, 0.6];
        assert!(!should_stop(&dec, 3, true));
    }

    #[test]
    fn plateaued_loss_stops_after_patience() {
        let flat = [1.0, 0.5, 0.5, 0.5, 0.5];
        // Baseline at idx 1 (= 0.5); next 3 are all 0.5 (no improvement) → stop.
        assert!(should_stop(&flat, 3, true));
    }

    #[test]
    fn short_history_below_patience_does_not_stop() {
        let two = [1.0, 0.5];
        assert!(!should_stop(&two, 3, true));
    }

    #[test]
    fn higher_is_better_mode() {
        // For accuracy: want max. Plateau at 0.85 should stop.
        let acc = [0.6, 0.85, 0.85, 0.85, 0.85];
        assert!(should_stop(&acc, 3, false));
        let rising = [0.6, 0.7, 0.8, 0.9, 0.95];
        assert!(!should_stop(&rising, 3, false));
    }
}
