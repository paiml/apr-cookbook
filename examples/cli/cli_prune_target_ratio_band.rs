//! # apr prune — `--target-ratio` Band Validator
//!
//! `apr prune --target-ratio <R>` accepts R ∈ (0, 1). R = 0 means "do
//! nothing" (no-op pruning); R = 1 means "remove everything" (broken
//! model). Both extremes reject. Above 0.9 is heroic and gets a warning
//! (model usually unusable beyond that).
//!
//! Demonstrates the **PRUNE.8** recipe for PMAT-104 (apr prune coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GH-247
//!
//! Run with: cargo run --example cli_prune_target_ratio_band
//!
//! Added by PMAT-104 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RatioVerdict {
    Ok,
    Aggressive { ratio: f64 },
    NoOp,
    BrokenModel,
    OutOfBand,
    NotFinite,
}

const HEROIC_FLOOR: f64 = 0.9;

pub fn validate_ratio(r: f64) -> RatioVerdict {
    if !r.is_finite() {
        return RatioVerdict::NotFinite;
    }
    if r == 0.0 {
        return RatioVerdict::NoOp;
    }
    if r >= 1.0 {
        return RatioVerdict::BrokenModel;
    }
    if r < 0.0 {
        return RatioVerdict::OutOfBand;
    }
    if r >= HEROIC_FLOOR {
        return RatioVerdict::Aggressive { ratio: r };
    }
    RatioVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_prune_target_ratio_band")?;

    for r in [0.0_f64, 0.1, 0.5, 0.85, 0.9, 0.95, 1.0, 1.5, -0.1, f64::NAN] {
        println!("--target-ratio {r:>6.2}  →  {:?}", validate_ratio(r));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn zero_is_noop() {
        assert_eq!(validate_ratio(0.0), RatioVerdict::NoOp);
    }

    #[test]
    fn one_is_broken_model() {
        assert_eq!(validate_ratio(1.0), RatioVerdict::BrokenModel);
        assert_eq!(validate_ratio(1.5), RatioVerdict::BrokenModel);
    }

    #[test]
    fn negative_rejected() {
        assert_eq!(validate_ratio(-0.1), RatioVerdict::OutOfBand);
    }

    #[test]
    fn nan_or_inf_rejected() {
        assert_eq!(validate_ratio(f64::NAN), RatioVerdict::NotFinite);
        assert_eq!(validate_ratio(f64::INFINITY), RatioVerdict::NotFinite);
    }

    #[test]
    fn happy_band_passes() {
        assert_eq!(validate_ratio(0.5), RatioVerdict::Ok);
        assert_eq!(validate_ratio(0.1), RatioVerdict::Ok);
        assert_eq!(validate_ratio(0.85), RatioVerdict::Ok);
    }

    #[test]
    fn boundary_at_0_9_is_aggressive() {
        // ≥ 0.9 → warn the operator that the model is likely unusable.
        let v = validate_ratio(0.9);
        assert!(matches!(v, RatioVerdict::Aggressive { .. }));
    }

    #[test]
    fn aggressive_band_below_one() {
        // 0.99 is aggressive but not BrokenModel (would need == 1.0 or higher).
        assert!(matches!(
            validate_ratio(0.99),
            RatioVerdict::Aggressive { .. }
        ));
    }

    #[test]
    fn boundary_below_0_9_is_ok() {
        assert_eq!(validate_ratio(0.89), RatioVerdict::Ok);
    }
}
