//! # apr distill — Soft + Hard Loss Combiner
//!
//! Distillation loss = α · KL(student ‖ teacher · T) · T² + (1 − α) · CE(student, true_label).
//! The T² scaling preserves gradient magnitude across temperature changes.
//! This recipe builds the combiner with explicit T² weighting and
//! validates α ∈ [0, 1].
//!
//! Demonstrates the **DISTILL.5** recipe for PMAT-113 (apr distill coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DISTILL-001 + Hinton et al. 2015 (Distillation)
//!
//! Run with: cargo run --example cli_distill_loss_combiner
//!
//! Added by PMAT-113 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CombineVerdict {
    Ok(f64),
    AlphaOutOfRange,
    NonPositiveTemperature,
    NonFiniteLoss,
}

pub fn combine_loss(kl_loss: f64, ce_loss: f64, alpha: f64, temperature: f64) -> CombineVerdict {
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return CombineVerdict::AlphaOutOfRange;
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return CombineVerdict::NonPositiveTemperature;
    }
    if !kl_loss.is_finite() || !ce_loss.is_finite() {
        return CombineVerdict::NonFiniteLoss;
    }
    let combined = alpha * kl_loss * temperature * temperature + (1.0 - alpha) * ce_loss;
    CombineVerdict::Ok(combined)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_distill_loss_combiner")?;

    let cases = [
        ("typical α=0.7 T=4", 0.5, 1.2, 0.7, 4.0),
        ("hard-only α=0", 0.5, 1.2, 0.0, 4.0),
        ("soft-only α=1", 0.5, 1.2, 1.0, 4.0),
        ("α=1.5 (bad)", 0.5, 1.2, 1.5, 4.0),
        ("T=0 (bad)", 0.5, 1.2, 0.5, 0.0),
    ];
    for (label, kl, ce, a, t) in cases {
        println!("{label:>22}  →  {:?}", combine_loss(kl, ce, a, t));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combiner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn alpha_zero_pure_hard_loss() {
        // α=0 → ignore KL, return CE.
        let v = combine_loss(99.0, 1.5, 0.0, 4.0);
        assert_eq!(v, CombineVerdict::Ok(1.5));
    }

    #[test]
    fn alpha_one_pure_soft_loss_with_t_squared() {
        // α=1, T=4 → KL · 16.
        let v = combine_loss(1.0, 99.0, 1.0, 4.0);
        assert_eq!(v, CombineVerdict::Ok(16.0));
    }

    #[test]
    fn alpha_out_of_range_rejected() {
        assert_eq!(
            combine_loss(1.0, 1.0, 1.5, 4.0),
            CombineVerdict::AlphaOutOfRange
        );
        assert_eq!(
            combine_loss(1.0, 1.0, -0.1, 4.0),
            CombineVerdict::AlphaOutOfRange
        );
    }

    #[test]
    fn nan_alpha_rejected() {
        assert_eq!(
            combine_loss(1.0, 1.0, f64::NAN, 4.0),
            CombineVerdict::AlphaOutOfRange
        );
    }

    #[test]
    fn zero_or_negative_temp_rejected() {
        assert_eq!(
            combine_loss(1.0, 1.0, 0.5, 0.0),
            CombineVerdict::NonPositiveTemperature
        );
        assert_eq!(
            combine_loss(1.0, 1.0, 0.5, -1.0),
            CombineVerdict::NonPositiveTemperature
        );
    }

    #[test]
    fn nan_loss_rejected() {
        assert_eq!(
            combine_loss(f64::NAN, 1.0, 0.5, 4.0),
            CombineVerdict::NonFiniteLoss
        );
        assert_eq!(
            combine_loss(1.0, f64::INFINITY, 0.5, 4.0),
            CombineVerdict::NonFiniteLoss
        );
    }

    #[test]
    fn typical_distillation_combination() {
        // α=0.7, T=4, KL=0.5, CE=1.2 → 0.7·0.5·16 + 0.3·1.2 = 5.6 + 0.36 = 5.96
        let v = combine_loss(0.5, 1.2, 0.7, 4.0);
        if let CombineVerdict::Ok(loss) = v {
            assert!((loss - 5.96).abs() < 1e-9);
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn temperature_squared_amplification() {
        // T=2 → KL contribution is 4×; T=4 → 16×.
        let lo = combine_loss(1.0, 0.0, 1.0, 2.0);
        let hi = combine_loss(1.0, 0.0, 1.0, 4.0);
        if let (CombineVerdict::Ok(lo_v), CombineVerdict::Ok(hi_v)) = (lo, hi) {
            assert_eq!(hi_v / lo_v, 4.0);
        } else {
            panic!("expected Ok pair");
        }
    }
}
