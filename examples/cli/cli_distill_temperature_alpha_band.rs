//! # apr distill — `--temperature` × `--alpha` Band Validator
//!
//! `apr distill --temperature <T> --alpha <A>` controls the KD loss:
//! `loss = α · KL(softmax(student_logits/T), softmax(teacher_logits/T)) · T² + (1-α) · CE(student, label)`.
//! Sane bands: T ∈ [1.0, 10.0], α ∈ [0.0, 1.0]. T < 1 collapses to argmax;
//! T > 10 makes the soft targets uninformative.
//!
//! Demonstrates the **DISTILL.10** recipe for PMAT-106 (apr distill coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender ALB-011 + Hinton et al. (2015) §2 (KD loss formula)
//!
//! Run with: cargo run --example cli_distill_temperature_alpha_band
//!
//! Added by PMAT-106 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BandVerdict {
    Ok,
    TemperatureTooLow,
    TemperatureTooHigh,
    AlphaOutOfBand,
    NotFinite,
}

const T_MIN: f64 = 1.0;
const T_MAX: f64 = 10.0;

pub fn validate(temperature: f64, alpha: f64) -> BandVerdict {
    if !temperature.is_finite() || !alpha.is_finite() {
        return BandVerdict::NotFinite;
    }
    if !(0.0..=1.0).contains(&alpha) {
        return BandVerdict::AlphaOutOfBand;
    }
    if temperature < T_MIN {
        return BandVerdict::TemperatureTooLow;
    }
    if temperature > T_MAX {
        return BandVerdict::TemperatureTooHigh;
    }
    BandVerdict::Ok
}

pub fn kd_weighting(temperature: f64, alpha: f64) -> Option<(f64, f64)> {
    if validate(temperature, alpha) != BandVerdict::Ok {
        return None;
    }
    let kd = alpha * temperature * temperature;
    let ce = 1.0 - alpha;
    Some((kd, ce))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_distill_temperature_alpha_band")?;

    let cases = [
        ("default 3.0 / 0.7", 3.0, 0.7),
        ("T=0.5 (too low)", 0.5, 0.7),
        ("T=15 (too high)", 15.0, 0.7),
        ("alpha=1.5", 3.0, 1.5),
        ("alpha=-0.1", 3.0, -0.1),
        ("nan T", f64::NAN, 0.7),
    ];
    for (label, t, a) in cases {
        println!(
            "{label:>20}  T={t:>5.1} α={a:>5.2}  →  {:?}  weights={:?}",
            validate(t, a),
            kd_weighting(t, a)
        );
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
    fn default_3_0_0_7_passes() {
        // CLI defaults — must always be valid.
        assert_eq!(validate(3.0, 0.7), BandVerdict::Ok);
    }

    #[test]
    fn temperature_below_1_rejected() {
        assert_eq!(validate(0.5, 0.7), BandVerdict::TemperatureTooLow);
    }

    #[test]
    fn temperature_above_10_rejected() {
        assert_eq!(validate(15.0, 0.7), BandVerdict::TemperatureTooHigh);
    }

    #[test]
    fn alpha_above_1_rejected() {
        assert_eq!(validate(3.0, 1.5), BandVerdict::AlphaOutOfBand);
    }

    #[test]
    fn alpha_below_0_rejected() {
        assert_eq!(validate(3.0, -0.1), BandVerdict::AlphaOutOfBand);
    }

    #[test]
    fn nan_anywhere_rejected() {
        assert_eq!(validate(f64::NAN, 0.7), BandVerdict::NotFinite);
        assert_eq!(validate(3.0, f64::NAN), BandVerdict::NotFinite);
    }

    #[test]
    fn boundaries_pass() {
        // T = 1.0, T = 10.0, α = 0.0, α = 1.0 all permitted.
        assert_eq!(validate(1.0, 0.7), BandVerdict::Ok);
        assert_eq!(validate(10.0, 0.7), BandVerdict::Ok);
        assert_eq!(validate(3.0, 0.0), BandVerdict::Ok);
        assert_eq!(validate(3.0, 1.0), BandVerdict::Ok);
    }

    #[test]
    fn kd_weighting_includes_t_squared() {
        // Per Hinton: KD term scales by T² to keep gradient magnitude balanced.
        let (kd, ce) = kd_weighting(3.0, 0.7).unwrap();
        assert!((kd - 0.7 * 9.0).abs() < 1e-9);
        assert!((ce - 0.3).abs() < 1e-9);
    }

    #[test]
    fn kd_weighting_returns_none_for_invalid() {
        assert!(kd_weighting(0.5, 0.7).is_none());
        assert!(kd_weighting(3.0, 1.5).is_none());
    }
}
