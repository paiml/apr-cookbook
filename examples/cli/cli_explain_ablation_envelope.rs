//! # apr explain --ablation — Per-Component Mask Envelope
//!
//! Ablation drops one component (head, layer, neuron) and measures
//! prediction delta. Constraints: target index < total components;
//! mask value ∈ {0.0 (zero-out), mean, noise}; baseline reference
//! required. This recipe builds the envelope.
//!
//! Demonstrates the **EXP.6** recipe for PMAT-114 (apr explain coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender EXP-001 + Olsson et al. 2022 (mech interp ablations)
//!
//! Run with: cargo run --example cli_explain_ablation_envelope
//!
//! Added by PMAT-114 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskMode {
    ZeroOut,
    MeanReplace,
    GaussianNoise,
}

impl MaskMode {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "zero" => Some(MaskMode::ZeroOut),
            "mean" => Some(MaskMode::MeanReplace),
            "noise" => Some(MaskMode::GaussianNoise),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum AblateVerdict {
    Ok,
    TargetOutOfRange { target: usize, total: usize },
    EmptyArchitecture,
    NoiseRequiresStdDev,
}

pub fn validate(
    target_index: usize,
    total_components: usize,
    mode: MaskMode,
    noise_std: Option<f64>,
) -> AblateVerdict {
    if total_components == 0 {
        return AblateVerdict::EmptyArchitecture;
    }
    if target_index >= total_components {
        return AblateVerdict::TargetOutOfRange {
            target: target_index,
            total: total_components,
        };
    }
    if mode == MaskMode::GaussianNoise {
        match noise_std {
            Some(s) if s.is_finite() && s > 0.0 => {}
            _ => return AblateVerdict::NoiseRequiresStdDev,
        }
    }
    AblateVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_explain_ablation_envelope")?;

    let cases = [
        ("ablate head 5/12", 5, 12, MaskMode::ZeroOut, None),
        ("ablate head 99/12 (bad)", 99, 12, MaskMode::ZeroOut, None),
        ("noise no std", 0, 12, MaskMode::GaussianNoise, None),
        ("noise with std", 0, 12, MaskMode::GaussianNoise, Some(0.1)),
    ];
    for (label, t, n, m, ns) in cases {
        println!("{label:>26}  →  {:?}", validate(t, n, m, ns));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_zero_ablation_passes() {
        assert_eq!(validate(5, 12, MaskMode::ZeroOut, None), AblateVerdict::Ok);
    }

    #[test]
    fn target_out_of_range_rejected() {
        let v = validate(99, 12, MaskMode::ZeroOut, None);
        assert!(matches!(v, AblateVerdict::TargetOutOfRange { .. }));
    }

    #[test]
    fn empty_architecture_rejected() {
        assert_eq!(
            validate(0, 0, MaskMode::ZeroOut, None),
            AblateVerdict::EmptyArchitecture
        );
    }

    #[test]
    fn noise_without_std_rejected() {
        assert_eq!(
            validate(0, 12, MaskMode::GaussianNoise, None),
            AblateVerdict::NoiseRequiresStdDev
        );
    }

    #[test]
    fn noise_with_zero_std_rejected() {
        assert_eq!(
            validate(0, 12, MaskMode::GaussianNoise, Some(0.0)),
            AblateVerdict::NoiseRequiresStdDev
        );
    }

    #[test]
    fn noise_with_nan_std_rejected() {
        assert_eq!(
            validate(0, 12, MaskMode::GaussianNoise, Some(f64::NAN)),
            AblateVerdict::NoiseRequiresStdDev
        );
    }

    #[test]
    fn mean_replace_no_std_required() {
        // MeanReplace doesn't require noise_std.
        assert_eq!(
            validate(0, 12, MaskMode::MeanReplace, None),
            AblateVerdict::Ok
        );
    }

    #[test]
    fn boundary_at_total_minus_one_passes() {
        assert_eq!(validate(11, 12, MaskMode::ZeroOut, None), AblateVerdict::Ok);
    }

    #[test]
    fn known_modes_round_trip() {
        for s in ["zero", "mean", "noise"] {
            assert!(MaskMode::from_str_strict(s).is_some());
        }
        assert!(MaskMode::from_str_strict("permute").is_none());
    }
}
