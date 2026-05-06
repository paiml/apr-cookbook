//! # apr export --onnx-opset — Opset Version Compatibility Matrix
//!
//! ONNX opsets evolve over time. Older runtimes pin to lower opsets;
//! newer ops require higher opsets. Rules: opset must be in [9, 21]
//! (current as of 2026); attention/SDPA ops require opset ≥ 14;
//! quantization ops require opset ≥ 13. This recipe builds the matrix.
//!
//! Demonstrates the **EXP.5** recipe for PMAT-117 (apr export coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender EXP-001 + ONNX opset versioning policy
//!
//! Run with: cargo run --example cli_export_opset_compat_matrix
//!
//! Added by PMAT-117 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MIN_OPSET: u32 = 9;
const MAX_OPSET: u32 = 21;
const SDPA_FLOOR: u32 = 14;
const QUANT_FLOOR: u32 = 13;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureRequirement {
    None,
    SdpaAttention,
    Quantization,
}

#[derive(Debug, PartialEq)]
pub enum OpsetVerdict {
    Ok,
    BelowFloor {
        recommended: u32,
    },
    AboveCeiling {
        recommended: u32,
    },
    FeatureNotAvailable {
        feature: FeatureRequirement,
        requires: u32,
    },
}

pub fn validate(opset: u32, requirement: FeatureRequirement) -> OpsetVerdict {
    if opset < MIN_OPSET {
        return OpsetVerdict::BelowFloor {
            recommended: MIN_OPSET,
        };
    }
    if opset > MAX_OPSET {
        return OpsetVerdict::AboveCeiling {
            recommended: MAX_OPSET,
        };
    }
    let required = match requirement {
        FeatureRequirement::None => 0,
        FeatureRequirement::SdpaAttention => SDPA_FLOOR,
        FeatureRequirement::Quantization => QUANT_FLOOR,
    };
    if opset < required {
        return OpsetVerdict::FeatureNotAvailable {
            feature: requirement,
            requires: required,
        };
    }
    OpsetVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_export_opset_compat_matrix")?;

    let cases = [
        (8, FeatureRequirement::None),
        (15, FeatureRequirement::SdpaAttention),
        (12, FeatureRequirement::SdpaAttention),
        (12, FeatureRequirement::Quantization),
        (25, FeatureRequirement::None),
        (18, FeatureRequirement::None),
    ];
    for (op, req) in cases {
        println!("opset={op} feature={req:?}  →  {:?}", validate(op, req));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_opset_passes() {
        assert_eq!(validate(18, FeatureRequirement::None), OpsetVerdict::Ok);
    }

    #[test]
    fn below_floor_rejected() {
        let v = validate(8, FeatureRequirement::None);
        assert!(matches!(v, OpsetVerdict::BelowFloor { .. }));
    }

    #[test]
    fn above_ceiling_rejected() {
        let v = validate(25, FeatureRequirement::None);
        assert!(matches!(v, OpsetVerdict::AboveCeiling { .. }));
    }

    #[test]
    fn sdpa_below_14_rejected() {
        let v = validate(12, FeatureRequirement::SdpaAttention);
        assert!(matches!(
            v,
            OpsetVerdict::FeatureNotAvailable {
                requires: SDPA_FLOOR,
                ..
            }
        ));
    }

    #[test]
    fn sdpa_at_14_passes() {
        assert_eq!(
            validate(14, FeatureRequirement::SdpaAttention),
            OpsetVerdict::Ok
        );
    }

    #[test]
    fn quantization_below_13_rejected() {
        let v = validate(12, FeatureRequirement::Quantization);
        assert!(matches!(v, OpsetVerdict::FeatureNotAvailable { .. }));
    }

    #[test]
    fn quantization_at_13_passes() {
        assert_eq!(
            validate(13, FeatureRequirement::Quantization),
            OpsetVerdict::Ok
        );
    }

    #[test]
    fn at_floor_with_no_feature_passes() {
        assert_eq!(
            validate(MIN_OPSET, FeatureRequirement::None),
            OpsetVerdict::Ok
        );
    }

    #[test]
    fn at_ceiling_with_features_passes() {
        // Maximum opset supports all features.
        assert_eq!(
            validate(MAX_OPSET, FeatureRequirement::SdpaAttention),
            OpsetVerdict::Ok
        );
        assert_eq!(
            validate(MAX_OPSET, FeatureRequirement::Quantization),
            OpsetVerdict::Ok
        );
    }
}
