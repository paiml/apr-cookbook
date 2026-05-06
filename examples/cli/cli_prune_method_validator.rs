//! # apr prune — `--method` Validator + Compatibility Matrix
//!
//! `apr prune --method <M>` accepts {magnitude, structured, depth, width,
//! wanda, sparsegpt}. Each method has different requirements: `wanda` and
//! `sparsegpt` require `--calibration`, `depth` requires `--remove-layers`,
//! `width` requires `--target-ratio` ≠ 0. This recipe builds the
//! compatibility matrix.
//!
//! Demonstrates the **PRUNE.6** recipe for PMAT-104 (apr prune coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GH-247 + Wanda (Sun 2024) + SparseGPT (Frantar 2023)
//!
//! Run with: cargo run --example cli_prune_method_validator
//!
//! Added by PMAT-104 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneMethod {
    Magnitude,
    Structured,
    Depth,
    Width,
    Wanda,
    SparseGpt,
}

impl PruneMethod {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "magnitude" => Some(PruneMethod::Magnitude),
            "structured" => Some(PruneMethod::Structured),
            "depth" => Some(PruneMethod::Depth),
            "width" => Some(PruneMethod::Width),
            "wanda" => Some(PruneMethod::Wanda),
            "sparsegpt" => Some(PruneMethod::SparseGpt),
            _ => None,
        }
    }

    pub fn requires_calibration(self) -> bool {
        matches!(self, PruneMethod::Wanda | PruneMethod::SparseGpt)
    }

    pub fn requires_layer_spec(self) -> bool {
        matches!(self, PruneMethod::Depth)
    }

    pub fn requires_target_ratio(self) -> bool {
        matches!(self, PruneMethod::Width | PruneMethod::Magnitude)
    }
}

#[derive(Debug, PartialEq)]
pub enum ValidateVerdict {
    Ok,
    UnknownMethod(String),
    MissingCalibration(PruneMethod),
    MissingLayerSpec(PruneMethod),
    MissingTargetRatio(PruneMethod),
}

pub fn validate(
    method: &str,
    has_calibration: bool,
    has_layer_spec: bool,
    target_ratio: f64,
) -> ValidateVerdict {
    let Some(m) = PruneMethod::from_str_strict(method) else {
        return ValidateVerdict::UnknownMethod(method.into());
    };
    if m.requires_calibration() && !has_calibration {
        return ValidateVerdict::MissingCalibration(m);
    }
    if m.requires_layer_spec() && !has_layer_spec {
        return ValidateVerdict::MissingLayerSpec(m);
    }
    if m.requires_target_ratio() && target_ratio <= 0.0 {
        return ValidateVerdict::MissingTargetRatio(m);
    }
    ValidateVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_prune_method_validator")?;

    let cases = [
        ("magnitude default", "magnitude", false, false, 0.5),
        ("wanda no calib", "wanda", false, false, 0.5),
        ("wanda with calib", "wanda", true, false, 0.5),
        ("depth no spec", "depth", false, false, 0.5),
        ("depth with spec", "depth", false, true, 0.5),
        ("width no ratio", "width", false, false, 0.0),
        ("typo", "magntude", false, false, 0.5),
    ];

    for (label, m, calib, layers, r) in cases {
        println!("{label:>22}  →  {:?}", validate(m, calib, layers, r));
    }
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
    fn known_methods_round_trip() {
        for s in [
            "magnitude",
            "structured",
            "depth",
            "width",
            "wanda",
            "sparsegpt",
        ] {
            assert!(PruneMethod::from_str_strict(s).is_some());
        }
    }

    #[test]
    fn unknown_method_rejected() {
        assert!(matches!(
            validate("magntude", false, false, 0.5),
            ValidateVerdict::UnknownMethod(_)
        ));
    }

    #[test]
    fn wanda_requires_calibration() {
        assert!(matches!(
            validate("wanda", false, false, 0.5),
            ValidateVerdict::MissingCalibration(PruneMethod::Wanda)
        ));
    }

    #[test]
    fn sparsegpt_requires_calibration() {
        assert!(matches!(
            validate("sparsegpt", false, false, 0.5),
            ValidateVerdict::MissingCalibration(PruneMethod::SparseGpt)
        ));
    }

    #[test]
    fn depth_requires_layer_spec() {
        assert!(matches!(
            validate("depth", false, false, 0.5),
            ValidateVerdict::MissingLayerSpec(PruneMethod::Depth)
        ));
    }

    #[test]
    fn width_requires_target_ratio() {
        assert!(matches!(
            validate("width", false, false, 0.0),
            ValidateVerdict::MissingTargetRatio(PruneMethod::Width)
        ));
    }

    #[test]
    fn happy_paths_pass() {
        assert_eq!(
            validate("magnitude", false, false, 0.5),
            ValidateVerdict::Ok
        );
        assert_eq!(validate("wanda", true, false, 0.5), ValidateVerdict::Ok);
        assert_eq!(validate("depth", false, true, 0.5), ValidateVerdict::Ok);
        assert_eq!(
            validate("structured", false, false, 0.5),
            ValidateVerdict::Ok
        );
    }
}
