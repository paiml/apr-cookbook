//! # Contracts-Macros Recipe Benchmark Envelope
//!
//! Validate that benchmark recipes declare required fields (warmup,
//! samples, tolerance, target) and reject envelopes outside config
//! ranges (e.g. samples too low for stable measurement).
//!
//! Demonstrates the **CMM.81** recipe for PMAT-184 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Criterion.rs benchmark configuration; Mahout micro-bench
//!  envelope conventions.
//!
//! Run with: cargo run --example contracts_macros_recipe_benchmark_envelope
//!
//! Added by PMAT-184 (catalog 1279→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone)]
pub enum EnvelopeIssue {
    MissingWarmup,
    MissingSamples,
    MissingTolerance,
    SamplesTooLow,
    ToleranceTooLoose,
}

#[derive(Debug, PartialEq)]
pub enum EnvelopeVerdict {
    Ok { issues: Vec<EnvelopeIssue> },
    InvalidConfig,
}

pub fn validate(
    has_warmup: bool,
    samples: u32,
    tolerance_pct: f64,
    min_samples: u32,
    max_tolerance_pct: f64,
) -> EnvelopeVerdict {
    if min_samples == 0 || !(0.0..100.0).contains(&max_tolerance_pct) {
        return EnvelopeVerdict::InvalidConfig;
    }
    let mut issues: Vec<EnvelopeIssue> = Vec::new();
    if !has_warmup {
        issues.push(EnvelopeIssue::MissingWarmup);
    }
    if samples == 0 {
        issues.push(EnvelopeIssue::MissingSamples);
    } else if samples < min_samples {
        issues.push(EnvelopeIssue::SamplesTooLow);
    }
    if tolerance_pct <= 0.0 {
        issues.push(EnvelopeIssue::MissingTolerance);
    } else if tolerance_pct > max_tolerance_pct {
        issues.push(EnvelopeIssue::ToleranceTooLoose);
    }
    EnvelopeVerdict::Ok { issues }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_benchmark_envelope")?;

    println!("ok: {:?}", validate(true, 100, 5.0, 30, 20.0));
    println!("issues: {:?}", validate(false, 5, 50.0, 30, 20.0));
    println!("invalid: {:?}", validate(true, 100, 5.0, 0, 20.0));
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
    fn complete_envelope_no_issues() {
        let v = validate(true, 100, 5.0, 30, 20.0);
        if let EnvelopeVerdict::Ok { issues } = v {
            assert!(issues.is_empty());
        }
    }

    #[test]
    fn missing_warmup_flagged() {
        let v = validate(false, 100, 5.0, 30, 20.0);
        if let EnvelopeVerdict::Ok { issues } = v {
            assert!(issues.contains(&EnvelopeIssue::MissingWarmup));
        }
    }

    #[test]
    fn zero_samples_flagged() {
        let v = validate(true, 0, 5.0, 30, 20.0);
        if let EnvelopeVerdict::Ok { issues } = v {
            assert!(issues.contains(&EnvelopeIssue::MissingSamples));
        }
    }

    #[test]
    fn samples_too_low_flagged() {
        let v = validate(true, 5, 5.0, 30, 20.0);
        if let EnvelopeVerdict::Ok { issues } = v {
            assert!(issues.contains(&EnvelopeIssue::SamplesTooLow));
        }
    }

    #[test]
    fn zero_tolerance_flagged() {
        let v = validate(true, 100, 0.0, 30, 20.0);
        if let EnvelopeVerdict::Ok { issues } = v {
            assert!(issues.contains(&EnvelopeIssue::MissingTolerance));
        }
    }

    #[test]
    fn tolerance_too_loose_flagged() {
        let v = validate(true, 100, 50.0, 30, 20.0);
        if let EnvelopeVerdict::Ok { issues } = v {
            assert!(issues.contains(&EnvelopeIssue::ToleranceTooLoose));
        }
    }

    #[test]
    fn invalid_zero_min_samples() {
        assert_eq!(
            validate(true, 100, 5.0, 0, 20.0),
            EnvelopeVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_max_tolerance_out_of_range() {
        assert_eq!(
            validate(true, 100, 5.0, 30, 200.0),
            EnvelopeVerdict::InvalidConfig
        );
    }

    #[test]
    fn multiple_issues_collected() {
        let v = validate(false, 5, 50.0, 30, 20.0);
        if let EnvelopeVerdict::Ok { issues } = v {
            assert!(issues.len() >= 3);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = validate(true, 100, 5.0, 30, 20.0);
        let r2 = validate(true, 100, 5.0, 30, 20.0);
        assert_eq!(r1, r2);
    }

    #[test]
    fn boundary_min_samples_passes() {
        let v = validate(true, 30, 5.0, 30, 20.0);
        if let EnvelopeVerdict::Ok { issues } = v {
            assert!(!issues.contains(&EnvelopeIssue::SamplesTooLow));
        }
    }
}
