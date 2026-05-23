//! # Contracts-Macros Recipe Risk Score
//!
//! Compute a 0-100 risk score = (severity × likelihood × age_factor)
//! capped at 100. Returns the score and a categorical band:
//! Low/Medium/High/Critical.
//!
//! Demonstrates the **CMM.143** recipe for PMAT-205 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NIST SP 800-30 risk-assessment framework; CVSS 3.1 base
//!  metric composition.
//!
//! Run with: cargo run --example contracts_macros_recipe_risk_score
//!
//! Added by PMAT-205 (catalog 1468→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RiskBand {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, PartialEq)]
pub enum RiskVerdict {
    Ok { score: u32, band: RiskBand },
    InvalidConfig,
}

pub fn compute(severity: u32, likelihood: u32, age_days: u32) -> RiskVerdict {
    if !(1..=10).contains(&severity) || !(1..=10).contains(&likelihood) {
        return RiskVerdict::InvalidConfig;
    }
    let age_factor = 1 + (age_days / 30).min(2); // caps at 3
    let raw = severity * likelihood * age_factor;
    let score = raw.min(100);
    let band = match score {
        0..=24 => RiskBand::Low,
        25..=49 => RiskBand::Medium,
        50..=74 => RiskBand::High,
        _ => RiskBand::Critical,
    };
    RiskVerdict::Ok { score, band }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_risk_score")?;

    println!("low: {:?}", compute(3, 2, 5));
    println!("critical: {:?}", compute(10, 10, 90));
    println!("invalid: {:?}", compute(0, 5, 5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scorer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn low_severity_low_band() {
        let v = compute(2, 2, 0);
        if let RiskVerdict::Ok { band, .. } = v {
            assert_eq!(band, RiskBand::Low);
        }
    }

    #[test]
    fn high_severity_high_band() {
        let v = compute(10, 10, 0);
        if let RiskVerdict::Ok { band, .. } = v {
            assert_eq!(band, RiskBand::Critical);
        }
    }

    #[test]
    fn invalid_severity_zero() {
        assert_eq!(compute(0, 5, 0), RiskVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_severity_over_ten() {
        assert_eq!(compute(11, 5, 0), RiskVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_likelihood_zero() {
        assert_eq!(compute(5, 0, 0), RiskVerdict::InvalidConfig);
    }

    #[test]
    fn score_capped_at_100() {
        let v = compute(10, 10, 365);
        if let RiskVerdict::Ok { score, .. } = v {
            assert_eq!(score, 100);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = compute(5, 5, 30);
        let r2 = compute(5, 5, 30);
        assert_eq!(r1, r2);
    }

    #[test]
    fn age_factor_increases_score() {
        let young = compute(5, 5, 0);
        let old = compute(5, 5, 60);
        if let (RiskVerdict::Ok { score: y, .. }, RiskVerdict::Ok { score: o, .. }) = (young, old) {
            assert!(o > y);
        }
    }

    #[test]
    fn band_low_threshold() {
        let v = compute(1, 1, 0);
        if let RiskVerdict::Ok { score, band } = v {
            assert_eq!(score, 1);
            assert_eq!(band, RiskBand::Low);
        }
    }

    #[test]
    fn band_medium_threshold() {
        // severity=5, likelihood=5, age=0 → 25 → Medium
        let v = compute(5, 5, 0);
        if let RiskVerdict::Ok { band, .. } = v {
            assert_eq!(band, RiskBand::Medium);
        }
    }

    #[test]
    fn band_high_threshold() {
        // severity=6, likelihood=10, age=0 → 60 → High
        let v = compute(6, 10, 0);
        if let RiskVerdict::Ok { band, .. } = v {
            assert_eq!(band, RiskBand::High);
        }
    }

    #[test]
    fn age_factor_caps_after_60_days() {
        let at_60 = compute(2, 2, 60);
        let at_365 = compute(2, 2, 365);
        // Age factor caps at 3 → score equal.
        assert_eq!(at_60, at_365);
    }
}
