//! # Contracts-Macros Severity Decay Curve
//!
//! Apply exponential decay to obligation severity over `days_elapsed`
//! using half-life `t_half_days`. Returns the decayed severity (×100
//! fixed point) and whether it has dropped below the actionable
//! threshold.
//!
//! Demonstrates the **CMM.167** recipe for PMAT-213 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: CVE risk-decay models (NVD); CVSS Temporal-metric
//!  remediation level.
//!
//! Run with: cargo run --example contracts_macros_severity_decay_curve
//!
//! Added by PMAT-213 (catalog 1540→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DecayVerdict {
    Ok {
        decayed_severity_x100: u32,
        below_threshold: bool,
    },
    InvalidConfig,
}

pub fn decay(
    severity: u8,
    days_elapsed: u32,
    t_half_days: u32,
    threshold_x100: u32,
) -> DecayVerdict {
    if !(1..=10).contains(&severity) || t_half_days == 0 {
        return DecayVerdict::InvalidConfig;
    }
    let half_lives = days_elapsed as f64 / t_half_days as f64;
    let factor = 0.5f64.powf(half_lives);
    let result = (severity as f64) * factor;
    let result_x100 = (result * 100.0) as u32;
    DecayVerdict::Ok {
        decayed_severity_x100: result_x100,
        below_threshold: result_x100 < threshold_x100,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_severity_decay_curve")?;

    println!("fresh: {:?}", decay(8, 0, 30, 100));
    println!("1-half-life: {:?}", decay(8, 30, 30, 100));
    println!("3-half-lives: {:?}", decay(8, 90, 30, 100));
    println!("invalid: {:?}", decay(0, 30, 30, 100));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decayer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn zero_elapsed_full_severity() {
        let v = decay(8, 0, 30, 100);
        if let DecayVerdict::Ok {
            decayed_severity_x100,
            ..
        } = v
        {
            assert_eq!(decayed_severity_x100, 800);
        }
    }

    #[test]
    fn one_half_life_halves_severity() {
        let v = decay(8, 30, 30, 100);
        if let DecayVerdict::Ok {
            decayed_severity_x100,
            ..
        } = v
        {
            assert_eq!(decayed_severity_x100, 400);
        }
    }

    #[test]
    fn three_half_lives_one_eighth() {
        let v = decay(8, 90, 30, 100);
        if let DecayVerdict::Ok {
            decayed_severity_x100,
            ..
        } = v
        {
            // 8 / 8 = 1.0 → 100 (×100)
            assert_eq!(decayed_severity_x100, 100);
        }
    }

    #[test]
    fn invalid_zero_severity() {
        assert_eq!(decay(0, 30, 30, 100), DecayVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_severity_over_ten() {
        assert_eq!(decay(11, 30, 30, 100), DecayVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_half_life() {
        assert_eq!(decay(8, 30, 0, 100), DecayVerdict::InvalidConfig);
    }

    #[test]
    fn below_threshold_when_decayed_far() {
        let v = decay(8, 300, 30, 100);
        if let DecayVerdict::Ok {
            below_threshold, ..
        } = v
        {
            assert!(below_threshold);
        }
    }

    #[test]
    fn above_threshold_when_fresh() {
        let v = decay(8, 0, 30, 100);
        if let DecayVerdict::Ok {
            below_threshold, ..
        } = v
        {
            assert!(!below_threshold);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = decay(5, 30, 30, 100);
        let r2 = decay(5, 30, 30, 100);
        assert_eq!(r1, r2);
    }

    #[test]
    fn monotone_decreasing_over_time() {
        let v_30 = decay(8, 30, 30, 100);
        let v_60 = decay(8, 60, 30, 100);
        if let (
            DecayVerdict::Ok {
                decayed_severity_x100: a,
                ..
            },
            DecayVerdict::Ok {
                decayed_severity_x100: b,
                ..
            },
        ) = (v_30, v_60)
        {
            assert!(b < a);
        }
    }

    #[test]
    fn longer_half_life_slower_decay() {
        let fast = decay(8, 30, 30, 100);
        let slow = decay(8, 30, 90, 100);
        if let (
            DecayVerdict::Ok {
                decayed_severity_x100: f,
                ..
            },
            DecayVerdict::Ok {
                decayed_severity_x100: s,
                ..
            },
        ) = (fast, slow)
        {
            assert!(s > f);
        }
    }

    #[test]
    fn boundary_severity_one() {
        let v = decay(1, 0, 30, 100);
        if let DecayVerdict::Ok {
            decayed_severity_x100,
            ..
        } = v
        {
            assert_eq!(decayed_severity_x100, 100);
        }
    }
}
