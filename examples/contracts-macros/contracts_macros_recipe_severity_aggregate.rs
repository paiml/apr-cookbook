//! # Contracts-Macros Recipe Severity Aggregate
//!
//! Aggregate recipe severities into Low/Medium/High/Critical buckets
//! and compute weighted total. Returns bucket counts and aggregate
//! score.
//!
//! Demonstrates the **CMM.155** recipe for PMAT-209 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ISO 31000 risk-bucketing; CVSS 3.1 categorical bands.
//!
//! Run with: cargo run --example contracts_macros_recipe_severity_aggregate
//!
//! Added by PMAT-209 (catalog 1504→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SeverityVerdict {
    Ok {
        low: u32,
        medium: u32,
        high: u32,
        critical: u32,
        weighted_total: u32,
    },
    InvalidConfig,
}

pub fn aggregate(severities: &[u8]) -> SeverityVerdict {
    if severities.is_empty() {
        return SeverityVerdict::InvalidConfig;
    }
    for s in severities {
        if !(1..=10).contains(s) {
            return SeverityVerdict::InvalidConfig;
        }
    }
    let mut low = 0u32;
    let mut medium = 0u32;
    let mut high = 0u32;
    let mut critical = 0u32;
    let mut weighted = 0u32;
    for s in severities {
        match *s {
            1..=3 => {
                low += 1;
                weighted += 1;
            }
            4..=6 => {
                medium += 1;
                weighted += 3;
            }
            7..=8 => {
                high += 1;
                weighted += 7;
            }
            _ => {
                critical += 1;
                weighted += 15;
            }
        }
    }
    SeverityVerdict::Ok {
        low,
        medium,
        high,
        critical,
        weighted_total: weighted,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_severity_aggregate")?;

    let s = [1u8, 5, 5, 7, 10];
    println!("aggregate: {:?}", aggregate(&s));
    println!("invalid: {:?}", aggregate(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn low_bucket_counted() {
        let v = aggregate(&[1, 2, 3]);
        if let SeverityVerdict::Ok { low, .. } = v {
            assert_eq!(low, 3);
        }
    }

    #[test]
    fn medium_bucket_counted() {
        let v = aggregate(&[4, 5, 6]);
        if let SeverityVerdict::Ok { medium, .. } = v {
            assert_eq!(medium, 3);
        }
    }

    #[test]
    fn high_bucket_counted() {
        let v = aggregate(&[7, 8]);
        if let SeverityVerdict::Ok { high, .. } = v {
            assert_eq!(high, 2);
        }
    }

    #[test]
    fn critical_bucket_counted() {
        let v = aggregate(&[9, 10]);
        if let SeverityVerdict::Ok { critical, .. } = v {
            assert_eq!(critical, 2);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(aggregate(&[]), SeverityVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_severity() {
        assert_eq!(aggregate(&[0]), SeverityVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_over_ten() {
        assert_eq!(aggregate(&[11]), SeverityVerdict::InvalidConfig);
    }

    #[test]
    fn weighted_total_correct() {
        // 1 (low=1), 5 (medium=3), 7 (high=7), 10 (critical=15) → 26
        let v = aggregate(&[1, 5, 7, 10]);
        if let SeverityVerdict::Ok { weighted_total, .. } = v {
            assert_eq!(weighted_total, 26);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = aggregate(&[1, 5]);
        let r2 = aggregate(&[1, 5]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn boundary_3_low() {
        let v = aggregate(&[3]);
        if let SeverityVerdict::Ok { low, .. } = v {
            assert_eq!(low, 1);
        }
    }

    #[test]
    fn boundary_4_medium() {
        let v = aggregate(&[4]);
        if let SeverityVerdict::Ok { medium, .. } = v {
            assert_eq!(medium, 1);
        }
    }

    #[test]
    fn boundary_8_high() {
        let v = aggregate(&[8]);
        if let SeverityVerdict::Ok { high, .. } = v {
            assert_eq!(high, 1);
        }
    }

    #[test]
    fn many_severities_handled() {
        let s: Vec<u8> = (0..30).map(|_| 5u8).collect();
        let v = aggregate(&s);
        if let SeverityVerdict::Ok { medium, .. } = v {
            assert_eq!(medium, 30);
        }
    }
}
