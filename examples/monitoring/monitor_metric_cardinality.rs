//! # Monitoring Metric Cardinality Budget
//!
//! Prometheus metrics with high cardinality (e.g., labels per
//! request_id, user_id) blow up storage. Best practice: cardinality
//! per metric < 10k unique series.
//!
//! Picker:
//!   estimated_series = product(distinct_values_per_label)
//!   < 1k → Healthy
//!   1k-10k → Warning
//!   10k-100k → AtRisk
//!   > 100k → Reject (requires bucketing)
//!
//! Demonstrates the **MON.26** recipe for PMAT-144 (monitoring round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Prometheus best practices on label cardinality.
//!
//! Run with: cargo run --example monitor_metric_cardinality
//!
//! Added by PMAT-144 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardinalityTier {
    Healthy,
    Warning,
    AtRisk,
    Reject,
}

#[derive(Debug, PartialEq)]
pub enum CardinalityVerdict {
    Ok {
        estimated_series: u64,
        tier: CardinalityTier,
    },
    EmptyLabels,
    OverflowOnEstimate,
}

pub fn check(distinct_values_per_label: &[u32]) -> CardinalityVerdict {
    if distinct_values_per_label.is_empty() {
        return CardinalityVerdict::EmptyLabels;
    }
    let mut series: u64 = 1;
    for &n in distinct_values_per_label {
        if n == 0 {
            return CardinalityVerdict::EmptyLabels;
        }
        series = match series.checked_mul(u64::from(n)) {
            Some(s) => s,
            None => return CardinalityVerdict::OverflowOnEstimate,
        };
    }
    let tier = if series < 1_000 {
        CardinalityTier::Healthy
    } else if series < 10_000 {
        CardinalityTier::Warning
    } else if series < 100_000 {
        CardinalityTier::AtRisk
    } else {
        CardinalityTier::Reject
    };
    CardinalityVerdict::Ok {
        estimated_series: series,
        tier,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_metric_cardinality")?;

    println!("3 labels small: {:?}", check(&[5, 4, 3]));
    println!("path × method: {:?}", check(&[100, 5]));
    println!("status × path × method: {:?}", check(&[10, 100, 5]));
    println!("user_id (huge): {:?}", check(&[100_000, 5]));
    println!("empty: {:?}", check(&[]));
    println!("zero label: {:?}", check(&[5, 0, 3]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_cardinality_healthy() {
        let v = check(&[5, 4, 3]);
        if let CardinalityVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CardinalityTier::Healthy);
        }
    }

    #[test]
    fn medium_cardinality_warning() {
        // 100 × 50 = 5000.
        let v = check(&[100, 50]);
        if let CardinalityVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CardinalityTier::Warning);
        }
    }

    #[test]
    fn high_cardinality_at_risk() {
        // 50 × 50 × 20 = 50000.
        let v = check(&[50, 50, 20]);
        if let CardinalityVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CardinalityTier::AtRisk);
        }
    }

    #[test]
    fn excessive_cardinality_reject() {
        let v = check(&[100_000, 5]);
        if let CardinalityVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CardinalityTier::Reject);
        }
    }

    #[test]
    fn empty_labels_rejected() {
        assert_eq!(check(&[]), CardinalityVerdict::EmptyLabels);
    }

    #[test]
    fn zero_label_rejected() {
        assert_eq!(check(&[5, 0, 3]), CardinalityVerdict::EmptyLabels);
    }

    #[test]
    fn overflow_detected() {
        // Three u32::MAX values overflow u64 multiply.
        let v = check(&[u32::MAX, u32::MAX, u32::MAX]);
        assert_eq!(v, CardinalityVerdict::OverflowOnEstimate);
    }

    #[test]
    fn estimated_series_correct() {
        let v = check(&[5, 4, 3]);
        if let CardinalityVerdict::Ok {
            estimated_series, ..
        } = v
        {
            assert_eq!(estimated_series, 60);
        }
    }

    #[test]
    fn at_1000_warning_starts() {
        let v = check(&[1_000]);
        if let CardinalityVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CardinalityTier::Warning);
        }
    }

    #[test]
    fn at_10000_at_risk_starts() {
        let v = check(&[10_000]);
        if let CardinalityVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CardinalityTier::AtRisk);
        }
    }

    #[test]
    fn at_100000_rejects() {
        let v = check(&[100_000]);
        if let CardinalityVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CardinalityTier::Reject);
        }
    }
}
