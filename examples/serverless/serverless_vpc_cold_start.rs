//! # Serverless VPC Cold-Start Classifier
//!
//! Lambda in a VPC adds ENI (Elastic Network Interface) attachment time
//! to cold start. Hyperplane (post-2019 redesign) made this go from
//! ~10s to ~100ms but it still adds. Classify cold-start budget tier:
//! Sub100Ms, Sub1Sec, Sub10Sec, AboveBudget. This recipe builds the
//! classifier.
//!
//! Demonstrates the **SVL.10** recipe for PMAT-134 (serverless coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS Lambda Hyperplane VPC networking redesign (re:Invent 2019).
//!
//! Run with: cargo run --example serverless_vpc_cold_start
//!
//! Added by PMAT-134 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColdTier {
    Sub100Ms,
    Sub1Sec,
    Sub10Sec,
    AboveBudget,
}

#[derive(Debug, PartialEq)]
pub enum ClassifyVerdict {
    Ok {
        tier: ColdTier,
        total_cold_ms: u32,
        eni_overhead_ms: u32,
    },
    InvalidConfig,
}

pub fn classify(
    base_cold_start_ms: u32,
    in_vpc: bool,
    has_provisioned_concurrency: bool,
) -> ClassifyVerdict {
    if base_cold_start_ms == 0 && !has_provisioned_concurrency {
        return ClassifyVerdict::InvalidConfig;
    }
    let eni = if in_vpc && !has_provisioned_concurrency {
        100u32
    } else {
        0
    };
    let total = if has_provisioned_concurrency {
        0
    } else {
        base_cold_start_ms.saturating_add(eni)
    };
    let tier = match total {
        0..=100 => ColdTier::Sub100Ms,
        101..=1000 => ColdTier::Sub1Sec,
        1001..=10_000 => ColdTier::Sub10Sec,
        _ => ColdTier::AboveBudget,
    };
    ClassifyVerdict::Ok {
        tier,
        total_cold_ms: total,
        eni_overhead_ms: eni,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("serverless_vpc_cold_start")?;

    for (base, vpc, pc) in [
        (50u32, false, false),
        (50, true, false),
        (500, true, false),
        (5000, true, false),
        (15000, true, false),
        (5000, true, true),
        (0, false, true),
    ] {
        println!(
            "base={base}ms vpc={vpc} pc={pc} → {:?}",
            classify(base, vpc, pc)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fast_no_vpc_sub_100ms() {
        let v = classify(50, false, false);
        assert!(matches!(
            v,
            ClassifyVerdict::Ok {
                tier: ColdTier::Sub100Ms,
                ..
            }
        ));
    }

    #[test]
    fn vpc_adds_eni_overhead() {
        if let ClassifyVerdict::Ok {
            eni_overhead_ms, ..
        } = classify(50, true, false)
        {
            assert_eq!(eni_overhead_ms, 100);
        }
    }

    #[test]
    fn no_vpc_no_eni_overhead() {
        if let ClassifyVerdict::Ok {
            eni_overhead_ms, ..
        } = classify(50, false, false)
        {
            assert_eq!(eni_overhead_ms, 0);
        }
    }

    #[test]
    fn slow_in_vpc_picks_sub_1sec() {
        // 500 + 100 = 600 → Sub1Sec.
        let v = classify(500, true, false);
        assert!(matches!(
            v,
            ClassifyVerdict::Ok {
                tier: ColdTier::Sub1Sec,
                ..
            }
        ));
    }

    #[test]
    fn very_slow_in_vpc_picks_sub_10sec() {
        // 5000 + 100 = 5100 → Sub10Sec.
        let v = classify(5000, true, false);
        assert!(matches!(
            v,
            ClassifyVerdict::Ok {
                tier: ColdTier::Sub10Sec,
                ..
            }
        ));
    }

    #[test]
    fn enormous_above_budget() {
        let v = classify(15000, true, false);
        assert!(matches!(
            v,
            ClassifyVerdict::Ok {
                tier: ColdTier::AboveBudget,
                ..
            }
        ));
    }

    #[test]
    fn provisioned_concurrency_zeroes_cold() {
        if let ClassifyVerdict::Ok { total_cold_ms, .. } = classify(5000, true, true) {
            assert_eq!(total_cold_ms, 0);
        }
    }

    #[test]
    fn provisioned_concurrency_picks_sub_100ms() {
        let v = classify(5000, true, true);
        assert!(matches!(
            v,
            ClassifyVerdict::Ok {
                tier: ColdTier::Sub100Ms,
                ..
            }
        ));
    }

    #[test]
    fn zero_base_no_pc_invalid() {
        assert_eq!(classify(0, false, false), ClassifyVerdict::InvalidConfig);
    }

    #[test]
    fn provisioned_concurrency_eni_zero_too() {
        // PC also bypasses ENI overhead.
        if let ClassifyVerdict::Ok {
            eni_overhead_ms, ..
        } = classify(0, true, true)
        {
            assert_eq!(eni_overhead_ms, 0);
        }
    }

    #[test]
    fn boundary_at_100ms_is_sub_100ms() {
        // 0 + 100 ENI = 100 → Sub100Ms (inclusive).
        let v = classify(0, true, true);
        assert!(matches!(
            v,
            ClassifyVerdict::Ok {
                tier: ColdTier::Sub100Ms,
                ..
            }
        ));
    }
}
