//! # Serverless Timeout Budget Picker
//!
//! Lambda timeouts: 1s/3s/15s/60s/15min. Pick the smallest tier where
//! P99_inference + cold_start_ms + 200ms safety < tier. Too small =
//! truncated requests; too large = wasted billing seconds. This recipe
//! builds the picker.
//!
//! Demonstrates the **SVL.8** recipe for PMAT-134 (serverless coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS Lambda execution-time pricing model.
//!
//! Run with: cargo run --example serverless_timeout_budget_picker
//!
//! Added by PMAT-134 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const TIERS_MS: [u32; 5] = [1_000, 3_000, 15_000, 60_000, 900_000];
const SAFETY_MARGIN_MS: u32 = 200;

#[derive(Debug, PartialEq)]
pub enum TimeoutVerdict {
    Ok {
        tier_ms: u32,
        budget_ms: u32,
        slack_ms: u32,
    },
    InvalidLatency,
    NeedsHigherTier {
        required_ms: u32,
    },
}

pub fn pick(p99_inference_ms: u32, cold_start_ms: u32) -> TimeoutVerdict {
    let total = p99_inference_ms
        .checked_add(cold_start_ms)
        .and_then(|x| x.checked_add(SAFETY_MARGIN_MS));
    let Some(budget_ms) = total else {
        return TimeoutVerdict::InvalidLatency;
    };
    if budget_ms == SAFETY_MARGIN_MS {
        return TimeoutVerdict::InvalidLatency;
    }
    for &tier in &TIERS_MS {
        if budget_ms <= tier {
            return TimeoutVerdict::Ok {
                tier_ms: tier,
                budget_ms,
                slack_ms: tier - budget_ms,
            };
        }
    }
    TimeoutVerdict::NeedsHigherTier {
        required_ms: budget_ms,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("serverless_timeout_budget_picker")?;

    for (p99, cold) in [
        (50u32, 200u32),
        (500, 1500),
        (5000, 2000),
        (30000, 5000),
        (1_000_000, 0),
        (0, 0),
    ] {
        println!("p99={p99}ms cold={cold}ms → {:?}", pick(p99, cold));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fast_inference_picks_one_second_tier() {
        // 50 + 200 + 200 = 450 ≤ 1000.
        let v = pick(50, 200);
        assert!(matches!(v, TimeoutVerdict::Ok { tier_ms: 1_000, .. }));
    }

    #[test]
    fn medium_picks_three_second_tier() {
        // 500 + 1500 + 200 = 2200 ≤ 3000.
        let v = pick(500, 1500);
        assert!(matches!(v, TimeoutVerdict::Ok { tier_ms: 3_000, .. }));
    }

    #[test]
    fn slow_picks_fifteen_second_tier() {
        // 5000 + 2000 + 200 = 7200 ≤ 15000.
        let v = pick(5000, 2000);
        assert!(matches!(
            v,
            TimeoutVerdict::Ok {
                tier_ms: 15_000,
                ..
            }
        ));
    }

    #[test]
    fn very_slow_picks_sixty_second_tier() {
        // 30000 + 5000 + 200 = 35200 ≤ 60000.
        let v = pick(30000, 5000);
        assert!(matches!(
            v,
            TimeoutVerdict::Ok {
                tier_ms: 60_000,
                ..
            }
        ));
    }

    #[test]
    fn excessive_picks_max_tier_or_overflow() {
        // 200000 + 0 + 200 = 200200 ≤ 900000 (15 min tier).
        let v = pick(200_000, 0);
        assert!(matches!(
            v,
            TimeoutVerdict::Ok {
                tier_ms: 900_000,
                ..
            }
        ));
    }

    #[test]
    fn beyond_lambda_max_needs_higher_tier() {
        let v = pick(1_000_000, 0);
        assert!(matches!(v, TimeoutVerdict::NeedsHigherTier { .. }));
    }

    #[test]
    fn zero_budget_invalid() {
        // 0 + 0 + 200 = 200 ms is just the safety margin → InvalidLatency.
        assert_eq!(pick(0, 0), TimeoutVerdict::InvalidLatency);
    }

    #[test]
    fn slack_correctly_computed() {
        // 50 + 200 + 200 = 450; tier = 1000; slack = 550.
        if let TimeoutVerdict::Ok { slack_ms, .. } = pick(50, 200) {
            assert_eq!(slack_ms, 550);
        }
    }

    #[test]
    fn safety_margin_applied() {
        // budget always includes 200 ms safety.
        if let TimeoutVerdict::Ok { budget_ms, .. } = pick(100, 100) {
            assert_eq!(budget_ms, 100 + 100 + 200);
        }
    }

    #[test]
    fn large_input_overflow_safe() {
        // u32::MAX would overflow on add → InvalidLatency.
        assert_eq!(pick(u32::MAX, 1000), TimeoutVerdict::InvalidLatency);
    }
}
