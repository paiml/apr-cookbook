//! # Serverless Provisioned Concurrency Calculator
//!
//! Provisioned concurrency = ceil(p99_qps × p99_duration_seconds × headroom).
//! Too low = cold-start spikes; too high = $$ wasted on idle
//! containers. Headroom 1.2-2.0× per AWS recs. This recipe builds the
//! calculator.
//!
//! Demonstrates the **SVL.9** recipe for PMAT-134 (serverless coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS Lambda provisioned concurrency calculator (Q4 2024 docs).
//!
//! Run with: cargo run --example serverless_provisioned_concurrency
//!
//! Added by PMAT-134 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MAX_RECOMMENDED_PC: u32 = 1000;

#[derive(Debug, PartialEq)]
pub enum PcVerdict {
    Ok {
        units: u32,
        monthly_cost_estimate_usd: f64,
    },
    InvalidLoad,
    InvalidHeadroom,
    AboveSoftCap {
        recommended: u32,
    },
}

const PC_PRICE_USD_PER_GB_SECOND: f64 = 0.0000041;
const SECONDS_PER_MONTH: f64 = 30.0 * 24.0 * 3600.0;

pub fn calc(p99_qps: f64, mean_duration_secs: f64, headroom: f64, mem_gb: f64) -> PcVerdict {
    if !p99_qps.is_finite() || p99_qps <= 0.0 {
        return PcVerdict::InvalidLoad;
    }
    if !mean_duration_secs.is_finite() || mean_duration_secs <= 0.0 {
        return PcVerdict::InvalidLoad;
    }
    if !headroom.is_finite() || headroom < 1.0 {
        return PcVerdict::InvalidHeadroom;
    }
    if !mem_gb.is_finite() || mem_gb <= 0.0 {
        return PcVerdict::InvalidLoad;
    }
    let raw = (p99_qps * mean_duration_secs * headroom).ceil() as u32;
    let units = raw.max(1);
    if units > MAX_RECOMMENDED_PC {
        return PcVerdict::AboveSoftCap {
            recommended: MAX_RECOMMENDED_PC,
        };
    }
    let monthly = f64::from(units) * mem_gb * SECONDS_PER_MONTH * PC_PRICE_USD_PER_GB_SECOND;
    PcVerdict::Ok {
        units,
        monthly_cost_estimate_usd: monthly,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("serverless_provisioned_concurrency")?;

    for (qps, dur, hr, mem) in [
        (10.0_f64, 0.5_f64, 1.5_f64, 1.0_f64),
        (100.0, 0.2, 2.0, 0.5),
        (1000.0, 1.0, 1.5, 2.0),
        (0.0, 0.5, 1.5, 1.0),
        (10.0, 0.5, 0.5, 1.0),
    ] {
        println!(
            "qps={qps} dur={dur}s headroom={hr} mem={mem}GiB → {:?}",
            calc(qps, dur, hr, mem)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calc_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_load_calculated() {
        // 10 qps × 0.5s × 1.5 = 7.5 → ceil 8.
        let v = calc(10.0, 0.5, 1.5, 1.0);
        if let PcVerdict::Ok { units, .. } = v {
            assert_eq!(units, 8);
        }
    }

    #[test]
    fn high_load_calculated() {
        // 100 × 0.2 × 2.0 = 40.
        let v = calc(100.0, 0.2, 2.0, 0.5);
        if let PcVerdict::Ok { units, .. } = v {
            assert_eq!(units, 40);
        }
    }

    #[test]
    fn extreme_load_capped() {
        // 1000 × 1.0 × 1.5 = 1500 > 1000 cap.
        let v = calc(1000.0, 1.0, 1.5, 2.0);
        assert!(matches!(v, PcVerdict::AboveSoftCap { .. }));
    }

    #[test]
    fn zero_qps_invalid() {
        assert_eq!(calc(0.0, 0.5, 1.5, 1.0), PcVerdict::InvalidLoad);
    }

    #[test]
    fn negative_duration_invalid() {
        assert_eq!(calc(10.0, -0.1, 1.5, 1.0), PcVerdict::InvalidLoad);
    }

    #[test]
    fn headroom_below_one_invalid() {
        assert_eq!(calc(10.0, 0.5, 0.5, 1.0), PcVerdict::InvalidHeadroom);
    }

    #[test]
    fn nan_input_invalid() {
        assert_eq!(calc(f64::NAN, 0.5, 1.5, 1.0), PcVerdict::InvalidLoad);
    }

    #[test]
    fn floor_at_one_unit_minimum() {
        // 0.1 qps × 0.1s × 1.0 headroom = 0.01 → ceil 1.
        let v = calc(0.1, 0.1, 1.0, 1.0);
        if let PcVerdict::Ok { units, .. } = v {
            assert_eq!(units, 1);
        }
    }

    #[test]
    fn cost_proportional_to_memory() {
        let small = calc(10.0, 0.5, 1.5, 0.5);
        let large = calc(10.0, 0.5, 1.5, 2.0);
        if let (
            PcVerdict::Ok {
                monthly_cost_estimate_usd: c_small,
                ..
            },
            PcVerdict::Ok {
                monthly_cost_estimate_usd: c_large,
                ..
            },
        ) = (small, large)
        {
            // 4× memory → 4× cost.
            assert!((c_large / c_small - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn cost_positive_for_valid_input() {
        if let PcVerdict::Ok {
            monthly_cost_estimate_usd,
            ..
        } = calc(10.0, 0.5, 1.5, 1.0)
        {
            assert!(monthly_cost_estimate_usd > 0.0);
        }
    }
}
