//! # apr profile --sample-rate-hz — Sampling Rate Budget
//!
//! `apr profile --sample-rate-hz <N>` controls profiler frequency.
//! Tradeoffs: 99 Hz default (minimal Heisenberg effect on 1 GHz+ CPU),
//! 999 Hz catches sub-millisecond hot paths, > 9999 Hz is sample-aliasing
//! territory. Floor: 10 Hz (statistically meaningless below). This recipe
//! builds the budget validator + overhead estimator.
//!
//! Demonstrates the **PROF.4** recipe for PMAT-115 (apr profile coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PROF-001 + Gregg 2016 (Systems Performance)
//!
//! Run with: cargo run --example cli_profile_sampling_rate_budget
//!
//! Added by PMAT-115 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RateVerdict {
    Ok,
    TooLow { recommended: u32 },
    TooHigh { recommended: u32 },
    InvalidZero,
}

const MIN_RATE_HZ: u32 = 10;
const MAX_RATE_HZ: u32 = 9_999;
const DEFAULT_RATE_HZ: u32 = 99;

pub fn classify(rate_hz: u32) -> RateVerdict {
    if rate_hz == 0 {
        return RateVerdict::InvalidZero;
    }
    if rate_hz < MIN_RATE_HZ {
        return RateVerdict::TooLow {
            recommended: DEFAULT_RATE_HZ,
        };
    }
    if rate_hz > MAX_RATE_HZ {
        return RateVerdict::TooHigh {
            recommended: MAX_RATE_HZ,
        };
    }
    RateVerdict::Ok
}

pub fn estimated_overhead_pct(rate_hz: u32, sample_cost_ns: u32) -> f64 {
    // Overhead = (rate × cost) / 1s_in_ns × 100.
    f64::from(rate_hz) * f64::from(sample_cost_ns) / 1_000_000_000.0 * 100.0
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_profile_sampling_rate_budget")?;

    for r in [0u32, 5, 99, 999, 9_999, 50_000] {
        let v = classify(r);
        let oh = estimated_overhead_pct(r, 5_000); // 5 µs per sample
        println!("rate={r:>5} → {v:?}  overhead≈{oh:.4}%");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn zero_invalid() {
        assert_eq!(classify(0), RateVerdict::InvalidZero);
    }

    #[test]
    fn under_floor_rejected() {
        let v = classify(5);
        assert!(matches!(v, RateVerdict::TooLow { .. }));
    }

    #[test]
    fn at_floor_passes() {
        assert_eq!(classify(MIN_RATE_HZ), RateVerdict::Ok);
    }

    #[test]
    fn default_99hz_passes() {
        assert_eq!(classify(DEFAULT_RATE_HZ), RateVerdict::Ok);
    }

    #[test]
    fn at_ceiling_passes() {
        assert_eq!(classify(MAX_RATE_HZ), RateVerdict::Ok);
    }

    #[test]
    fn above_ceiling_rejected() {
        let v = classify(50_000);
        assert!(matches!(v, RateVerdict::TooHigh { .. }));
    }

    #[test]
    fn overhead_scales_with_rate() {
        let lo = estimated_overhead_pct(99, 5_000);
        let hi = estimated_overhead_pct(999, 5_000);
        assert!(hi > lo);
    }

    #[test]
    fn overhead_scales_with_sample_cost() {
        let cheap = estimated_overhead_pct(99, 1_000);
        let expensive = estimated_overhead_pct(99, 10_000);
        assert!(expensive > cheap);
    }

    #[test]
    fn typical_99hz_5us_overhead_under_0_1pct() {
        // 99 × 5000 ns = 495,000 ns/s = 0.0495%
        let oh = estimated_overhead_pct(99, 5_000);
        assert!(oh < 0.1, "got {oh}%");
    }
}
