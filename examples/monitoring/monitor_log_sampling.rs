//! # Monitoring Log Sampling Strategy Picker
//!
//! At high QPS, log every request → too expensive. Strategies:
//!   Head: log first N
//!   Tail: log only last N
//!   Uniform: log every K-th
//!   Stratified: log all errors + sample of successes
//!   AdaptiveOnError: full rate while error_rate > threshold, then sample
//!
//! Picker chooses based on (qps, error_rate, retention_secs, budget_per_sec).
//!
//! Demonstrates the **MON.22** recipe for PMAT-141 (monitoring round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Honeycomb / Datadog dynamic-sampling literature.
//!
//! Run with: cargo run --example monitor_log_sampling
//!
//! Added by PMAT-141 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Head,
    Tail,
    Uniform,
    Stratified,
    AdaptiveOnError,
}

#[derive(Debug, PartialEq)]
pub enum SamplingVerdict {
    Ok {
        strategy: Strategy,
        sample_rate: f64,
    },
    InvalidQps,
    InvalidBudget,
    InvalidErrorRate,
}

const HIGH_ERROR_THRESHOLD: f64 = 0.05;
const HIGH_QPS_THRESHOLD: f64 = 1_000.0;

pub fn pick(qps: f64, error_rate: f64, budget_per_sec: f64) -> SamplingVerdict {
    if !qps.is_finite() || qps <= 0.0 {
        return SamplingVerdict::InvalidQps;
    }
    if !budget_per_sec.is_finite() || budget_per_sec <= 0.0 {
        return SamplingVerdict::InvalidBudget;
    }
    if !error_rate.is_finite() || !(0.0..=1.0).contains(&error_rate) {
        return SamplingVerdict::InvalidErrorRate;
    }
    let strategy = if error_rate >= HIGH_ERROR_THRESHOLD {
        Strategy::AdaptiveOnError
    } else if qps <= budget_per_sec {
        Strategy::Head
    } else if error_rate > 0.0 {
        Strategy::Stratified
    } else if qps >= HIGH_QPS_THRESHOLD {
        Strategy::Uniform
    } else {
        Strategy::Tail
    };
    let sample_rate = (budget_per_sec / qps).clamp(0.0, 1.0);
    SamplingVerdict::Ok {
        strategy,
        sample_rate,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_log_sampling")?;

    println!("low qps under budget: {:?}", pick(50.0, 0.001, 100.0));
    println!("high qps + low error: {:?}", pick(2_000.0, 0.001, 100.0));
    println!("high qps + high error: {:?}", pick(2_000.0, 0.10, 100.0));
    println!("medium qps + some errors: {:?}", pick(500.0, 0.01, 100.0));
    println!("invalid qps: {:?}", pick(0.0, 0.01, 100.0));
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
    fn high_error_picks_adaptive() {
        let v = pick(500.0, 0.10, 100.0);
        if let SamplingVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, Strategy::AdaptiveOnError);
        }
    }

    #[test]
    fn under_budget_picks_head() {
        let v = pick(50.0, 0.001, 100.0);
        if let SamplingVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, Strategy::Head);
        }
    }

    #[test]
    fn over_budget_with_errors_picks_stratified() {
        let v = pick(500.0, 0.01, 100.0);
        if let SamplingVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, Strategy::Stratified);
        }
    }

    #[test]
    fn high_qps_no_errors_picks_uniform() {
        let v = pick(2_000.0, 0.0, 100.0);
        if let SamplingVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, Strategy::Uniform);
        }
    }

    #[test]
    fn medium_qps_no_errors_picks_tail() {
        // qps > budget, error_rate == 0.0, qps < HIGH_QPS_THRESHOLD.
        let v = pick(500.0, 0.0, 100.0);
        if let SamplingVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, Strategy::Tail);
        }
    }

    #[test]
    fn invalid_qps_zero_rejected() {
        assert_eq!(pick(0.0, 0.01, 100.0), SamplingVerdict::InvalidQps);
    }

    #[test]
    fn invalid_qps_negative_rejected() {
        assert_eq!(pick(-1.0, 0.01, 100.0), SamplingVerdict::InvalidQps);
    }

    #[test]
    fn invalid_budget_rejected() {
        assert_eq!(pick(100.0, 0.01, 0.0), SamplingVerdict::InvalidBudget);
    }

    #[test]
    fn invalid_error_rate_rejected() {
        assert_eq!(pick(100.0, 1.5, 100.0), SamplingVerdict::InvalidErrorRate);
        assert_eq!(pick(100.0, -0.1, 100.0), SamplingVerdict::InvalidErrorRate);
    }

    #[test]
    fn sample_rate_clamped_to_one() {
        // qps under budget → ratio > 1, clamped to 1.
        let v = pick(50.0, 0.001, 100.0);
        if let SamplingVerdict::Ok { sample_rate, .. } = v {
            assert!((sample_rate - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn sample_rate_below_one_when_over_budget() {
        let v = pick(500.0, 0.01, 100.0);
        if let SamplingVerdict::Ok { sample_rate, .. } = v {
            assert!(sample_rate < 1.0);
        }
    }
}
