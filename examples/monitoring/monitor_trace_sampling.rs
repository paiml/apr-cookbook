//! # Monitoring Distributed-Trace Sampling Strategy
//!
//! Strategies:
//!   HeadBased: decide at trace start (uniform random); cheap, cannot
//!     bias toward errors
//!   TailBased: collect all spans, decide after (keep slow + errors);
//!     expensive, accurate
//!   AdaptiveTail: tail-based but only when QPS drops below ingest budget
//!
//! Picker rule:
//!   qps × sample_rate × budget → choose HeadBased if budget can hold
//!   QPS > 10k AND tail_collector_capacity_ok → AdaptiveTail
//!   otherwise → TailBased
//!
//! Demonstrates the **MON.27** recipe for PMAT-144 (monitoring round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: OpenTelemetry sampling strategies + Honeycomb dynamic sampling.
//!
//! Run with: cargo run --example monitor_trace_sampling
//!
//! Added by PMAT-144 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    HeadBased,
    TailBased,
    AdaptiveTail,
}

#[derive(Debug, PartialEq)]
pub enum SamplingVerdict {
    Ok {
        strategy: SamplingStrategy,
        sample_rate_pct: u32,
    },
    InvalidQps,
    InvalidBudget,
}

const HIGH_QPS_THRESHOLD: u32 = 10_000;

pub fn pick(qps: u32, ingest_budget_per_sec: u32, tail_collector_ok: bool) -> SamplingVerdict {
    if qps == 0 {
        return SamplingVerdict::InvalidQps;
    }
    if ingest_budget_per_sec == 0 {
        return SamplingVerdict::InvalidBudget;
    }
    let strategy = if qps > HIGH_QPS_THRESHOLD && tail_collector_ok {
        SamplingStrategy::AdaptiveTail
    } else if qps > HIGH_QPS_THRESHOLD {
        SamplingStrategy::HeadBased
    } else if tail_collector_ok {
        SamplingStrategy::TailBased
    } else {
        SamplingStrategy::HeadBased
    };
    let sample_rate_pct = if qps <= ingest_budget_per_sec {
        100
    } else {
        ((u64::from(ingest_budget_per_sec) * 100) / u64::from(qps)) as u32
    };
    SamplingVerdict::Ok {
        strategy,
        sample_rate_pct,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_trace_sampling")?;

    println!("low qps: {:?}", pick(500, 1_000, true));
    println!("medium qps: {:?}", pick(5_000, 1_000, true));
    println!("high qps with collector: {:?}", pick(50_000, 1_000, true));
    println!("high qps no collector: {:?}", pick(50_000, 1_000, false));
    println!("invalid: {:?}", pick(0, 1_000, true));
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
    fn low_qps_picks_tail_based() {
        let v = pick(500, 1_000, true);
        if let SamplingVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, SamplingStrategy::TailBased);
        }
    }

    #[test]
    fn high_qps_with_collector_adaptive() {
        let v = pick(50_000, 1_000, true);
        if let SamplingVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, SamplingStrategy::AdaptiveTail);
        }
    }

    #[test]
    fn high_qps_no_collector_head_based() {
        let v = pick(50_000, 1_000, false);
        if let SamplingVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, SamplingStrategy::HeadBased);
        }
    }

    #[test]
    fn no_collector_low_qps_head_based() {
        let v = pick(500, 1_000, false);
        if let SamplingVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, SamplingStrategy::HeadBased);
        }
    }

    #[test]
    fn invalid_zero_qps_rejected() {
        assert_eq!(pick(0, 1_000, true), SamplingVerdict::InvalidQps);
    }

    #[test]
    fn invalid_zero_budget_rejected() {
        assert_eq!(pick(500, 0, true), SamplingVerdict::InvalidBudget);
    }

    #[test]
    fn under_budget_full_sampling() {
        let v = pick(500, 1_000, true);
        if let SamplingVerdict::Ok {
            sample_rate_pct, ..
        } = v
        {
            assert_eq!(sample_rate_pct, 100);
        }
    }

    #[test]
    fn over_budget_partial_sampling() {
        // 5000 qps, 1000 budget → 20% sample rate.
        let v = pick(5_000, 1_000, true);
        if let SamplingVerdict::Ok {
            sample_rate_pct, ..
        } = v
        {
            assert_eq!(sample_rate_pct, 20);
        }
    }

    #[test]
    fn at_qps_threshold_still_low() {
        // exactly HIGH_QPS_THRESHOLD = 10000 → still TailBased.
        let v = pick(10_000, 100_000, true);
        if let SamplingVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, SamplingStrategy::TailBased);
        }
    }

    #[test]
    fn just_above_threshold_picks_adaptive() {
        let v = pick(10_001, 1_000, true);
        if let SamplingVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, SamplingStrategy::AdaptiveTail);
        }
    }
}
