//! # Monitoring Token Throughput
//!
//! Track tokens/sec inference rate vs target SLO. Verdict:
//!   ≥target × 0.95: Healthy
//!   target × 0.80–0.95: Degraded
//!   <target × 0.80: Severely degraded
//!
//! Demonstrates the **MON.45** recipe for PMAT-158 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vLLM and TGI throughput SLO conventions.
//!
//! Run with: cargo run --example monitor_token_throughput
//!
//! Added by PMAT-158 (catalog 1045→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ThroughputVerdict {
    Healthy { observed: f64, target: f64 },
    Degraded { observed: f64, target: f64 },
    Severe { observed: f64, target: f64 },
    InvalidConfig,
}

pub fn check(observed_tokens_per_sec: f64, target_tokens_per_sec: f64) -> ThroughputVerdict {
    if !observed_tokens_per_sec.is_finite()
        || !target_tokens_per_sec.is_finite()
        || observed_tokens_per_sec < 0.0
        || target_tokens_per_sec <= 0.0
    {
        return ThroughputVerdict::InvalidConfig;
    }
    let ratio = observed_tokens_per_sec / target_tokens_per_sec;
    if ratio >= 0.95 {
        ThroughputVerdict::Healthy {
            observed: observed_tokens_per_sec,
            target: target_tokens_per_sec,
        }
    } else if ratio >= 0.80 {
        ThroughputVerdict::Degraded {
            observed: observed_tokens_per_sec,
            target: target_tokens_per_sec,
        }
    } else {
        ThroughputVerdict::Severe {
            observed: observed_tokens_per_sec,
            target: target_tokens_per_sec,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_token_throughput")?;

    println!("healthy: {:?}", check(95.0, 100.0));
    println!("degraded: {:?}", check(85.0, 100.0));
    println!("severe: {:?}", check(50.0, 100.0));
    println!("invalid: {:?}", check(50.0, 0.0));
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
    fn at_target_healthy() {
        let v = check(100.0, 100.0);
        assert!(matches!(v, ThroughputVerdict::Healthy { .. }));
    }

    #[test]
    fn slightly_below_healthy() {
        let v = check(95.0, 100.0);
        assert!(matches!(v, ThroughputVerdict::Healthy { .. }));
    }

    #[test]
    fn moderately_below_degraded() {
        let v = check(85.0, 100.0);
        assert!(matches!(v, ThroughputVerdict::Degraded { .. }));
    }

    #[test]
    fn far_below_severe() {
        let v = check(50.0, 100.0);
        assert!(matches!(v, ThroughputVerdict::Severe { .. }));
    }

    #[test]
    fn boundary_at_95_pct_healthy() {
        let v = check(95.0, 100.0);
        assert!(matches!(v, ThroughputVerdict::Healthy { .. }));
    }

    #[test]
    fn boundary_at_80_pct_degraded() {
        let v = check(80.0, 100.0);
        assert!(matches!(v, ThroughputVerdict::Degraded { .. }));
    }

    #[test]
    fn just_below_80_severe() {
        let v = check(79.0, 100.0);
        assert!(matches!(v, ThroughputVerdict::Severe { .. }));
    }

    #[test]
    fn negative_observed_invalid() {
        assert_eq!(check(-1.0, 100.0), ThroughputVerdict::InvalidConfig);
    }

    #[test]
    fn zero_target_invalid() {
        assert_eq!(check(50.0, 0.0), ThroughputVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(check(f64::NAN, 100.0), ThroughputVerdict::InvalidConfig);
    }

    #[test]
    fn over_target_still_healthy() {
        let v = check(150.0, 100.0);
        assert!(matches!(v, ThroughputVerdict::Healthy { .. }));
    }

    #[test]
    fn deterministic() {
        let a = check(95.0, 100.0);
        let b = check(95.0, 100.0);
        assert_eq!(a, b);
    }
}
