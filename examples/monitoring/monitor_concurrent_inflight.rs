//! # Monitoring Concurrent In-Flight Requests
//!
//! Watch concurrent in-flight inference requests vs the server's
//! configured max-concurrency. Verdict:
//!   <50%: Underutilized (too much capacity)
//!   50-90%: Healthy
//!   ≥90%: NearLimit (admit-rate should drop)
//!
//! Demonstrates the **MON.44** recipe for PMAT-157 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Little's Law (queueing) for concurrency planning.
//!
//! Run with: cargo run --example monitor_concurrent_inflight
//!
//! Added by PMAT-157 (catalog 1036→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum InflightVerdict {
    Underutilized { utilization_pct: f64 },
    Healthy { utilization_pct: f64 },
    NearLimit { utilization_pct: f64 },
    OverLimit { excess: u32 },
    InvalidConfig,
}

pub fn check(inflight_requests: u32, max_concurrency: u32) -> InflightVerdict {
    if max_concurrency == 0 {
        return InflightVerdict::InvalidConfig;
    }
    if inflight_requests > max_concurrency {
        return InflightVerdict::OverLimit {
            excess: inflight_requests - max_concurrency,
        };
    }
    let utilization_pct = (f64::from(inflight_requests) / f64::from(max_concurrency)) * 100.0;
    if utilization_pct >= 90.0 {
        InflightVerdict::NearLimit { utilization_pct }
    } else if utilization_pct >= 50.0 {
        InflightVerdict::Healthy { utilization_pct }
    } else {
        InflightVerdict::Underutilized { utilization_pct }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_concurrent_inflight")?;

    println!("under: {:?}", check(10, 100));
    println!("healthy: {:?}", check(60, 100));
    println!("near limit: {:?}", check(95, 100));
    println!("over: {:?}", check(105, 100));
    println!("invalid: {:?}", check(10, 0));
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
    fn under_50_pct() {
        let v = check(10, 100);
        assert!(matches!(v, InflightVerdict::Underutilized { .. }));
    }

    #[test]
    fn healthy_50_to_90() {
        let v = check(60, 100);
        assert!(matches!(v, InflightVerdict::Healthy { .. }));
    }

    #[test]
    fn near_limit_above_90() {
        let v = check(95, 100);
        assert!(matches!(v, InflightVerdict::NearLimit { .. }));
    }

    #[test]
    fn over_limit() {
        let v = check(105, 100);
        if let InflightVerdict::OverLimit { excess } = v {
            assert_eq!(excess, 5);
        }
    }

    #[test]
    fn zero_max_invalid() {
        assert_eq!(check(10, 0), InflightVerdict::InvalidConfig);
    }

    #[test]
    fn boundary_at_50_healthy() {
        let v = check(50, 100);
        assert!(matches!(v, InflightVerdict::Healthy { .. }));
    }

    #[test]
    fn boundary_at_90_near_limit() {
        let v = check(90, 100);
        assert!(matches!(v, InflightVerdict::NearLimit { .. }));
    }

    #[test]
    fn at_max_near_limit() {
        let v = check(100, 100);
        assert!(matches!(v, InflightVerdict::NearLimit { .. }));
    }

    #[test]
    fn zero_inflight_underutilized() {
        let v = check(0, 100);
        if let InflightVerdict::Underutilized { utilization_pct } = v {
            assert!((utilization_pct - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn one_excess_returned() {
        let v = check(101, 100);
        if let InflightVerdict::OverLimit { excess } = v {
            assert_eq!(excess, 1);
        }
    }

    #[test]
    fn deterministic() {
        let a = check(60, 100);
        let b = check(60, 100);
        assert_eq!(a, b);
    }
}
