//! # Monitoring Capacity Planner
//!
//! Required worker count = ceil(peak_qps × p99_secs / target_utilization).
//! With:
//!   peak_qps from forecasted traffic (typically prior-period × growth)
//!   p99_secs from current latency telemetry
//!   target_utilization = 0.7 (leaves 30% headroom for spikes)
//!
//! Plus tier classification: WellProvisioned (utilization < 50%),
//! Healthy (50-70%), AtCapacity (70-85%), Overloaded (≥85%).
//!
//! Demonstrates the **MON.23** recipe for PMAT-141 (monitoring round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Brendan Gregg's USE method (Utilization, Saturation, Errors).
//!
//! Run with: cargo run --example monitor_capacity_planner
//!
//! Added by PMAT-141 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const DEFAULT_TARGET_UTIL: f64 = 0.70;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacityTier {
    WellProvisioned,
    Healthy,
    AtCapacity,
    Overloaded,
}

#[derive(Debug, PartialEq)]
pub enum CapacityVerdict {
    Ok {
        required_workers: u32,
        current_utilization: f64,
        tier: CapacityTier,
    },
    InvalidQps,
    InvalidLatency,
    InvalidWorkerCount,
}

pub fn plan(peak_qps: f64, p99_secs: f64, current_workers: u32) -> CapacityVerdict {
    if !peak_qps.is_finite() || peak_qps <= 0.0 {
        return CapacityVerdict::InvalidQps;
    }
    if !p99_secs.is_finite() || p99_secs <= 0.0 {
        return CapacityVerdict::InvalidLatency;
    }
    if current_workers == 0 {
        return CapacityVerdict::InvalidWorkerCount;
    }
    let required = (peak_qps * p99_secs / DEFAULT_TARGET_UTIL).ceil() as u32;
    let current_capacity_qps = f64::from(current_workers) / p99_secs;
    let current_utilization = peak_qps / current_capacity_qps;
    let tier = if current_utilization < 0.50 {
        CapacityTier::WellProvisioned
    } else if current_utilization < 0.70 {
        CapacityTier::Healthy
    } else if current_utilization < 0.85 {
        CapacityTier::AtCapacity
    } else {
        CapacityTier::Overloaded
    };
    CapacityVerdict::Ok {
        required_workers: required,
        current_utilization,
        tier,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_capacity_planner")?;

    // 100 qps × 0.5s p99 / 0.7 = 71.4 → ceil 72 workers.
    println!("100 qps with 50 workers: {:?}", plan(100.0, 0.5, 50));
    println!("100 qps with 100 workers: {:?}", plan(100.0, 0.5, 100));
    println!("1000 qps with 100 workers: {:?}", plan(1000.0, 0.1, 100));
    println!("invalid: {:?}", plan(0.0, 0.5, 50));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_capacity() {
        // 100 qps × 0.5 s = 50; / 0.7 = 71.4 → 72.
        let v = plan(100.0, 0.5, 50);
        if let CapacityVerdict::Ok {
            required_workers, ..
        } = v
        {
            assert_eq!(required_workers, 72);
        }
    }

    #[test]
    fn well_provisioned_at_low_utilization() {
        let v = plan(50.0, 0.5, 200);
        if let CapacityVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CapacityTier::WellProvisioned);
        }
    }

    #[test]
    fn at_capacity_above_70() {
        // util = 0.5 × 100 / 60 = 0.833.
        let v = plan(100.0, 0.5, 60);
        if let CapacityVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CapacityTier::AtCapacity);
        }
    }

    #[test]
    fn overloaded_above_85() {
        // util = 0.9 × 100 / 100 = 0.9.
        let v = plan(100.0, 0.9, 100);
        if let CapacityVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CapacityTier::Overloaded);
        }
    }

    #[test]
    fn invalid_qps_rejected() {
        assert_eq!(plan(0.0, 0.5, 50), CapacityVerdict::InvalidQps);
    }

    #[test]
    fn invalid_latency_rejected() {
        assert_eq!(plan(100.0, 0.0, 50), CapacityVerdict::InvalidLatency);
    }

    #[test]
    fn invalid_zero_workers_rejected() {
        assert_eq!(plan(100.0, 0.5, 0), CapacityVerdict::InvalidWorkerCount);
    }

    #[test]
    fn higher_qps_more_workers_required() {
        let v_low = plan(100.0, 0.5, 100);
        let v_high = plan(1000.0, 0.5, 100);
        if let (
            CapacityVerdict::Ok {
                required_workers: l,
                ..
            },
            CapacityVerdict::Ok {
                required_workers: h,
                ..
            },
        ) = (v_low, v_high)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn higher_latency_more_workers_required() {
        let v_fast = plan(100.0, 0.1, 100);
        let v_slow = plan(100.0, 0.5, 100);
        if let (
            CapacityVerdict::Ok {
                required_workers: f,
                ..
            },
            CapacityVerdict::Ok {
                required_workers: s,
                ..
            },
        ) = (v_fast, v_slow)
        {
            assert!(s > f);
        }
    }

    #[test]
    fn current_util_returned_for_diagnostics() {
        if let CapacityVerdict::Ok {
            current_utilization,
            ..
        } = plan(100.0, 0.5, 100)
        {
            // 100 qps × 0.5 s / 100 workers = 0.5.
            assert!((current_utilization - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn nan_rejected() {
        assert_eq!(plan(f64::NAN, 0.5, 50), CapacityVerdict::InvalidQps);
    }
}
