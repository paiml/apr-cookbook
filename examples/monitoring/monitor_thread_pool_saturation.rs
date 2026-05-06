//! # Monitoring Thread-Pool Saturation
//!
//! When inference workers are saturated (queue ≈ pool size, latency
//! climbing), reject incoming requests early instead of pile-up.
//! Detector compares queue length and utilization to per-server limits.
//!
//! Demonstrates the **MON.38** recipe for PMAT-155 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Little's Law (queueing theory) + load-shedding in TCP.
//!
//! Run with: cargo run --example monitor_thread_pool_saturation
//!
//! Added by PMAT-155 (catalog 1018→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SaturationVerdict {
    Ok { utilization: f64 },
    NearSaturation { utilization: f64 },
    Saturated { utilization: f64, queue_ratio: f64 },
    InvalidConfig,
}

pub fn check(active_threads: u32, pool_size: u32, queue_len: u32) -> SaturationVerdict {
    if pool_size == 0 {
        return SaturationVerdict::InvalidConfig;
    }
    if active_threads > pool_size {
        return SaturationVerdict::InvalidConfig;
    }
    let utilization = f64::from(active_threads) / f64::from(pool_size);
    let queue_ratio = f64::from(queue_len) / f64::from(pool_size);
    if utilization >= 0.95 && queue_ratio >= 1.0 {
        SaturationVerdict::Saturated {
            utilization,
            queue_ratio,
        }
    } else if utilization >= 0.80 {
        SaturationVerdict::NearSaturation { utilization }
    } else {
        SaturationVerdict::Ok { utilization }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_thread_pool_saturation")?;

    println!("light load: {:?}", check(2, 8, 0));
    println!("near saturation: {:?}", check(7, 8, 2));
    println!("saturated: {:?}", check(8, 8, 12));
    println!("invalid: {:?}", check(0, 0, 0));
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
    fn light_load_ok() {
        let v = check(2, 8, 0);
        assert!(matches!(v, SaturationVerdict::Ok { .. }));
    }

    #[test]
    fn near_saturation_at_80_pct() {
        // 7/8 = 87.5% → near.
        let v = check(7, 8, 2);
        assert!(matches!(v, SaturationVerdict::NearSaturation { .. }));
    }

    #[test]
    fn saturated_at_full() {
        // 8/8 = 100%, queue 12 (≥ 8) → saturated.
        let v = check(8, 8, 12);
        assert!(matches!(v, SaturationVerdict::Saturated { .. }));
    }

    #[test]
    fn zero_pool_invalid() {
        assert_eq!(check(0, 0, 0), SaturationVerdict::InvalidConfig);
    }

    #[test]
    fn over_pool_threads_invalid() {
        assert_eq!(check(10, 8, 0), SaturationVerdict::InvalidConfig);
    }

    #[test]
    fn boundary_at_80_near() {
        // 80% exactly → near saturation.
        let v = check(8, 10, 0);
        assert!(matches!(v, SaturationVerdict::NearSaturation { .. }));
    }

    #[test]
    fn boundary_at_95_with_queue_saturated() {
        // Just at 95% with queue at limit.
        let v = check(19, 20, 20);
        assert!(matches!(v, SaturationVerdict::Saturated { .. }));
    }

    #[test]
    fn high_util_no_queue_just_near() {
        // 100% utilization but queue is empty → near, not saturated.
        let v = check(8, 8, 0);
        assert!(matches!(v, SaturationVerdict::NearSaturation { .. }));
    }

    #[test]
    fn util_value_correct() {
        let v = check(4, 8, 0);
        if let SaturationVerdict::Ok { utilization } = v {
            assert!((utilization - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn queue_ratio_in_saturated() {
        let v = check(8, 8, 16);
        if let SaturationVerdict::Saturated { queue_ratio, .. } = v {
            assert!((queue_ratio - 2.0).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic() {
        let a = check(7, 8, 2);
        let b = check(7, 8, 2);
        assert_eq!(a, b);
    }
}
