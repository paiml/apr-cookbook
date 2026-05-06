//! # Monitoring DB Connection-Pool Exhaustion
//!
//! Pool exhaustion: requests stall while waiting for a free DB
//! connection. Detector compares active connections + pending waiters
//! to pool size and threshold-flags risk.
//!
//! Demonstrates the **MON.50** recipe for PMAT-159 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PostgreSQL pool exhaustion + HikariCP "leak" detection.
//!
//! Run with: cargo run --example monitor_db_pool_exhaustion
//!
//! Added by PMAT-159 (catalog 1054→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PoolVerdict {
    Healthy { utilization_pct: f64 },
    NearExhaustion { utilization_pct: f64 },
    Exhausted { waiting: u32 },
    InvalidConfig,
}

pub fn check(active_connections: u32, pool_size: u32, waiters: u32) -> PoolVerdict {
    if pool_size == 0 || active_connections > pool_size {
        return PoolVerdict::InvalidConfig;
    }
    if active_connections == pool_size && waiters > 0 {
        return PoolVerdict::Exhausted { waiting: waiters };
    }
    let utilization_pct = (f64::from(active_connections) / f64::from(pool_size)) * 100.0;
    if utilization_pct >= 80.0 {
        PoolVerdict::NearExhaustion { utilization_pct }
    } else {
        PoolVerdict::Healthy { utilization_pct }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_db_pool_exhaustion")?;

    println!("healthy: {:?}", check(5, 20, 0));
    println!("near: {:?}", check(17, 20, 0));
    println!("exhausted: {:?}", check(20, 20, 5));
    println!("invalid: {:?}", check(25, 20, 0));
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
    fn healthy_low_util() {
        let v = check(5, 20, 0);
        assert!(matches!(v, PoolVerdict::Healthy { .. }));
    }

    #[test]
    fn near_exhaustion_above_80() {
        let v = check(17, 20, 0);
        assert!(matches!(v, PoolVerdict::NearExhaustion { .. }));
    }

    #[test]
    fn exhausted_when_full_with_waiters() {
        let v = check(20, 20, 5);
        if let PoolVerdict::Exhausted { waiting } = v {
            assert_eq!(waiting, 5);
        }
    }

    #[test]
    fn full_no_waiters_near() {
        // Fully active but nobody waiting → just near-exhaustion (100%).
        let v = check(20, 20, 0);
        assert!(matches!(v, PoolVerdict::NearExhaustion { .. }));
    }

    #[test]
    fn invalid_zero_pool() {
        assert_eq!(check(0, 0, 0), PoolVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_active_over_pool() {
        assert_eq!(check(25, 20, 0), PoolVerdict::InvalidConfig);
    }

    #[test]
    fn boundary_at_80_pct_near() {
        let v = check(8, 10, 0);
        assert!(matches!(v, PoolVerdict::NearExhaustion { .. }));
    }

    #[test]
    fn zero_active_healthy() {
        let v = check(0, 20, 0);
        if let PoolVerdict::Healthy { utilization_pct } = v {
            assert!((utilization_pct - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn util_value_correct() {
        let v = check(10, 20, 0);
        if let PoolVerdict::Healthy { utilization_pct } = v {
            assert!((utilization_pct - 50.0).abs() < 1e-9);
        }
    }

    #[test]
    fn waiters_when_not_exhausted_no_flag() {
        // Even with waiters, if pool isn't full, it's not "exhausted".
        let v = check(5, 20, 3);
        assert!(matches!(v, PoolVerdict::Healthy { .. }));
    }

    #[test]
    fn deterministic() {
        let a = check(17, 20, 0);
        let b = check(17, 20, 0);
        assert_eq!(a, b);
    }
}
