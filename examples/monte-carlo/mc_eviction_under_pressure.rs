//! # Monte-Carlo Eviction Under Memory Pressure
//!
//! Simulate cache eviction rate as a function of memory pressure.
//! Returns observed eviction rate and time-to-OOM if pressure is
//! above the eviction-rate ceiling.
//!
//! Demonstrates the **MC.30** recipe for PMAT-167 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cgroup memory pressure + LRU eviction policies.
//!
//! Run with: cargo run --example mc_eviction_under_pressure
//!
//! Added by PMAT-167 (catalog 1126→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum EvictVerdict {
    Stable {
        eviction_rate: f64,
    },
    Stressed {
        eviction_rate: f64,
        time_to_oom_secs: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    cache_size_bytes: u64,
    pressure_bytes_per_sec: f64,
    eviction_ceiling_per_sec: f64,
) -> EvictVerdict {
    if cache_size_bytes == 0
        || !pressure_bytes_per_sec.is_finite()
        || pressure_bytes_per_sec < 0.0
        || !eviction_ceiling_per_sec.is_finite()
        || eviction_ceiling_per_sec <= 0.0
    {
        return EvictVerdict::InvalidConfig;
    }
    if pressure_bytes_per_sec <= eviction_ceiling_per_sec {
        return EvictVerdict::Stable {
            eviction_rate: pressure_bytes_per_sec,
        };
    }
    let net_pressure = pressure_bytes_per_sec - eviction_ceiling_per_sec;
    let time_to_oom_secs = cache_size_bytes as f64 / net_pressure;
    EvictVerdict::Stressed {
        eviction_rate: eviction_ceiling_per_sec,
        time_to_oom_secs,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_eviction_under_pressure")?;

    println!(
        "stable: {:?}",
        simulate(10_000_000_000, 100_000.0, 200_000.0)
    );
    println!(
        "stressed: {:?}",
        simulate(10_000_000_000, 500_000.0, 200_000.0)
    );
    println!("invalid: {:?}", simulate(0, 100.0, 200.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn pressure_under_ceiling_stable() {
        let v = simulate(10_000_000_000, 100_000.0, 200_000.0);
        assert!(matches!(v, EvictVerdict::Stable { .. }));
    }

    #[test]
    fn pressure_over_ceiling_stressed() {
        let v = simulate(10_000_000_000, 500_000.0, 200_000.0);
        assert!(matches!(v, EvictVerdict::Stressed { .. }));
    }

    #[test]
    fn at_ceiling_stable() {
        let v = simulate(10_000_000_000, 200_000.0, 200_000.0);
        assert!(matches!(v, EvictVerdict::Stable { .. }));
    }

    #[test]
    fn invalid_zero_cache() {
        assert_eq!(simulate(0, 100.0, 200.0), EvictVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_neg_pressure() {
        assert_eq!(simulate(1000, -100.0, 200.0), EvictVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_ceiling() {
        assert_eq!(simulate(1000, 100.0, 0.0), EvictVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(simulate(1000, f64::NAN, 200.0), EvictVerdict::InvalidConfig);
    }

    #[test]
    fn time_to_oom_correct() {
        // 10GB cache, net pressure = 300_000/s → 33333 sec.
        let v = simulate(10_000_000_000, 500_000.0, 200_000.0);
        if let EvictVerdict::Stressed {
            time_to_oom_secs, ..
        } = v
        {
            assert!((time_to_oom_secs - 33333.33).abs() < 1.0);
        }
    }

    #[test]
    fn higher_pressure_faster_oom() {
        let lo = simulate(10_000_000_000, 300_000.0, 200_000.0);
        let hi = simulate(10_000_000_000, 1_000_000.0, 200_000.0);
        if let (
            EvictVerdict::Stressed {
                time_to_oom_secs: l,
                ..
            },
            EvictVerdict::Stressed {
                time_to_oom_secs: h,
                ..
            },
        ) = (lo, hi)
        {
            assert!(h < l);
        }
    }

    #[test]
    fn deterministic() {
        let a = simulate(1_000_000, 100.0, 200.0);
        let b = simulate(1_000_000, 100.0, 200.0);
        assert_eq!(a, b);
    }
}
