//! # Monte-Carlo Load-Balancer Health Check
//!
//! Sim periodic health probes against a backend with random down
//! periods. Reports observed uptime % and false-down probe rate.
//!
//! Demonstrates the **MC.99** recipe for PMAT-192 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HAProxy/nginx active health-check protocols; Google
//!  SRE Workbook ch.6 (probes).
//!
//! Run with: cargo run --example mc_loadbalancer_health_check
//!
//! Added by PMAT-192 (catalog 1351→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum HealthVerdict {
    Ok {
        uptime_pct: f64,
        probes_total: u32,
        probes_failed: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    seconds: u32,
    probe_interval_sec: u32,
    p_down_per_sec: f64,
    seed: u64,
) -> HealthVerdict {
    if seconds == 0 || probe_interval_sec == 0 || !(0.0..=1.0).contains(&p_down_per_sec) {
        return HealthVerdict::InvalidConfig;
    }
    let mut up: bool = true;
    let mut up_seconds: u32 = 0;
    let mut probes_total: u32 = 0;
    let mut probes_failed: u32 = 0;
    let mut rng_state = seed | 1;
    for sec in 0..seconds {
        // Health-check probe occurs at multiples of probe_interval_sec.
        if sec % probe_interval_sec == 0 {
            probes_total += 1;
            if !up {
                probes_failed += 1;
            }
        }
        if up {
            up_seconds += 1;
        }
        // Transition: up → down with prob p_down; down → up with prob 1/recovery.
        let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        if up && r < p_down_per_sec {
            up = false;
        } else if !up && r < 0.5 {
            up = true;
        }
    }
    let uptime_pct = 100.0 * f64::from(up_seconds) / f64::from(seconds);
    HealthVerdict::Ok {
        uptime_pct,
        probes_total,
        probes_failed,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_loadbalancer_health_check")?;

    println!("stable backend: {:?}", simulate(3600, 30, 0.001, 42));
    println!("flaky backend: {:?}", simulate(3600, 30, 0.05, 42));
    println!("invalid: {:?}", simulate(0, 30, 0.001, 42));
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
    fn stable_backend_high_uptime() {
        let v = simulate(3600, 30, 0.0001, 42);
        if let HealthVerdict::Ok { uptime_pct, .. } = v {
            assert!(uptime_pct > 95.0);
        }
    }

    #[test]
    fn flaky_backend_low_uptime() {
        let v = simulate(3600, 30, 0.50, 42);
        if let HealthVerdict::Ok { uptime_pct, .. } = v {
            assert!(uptime_pct < 80.0);
        }
    }

    #[test]
    fn invalid_zero_seconds() {
        assert_eq!(simulate(0, 30, 0.001, 42), HealthVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_interval() {
        assert_eq!(simulate(3600, 0, 0.001, 42), HealthVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_prob_out_of_range() {
        assert_eq!(simulate(3600, 30, 1.5, 42), HealthVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(600, 30, 0.01, 42);
        let b = simulate(600, 30, 0.01, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn probes_total_correct() {
        let v = simulate(60, 10, 0.0, 42);
        if let HealthVerdict::Ok { probes_total, .. } = v {
            // 60 sec / 10 sec interval → 6 probes (sec 0, 10, ..., 50).
            assert_eq!(probes_total, 6);
        }
    }

    #[test]
    fn always_up_no_failures() {
        let v = simulate(3600, 30, 0.0, 42);
        if let HealthVerdict::Ok {
            uptime_pct,
            probes_failed,
            ..
        } = v
        {
            assert_eq!(uptime_pct, 100.0);
            assert_eq!(probes_failed, 0);
        }
    }

    #[test]
    fn uptime_in_unit_range() {
        let v = simulate(600, 30, 0.05, 42);
        if let HealthVerdict::Ok { uptime_pct, .. } = v {
            assert!((0.0..=100.0).contains(&uptime_pct));
        }
    }

    #[test]
    fn higher_failure_more_failed_probes() {
        let lo = simulate(3600, 30, 0.005, 42);
        let hi = simulate(3600, 30, 0.20, 42);
        if let (
            HealthVerdict::Ok {
                probes_failed: l, ..
            },
            HealthVerdict::Ok {
                probes_failed: h, ..
            },
        ) = (lo, hi)
        {
            assert!(h >= l);
        }
    }

    #[test]
    fn probes_failed_le_probes_total() {
        let v = simulate(600, 30, 0.10, 42);
        if let HealthVerdict::Ok {
            probes_total,
            probes_failed,
            ..
        } = v
        {
            assert!(probes_failed <= probes_total);
        }
    }
}
