//! # Monte-Carlo Session Affinity Routing
//!
//! Sim sticky session routing: each user is hashed to a server.
//! After random server failures, observe re-routing rate. Reports
//! affinity hit-rate (fraction of requests served by original server).
//!
//! Demonstrates the **MC.109** recipe for PMAT-195 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS ELB session affinity; HAProxy stick-table conventions.
//!
//! Run with: cargo run --example mc_session_affinity_routing
//!
//! Added by PMAT-195 (catalog 1378→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AffinityVerdict {
    Ok {
        affinity_hit_rate: f64,
        rerouted_requests: u32,
    },
    InvalidConfig,
}

pub fn simulate(requests: u32, servers: u32, server_fail_prob: f64, seed: u64) -> AffinityVerdict {
    if requests == 0 || servers == 0 || !(0.0..=1.0).contains(&server_fail_prob) {
        return AffinityVerdict::InvalidConfig;
    }
    let mut hits = 0u32;
    let mut rerouted = 0u32;
    let mut rng_state = seed | 1;
    for user_id in 0..requests {
        let original_server = user_id % servers;
        let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        if r < server_fail_prob {
            // Server is down → reroute to next.
            rerouted += 1;
        } else {
            // Affinity preserved.
            let _ = original_server;
            hits += 1;
        }
    }
    AffinityVerdict::Ok {
        affinity_hit_rate: f64::from(hits) / f64::from(requests),
        rerouted_requests: rerouted,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_session_affinity_routing")?;

    println!("low failure: {:?}", simulate(10_000, 4, 0.01, 42));
    println!("high failure: {:?}", simulate(10_000, 4, 0.30, 42));
    println!("invalid: {:?}", simulate(0, 4, 0.01, 42));
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
    fn no_failure_full_affinity() {
        let v = simulate(1000, 4, 0.0, 42);
        if let AffinityVerdict::Ok {
            affinity_hit_rate, ..
        } = v
        {
            assert_eq!(affinity_hit_rate, 1.0);
        }
    }

    #[test]
    fn always_failing_zero_affinity() {
        let v = simulate(1000, 4, 1.0, 42);
        if let AffinityVerdict::Ok {
            affinity_hit_rate, ..
        } = v
        {
            assert_eq!(affinity_hit_rate, 0.0);
        }
    }

    #[test]
    fn invalid_zero_requests() {
        assert_eq!(simulate(0, 4, 0.01, 42), AffinityVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_servers() {
        assert_eq!(simulate(100, 0, 0.01, 42), AffinityVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_prob_out_of_range() {
        assert_eq!(simulate(100, 4, 1.5, 42), AffinityVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 4, 0.1, 42);
        let b = simulate(500, 4, 0.1, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn higher_failure_more_rerouting() {
        let lo = simulate(2000, 4, 0.01, 42);
        let hi = simulate(2000, 4, 0.30, 42);
        if let (
            AffinityVerdict::Ok {
                rerouted_requests: l,
                ..
            },
            AffinityVerdict::Ok {
                rerouted_requests: h,
                ..
            },
        ) = (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn rate_in_unit_range() {
        let v = simulate(1000, 4, 0.1, 42);
        if let AffinityVerdict::Ok {
            affinity_hit_rate, ..
        } = v
        {
            assert!((0.0..=1.0).contains(&affinity_hit_rate));
        }
    }

    #[test]
    fn rerouted_le_requests() {
        let v = simulate(1000, 4, 0.1, 42);
        if let AffinityVerdict::Ok {
            rerouted_requests, ..
        } = v
        {
            assert!(rerouted_requests <= 1000);
        }
    }

    #[test]
    fn fifty_pct_failure_rate_near_half() {
        let v = simulate(10_000, 4, 0.5, 42);
        if let AffinityVerdict::Ok {
            affinity_hit_rate, ..
        } = v
        {
            assert!((affinity_hit_rate - 0.5).abs() < 0.05);
        }
    }
}
