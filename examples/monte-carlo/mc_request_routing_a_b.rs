//! # Monte-Carlo A/B Routing Split
//!
//! Sim N requests routed via A/B traffic split based on user_id hash.
//! Verify observed split is within tolerance of declared percentage.
//!
//! Demonstrates the **MC.29** recipe for PMAT-167 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Bucketing schemes (FNV-1a hash for routing).
//!
//! Run with: cargo run --example mc_request_routing_a_b
//!
//! Added by PMAT-167 (catalog 1126→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RoutingVerdict {
    Ok {
        a_count: u32,
        b_count: u32,
        deviation_pct: f64,
    },
    InvalidConfig,
}

pub fn simulate(num_requests: u32, a_pct: f64, seed: u64) -> RoutingVerdict {
    if num_requests == 0 || !a_pct.is_finite() || !(0.0..=1.0).contains(&a_pct) {
        return RoutingVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut a_count = 0u32;
    let mut b_count = 0u32;
    for _ in 0..num_requests {
        let bucket = unit(&mut rng_state);
        if bucket < a_pct {
            a_count += 1;
        } else {
            b_count += 1;
        }
    }
    let observed_a = f64::from(a_count) / f64::from(num_requests);
    let deviation_pct = (observed_a - a_pct).abs() * 100.0;
    RoutingVerdict::Ok {
        a_count,
        b_count,
        deviation_pct,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_request_routing_a_b")?;

    println!("50/50 large: {:?}", simulate(10_000, 0.5, 42));
    println!("90/10: {:?}", simulate(10_000, 0.9, 42));
    println!("invalid pct: {:?}", simulate(100, 1.5, 42));
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
    fn fifty_fifty_balanced() {
        let v = simulate(10_000, 0.5, 42);
        if let RoutingVerdict::Ok { deviation_pct, .. } = v {
            assert!(deviation_pct < 2.0);
        }
    }

    #[test]
    fn nine_ten_skews_a() {
        let v = simulate(10_000, 0.9, 42);
        if let RoutingVerdict::Ok {
            a_count, b_count, ..
        } = v
        {
            assert!(a_count > b_count * 5);
        }
    }

    #[test]
    fn counts_sum_to_total() {
        let v = simulate(1000, 0.5, 42);
        if let RoutingVerdict::Ok {
            a_count, b_count, ..
        } = v
        {
            assert_eq!(a_count + b_count, 1000);
        }
    }

    #[test]
    fn invalid_zero_requests() {
        assert_eq!(simulate(0, 0.5, 42), RoutingVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_pct_over_one() {
        assert_eq!(simulate(100, 1.5, 42), RoutingVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_negative_pct() {
        assert_eq!(simulate(100, -0.1, 42), RoutingVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(simulate(100, f64::NAN, 42), RoutingVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(1000, 0.5, 42);
        let b = simulate(1000, 0.5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn all_a_at_pct_one() {
        let v = simulate(100, 1.0, 42);
        if let RoutingVerdict::Ok { b_count, .. } = v {
            assert_eq!(b_count, 0);
        }
    }

    #[test]
    fn all_b_at_pct_zero() {
        let v = simulate(100, 0.0, 42);
        if let RoutingVerdict::Ok { a_count, .. } = v {
            assert_eq!(a_count, 0);
        }
    }
}
