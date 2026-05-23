//! # Monte-Carlo Coupon Collector Problem
//!
//! Sim drawing from N coupons with replacement until all distinct
//! coupons collected. Reports mean draws (theoretical: N × H(N)).
//!
//! Demonstrates the **MC.122** recipe for PMAT-199 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Feller, An Introduction to Probability Theory §IX.3
//!  (1968); coupon collector classic problem.
//!
//! Run with: cargo run --example mc_coupon_collector
//!
//! Added by PMAT-199 (catalog 1414→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum CouponVerdict {
    Ok {
        mean_draws: f64,
        max_draws: u32,
        min_draws: u32,
    },
    InvalidConfig,
}

pub fn simulate(trials: u32, coupons: u32, seed: u64) -> CouponVerdict {
    if trials == 0 || coupons == 0 {
        return CouponVerdict::InvalidConfig;
    }
    let mut total_draws: u64 = 0;
    let mut max_draws: u32 = 0;
    let mut min_draws: u32 = u32::MAX;
    let mut rng_state = seed | 1;
    for _ in 0..trials {
        let mut collected: BTreeSet<u32> = BTreeSet::new();
        let mut draws: u32 = 0;
        while collected.len() < coupons as usize {
            let coupon = ((lcg(&mut rng_state) >> 32) as u32) % coupons;
            collected.insert(coupon);
            draws += 1;
        }
        total_draws += u64::from(draws);
        if draws > max_draws {
            max_draws = draws;
        }
        if draws < min_draws {
            min_draws = draws;
        }
    }
    CouponVerdict::Ok {
        mean_draws: total_draws as f64 / f64::from(trials),
        max_draws,
        min_draws,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_coupon_collector")?;

    println!("10 coupons: {:?}", simulate(1000, 10, 42));
    println!("50 coupons: {:?}", simulate(1000, 50, 42));
    println!("invalid: {:?}", simulate(0, 10, 42));
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
    fn mean_close_to_theoretical_for_small_n() {
        // Theoretical: 10 × H(10) = 10 × 2.929 = 29.29.
        let v = simulate(2000, 10, 42);
        if let CouponVerdict::Ok { mean_draws, .. } = v {
            assert!((mean_draws - 29.29).abs() < 5.0);
        }
    }

    #[test]
    fn invalid_zero_trials() {
        assert_eq!(simulate(0, 10, 42), CouponVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_coupons() {
        assert_eq!(simulate(100, 0, 42), CouponVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 10, 42);
        let b = simulate(500, 10, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn min_at_least_n() {
        let v = simulate(500, 10, 42);
        if let CouponVerdict::Ok { min_draws, .. } = v {
            assert!(min_draws >= 10);
        }
    }

    #[test]
    fn mean_le_max() {
        let v = simulate(500, 10, 42);
        if let CouponVerdict::Ok {
            mean_draws,
            max_draws,
            ..
        } = v
        {
            assert!(mean_draws <= f64::from(max_draws));
        }
    }

    #[test]
    fn max_ge_min() {
        let v = simulate(500, 10, 42);
        if let CouponVerdict::Ok {
            max_draws,
            min_draws,
            ..
        } = v
        {
            assert!(max_draws >= min_draws);
        }
    }

    #[test]
    fn larger_n_more_draws() {
        let small = simulate(500, 5, 42);
        let big = simulate(500, 50, 42);
        if let (CouponVerdict::Ok { mean_draws: s, .. }, CouponVerdict::Ok { mean_draws: b, .. }) =
            (small, big)
        {
            assert!(b > s);
        }
    }

    #[test]
    fn single_coupon_one_draw() {
        let v = simulate(100, 1, 42);
        if let CouponVerdict::Ok { mean_draws, .. } = v {
            assert!((mean_draws - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn finite_outputs() {
        let v = simulate(100, 10, 42);
        if let CouponVerdict::Ok { mean_draws, .. } = v {
            assert!(mean_draws.is_finite());
        }
    }

    #[test]
    fn many_coupons_handled() {
        let v = simulate(20, 200, 42);
        assert!(matches!(v, CouponVerdict::Ok { .. }));
    }
}
