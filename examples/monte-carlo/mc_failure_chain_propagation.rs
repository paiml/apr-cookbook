//! # Monte-Carlo Failure-Chain Propagation
//!
//! Simulate cascading failures: each service in a chain fails with
//! probability p, conditional on its parent failing. Returns expected
//! failure depth per simulation run.
//!
//! Demonstrates the **MC.09** recipe for PMAT-160 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Reliability theory; "Bayesian network" failure cascade.
//!
//! Run with: cargo run --example mc_failure_chain_propagation
//!
//! Added by PMAT-160 (catalog 1063→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CascadeVerdict {
    Ok {
        mean_depth: f64,
        max_depth: u32,
        full_cascade_pct: f64,
    },
    InvalidConfig,
}

pub fn simulate(chain_length: u32, fail_prob: f64, num_runs: u32, seed: u64) -> CascadeVerdict {
    if chain_length == 0
        || num_runs == 0
        || !fail_prob.is_finite()
        || !(0.0..=1.0).contains(&fail_prob)
    {
        return CascadeVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut total_depth: u64 = 0;
    let mut max_depth: u32 = 0;
    let mut full_cascades: u32 = 0;
    for _ in 0..num_runs {
        let mut depth: u32 = 0;
        for _ in 0..chain_length {
            if unit(&mut rng_state) < fail_prob {
                depth += 1;
            } else {
                break;
            }
        }
        total_depth += u64::from(depth);
        max_depth = max_depth.max(depth);
        if depth == chain_length {
            full_cascades += 1;
        }
    }
    let mean_depth = total_depth as f64 / f64::from(num_runs);
    let full_cascade_pct = (f64::from(full_cascades) / f64::from(num_runs)) * 100.0;
    CascadeVerdict::Ok {
        mean_depth,
        max_depth,
        full_cascade_pct,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_failure_chain_propagation")?;

    println!("low p: {:?}", simulate(5, 0.1, 1000, 42));
    println!("medium p: {:?}", simulate(5, 0.5, 1000, 42));
    println!("high p: {:?}", simulate(5, 0.9, 1000, 42));
    println!("invalid: {:?}", simulate(0, 0.5, 1000, 42));
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
    fn low_prob_low_cascade() {
        let v = simulate(5, 0.1, 1000, 42);
        if let CascadeVerdict::Ok { mean_depth, .. } = v {
            assert!(mean_depth < 1.0);
        }
    }

    #[test]
    fn high_prob_full_cascade_likely() {
        let v = simulate(5, 0.9, 1000, 42);
        if let CascadeVerdict::Ok {
            full_cascade_pct, ..
        } = v
        {
            assert!(full_cascade_pct > 30.0);
        }
    }

    #[test]
    fn zero_prob_zero_depth() {
        let v = simulate(5, 0.0, 100, 42);
        if let CascadeVerdict::Ok {
            mean_depth,
            max_depth,
            ..
        } = v
        {
            assert!((mean_depth - 0.0).abs() < 1e-9);
            assert_eq!(max_depth, 0);
        }
    }

    #[test]
    fn one_prob_full_cascade() {
        let v = simulate(5, 1.0, 100, 42);
        if let CascadeVerdict::Ok {
            mean_depth,
            full_cascade_pct,
            ..
        } = v
        {
            assert!((mean_depth - 5.0).abs() < 1e-9);
            assert!((full_cascade_pct - 100.0).abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_zero_chain() {
        assert_eq!(simulate(0, 0.5, 100, 42), CascadeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_runs() {
        assert_eq!(simulate(5, 0.5, 0, 42), CascadeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_negative_prob() {
        assert_eq!(simulate(5, -0.1, 100, 42), CascadeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_over_one_prob() {
        assert_eq!(simulate(5, 1.5, 100, 42), CascadeVerdict::InvalidConfig);
    }

    #[test]
    fn nan_prob_invalid() {
        assert_eq!(
            simulate(5, f64::NAN, 100, 42),
            CascadeVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = simulate(5, 0.5, 1000, 42);
        let b = simulate(5, 0.5, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn max_depth_bounded_by_chain() {
        let v = simulate(7, 0.7, 100, 42);
        if let CascadeVerdict::Ok { max_depth, .. } = v {
            assert!(max_depth <= 7);
        }
    }
}
