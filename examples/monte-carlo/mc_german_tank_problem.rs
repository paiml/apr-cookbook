//! # Monte-Carlo German Tank Problem
//!
//! Estimate population size N from a sample of k uniformly-drawn
//! integer serial numbers using the minimum-variance unbiased
//! estimator: N̂ = m + m/k - 1, where m = max observed.
//!
//! Demonstrates the **MC.160** recipe for PMAT-212 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Wald (1942); Goodman, "Serial Number Analysis" JASA 1952;
//!  WW2 German tank production estimation.
//!
//! Run with: cargo run --example mc_german_tank_problem
//!
//! Added by PMAT-212 (catalog 1531→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TankVerdict {
    Ok { mean_estimate: u32, true_n: u32 },
    InvalidConfig,
}

pub fn simulate(true_n: u32, sample_size: u32, trials: u32, seed: u64) -> TankVerdict {
    if true_n < 10 || sample_size < 2 || sample_size > true_n || trials < 100 {
        return TankVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut total: u64 = 0;
    for _ in 0..trials {
        // Sample without replacement: random subset of [1, true_n].
        let mut pool: Vec<u32> = (1..=true_n).collect();
        for i in (1..pool.len()).rev() {
            let j = (lcg(&mut state) as usize) % (i + 1);
            pool.swap(i, j);
        }
        let sample: &[u32] = &pool[..sample_size as usize];
        let m = *sample.iter().max().unwrap_or(&1);
        // Goodman MVUE: N̂ = m + m/k - 1
        let n_hat = m as u64 + (m / sample_size) as u64 - 1;
        total += n_hat;
    }
    TankVerdict::Ok {
        mean_estimate: (total / trials as u64) as u32,
        true_n,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_german_tank_problem")?;

    println!("N=1000, k=5: {:?}", simulate(1000, 5, 1000, 42));
    println!("invalid: {:?}", simulate(5, 5, 1000, 42));
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
    fn invalid_too_small_n() {
        assert_eq!(simulate(5, 2, 100, 42), TankVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_sample_too_small() {
        assert_eq!(simulate(100, 1, 100, 42), TankVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_sample_over_n() {
        assert_eq!(simulate(100, 200, 100, 42), TankVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(simulate(100, 10, 50, 42), TankVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 5, 200, 42);
        let b = simulate(100, 5, 200, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn estimate_near_true_n() {
        // Allow ±20% with sample size 10 of 1000.
        let v = simulate(1000, 10, 5000, 42);
        if let TankVerdict::Ok { mean_estimate, .. } = v {
            assert!((800..=1200).contains(&mean_estimate));
        }
    }

    #[test]
    fn larger_sample_within_tight_band() {
        // With k=100 and 5000 trials, estimate should be within ±5%.
        let v = simulate(1000, 100, 5000, 42);
        if let TankVerdict::Ok { mean_estimate, .. } = v {
            assert!((950..=1050).contains(&mean_estimate));
        }
    }

    #[test]
    fn estimate_at_least_one() {
        let v = simulate(100, 5, 100, 42);
        if let TankVerdict::Ok { mean_estimate, .. } = v {
            assert!(mean_estimate >= 1);
        }
    }

    #[test]
    fn true_n_returned() {
        let v = simulate(500, 5, 100, 42);
        if let TankVerdict::Ok { true_n, .. } = v {
            assert_eq!(true_n, 500);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(10, 2, 100, 42);
        assert!(matches!(v, TankVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(100, 5, 10_000, 42);
        assert!(matches!(v, TankVerdict::Ok { .. }));
    }
}
