//! # Monte-Carlo Metropolis-Hastings on a Discrete Target
//!
//! MCMC sampling from a discrete target distribution (probabilities
//! proportional to weights). Symmetric random-walk proposal on
//! integer state space. Returns visit counts vs target frequencies.
//!
//! Demonstrates the **MC.137** recipe for PMAT-204 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Metropolis, Rosenbluth et al., "Equation of State
//!  Calculations by Fast Computing Machines" J. Chem. Phys. 21(6)
//!  (1953); Hastings, Biometrika 57(1):97-109 (1970).
//!
//! Run with: cargo run --example mc_metropolis_hastings_target
//!
//! Added by PMAT-204 (catalog 1459→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MhVerdict {
    Ok {
        visit_counts: Vec<u32>,
        accept_rate: f64,
    },
    InvalidConfig,
}

pub fn simulate(target_weights: &[u32], iterations: u32, seed: u64) -> MhVerdict {
    if target_weights.is_empty() || iterations < 100 || target_weights.iter().sum::<u32>() == 0 {
        return MhVerdict::InvalidConfig;
    }
    let n = target_weights.len();
    let mut state = seed | 1;
    let mut current = 0usize;
    let mut counts: Vec<u32> = vec![0; n];
    let mut accepts = 0u32;
    for _ in 0..iterations {
        // Symmetric proposal: ±1 step (with wrap)
        let step = if (lcg(&mut state) >> 32) % 2 == 0 {
            1
        } else {
            n - 1
        };
        let proposed = (current + step) % n;
        let cur_w = target_weights[current] as f64;
        let prop_w = target_weights[proposed] as f64;
        // Acceptance probability = min(1, π(prop)/π(cur))
        let alpha = if cur_w == 0.0 {
            1.0
        } else {
            (prop_w / cur_w).min(1.0)
        };
        let r = (lcg(&mut state) >> 32) as f64 / (u32::MAX as f64);
        if r < alpha {
            current = proposed;
            accepts += 1;
        }
        counts[current] += 1;
    }
    MhVerdict::Ok {
        visit_counts: counts,
        accept_rate: accepts as f64 / iterations as f64,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_metropolis_hastings_target")?;

    let weights = [1u32, 4, 9, 4, 1];
    println!("target: {:?}", simulate(&weights, 10_000, 42));
    println!("invalid: {:?}", simulate(&[], 10_000, 42));
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
    fn invalid_empty_weights() {
        assert_eq!(simulate(&[], 10_000, 42), MhVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_iterations() {
        assert_eq!(simulate(&[1, 1], 50, 42), MhVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_all_zero_weights() {
        assert_eq!(simulate(&[0, 0, 0], 10_000, 42), MhVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(&[1, 2, 1], 1000, 42);
        let b = simulate(&[1, 2, 1], 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn visit_counts_sum_to_iterations() {
        let v = simulate(&[1, 2, 1], 1000, 42);
        if let MhVerdict::Ok { visit_counts, .. } = v {
            let total: u32 = visit_counts.iter().sum();
            assert_eq!(total, 1000);
        }
    }

    #[test]
    fn higher_weight_more_visits() {
        let v = simulate(&[1, 100, 1], 100_000, 42);
        if let MhVerdict::Ok { visit_counts, .. } = v {
            // Center bin (weight 100) should dominate.
            assert!(visit_counts[1] > visit_counts[0]);
            assert!(visit_counts[1] > visit_counts[2]);
        }
    }

    #[test]
    fn accept_rate_in_zero_one() {
        let v = simulate(&[1, 1, 1], 1000, 42);
        if let MhVerdict::Ok { accept_rate, .. } = v {
            assert!((0.0..=1.0).contains(&accept_rate));
        }
    }

    #[test]
    fn uniform_target_balanced_visits() {
        let v = simulate(&[1, 1, 1], 100_000, 42);
        if let MhVerdict::Ok { visit_counts, .. } = v {
            // All counts within 5% of 33,333.
            for c in &visit_counts {
                let f = *c as f64 / 100_000.0;
                assert!((0.28..=0.38).contains(&f));
            }
        }
    }

    #[test]
    fn single_state_handled() {
        let v = simulate(&[5], 1000, 42);
        if let MhVerdict::Ok { visit_counts, .. } = v {
            assert_eq!(visit_counts, vec![1000]);
        }
    }

    #[test]
    fn visit_count_length_matches_weights() {
        let v = simulate(&[1, 2, 3, 4, 5], 1000, 42);
        if let MhVerdict::Ok { visit_counts, .. } = v {
            assert_eq!(visit_counts.len(), 5);
        }
    }

    #[test]
    fn high_iteration_count_handled() {
        let v = simulate(&[1, 2, 1], 100_000, 42);
        assert!(matches!(v, MhVerdict::Ok { .. }));
    }
}
