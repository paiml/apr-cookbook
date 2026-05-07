//! # Monte-Carlo Random Leader Election
//!
//! Sim distributed leader election: N nodes flip coins until exactly
//! one stays "in the running". Returns mean rounds-to-leader and
//! success rate within bounded rounds.
//!
//! Demonstrates the **MC.200** recipe for PMAT-225 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Itai & Rodeh, "Symmetry breaking in distributed
//!  networks" Information & Computation 88(1) (1990); randomized
//!  consensus.
//!
//! Run with: cargo run --example mc_leader_election_random
//!
//! Added by PMAT-225 (catalog 1648→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LeaderVerdict {
    Ok {
        mean_rounds: u32,
        success_pct_x100: u32,
    },
    InvalidConfig,
}

pub fn simulate(n_nodes: u32, max_rounds: u32, trials: u32, seed: u64) -> LeaderVerdict {
    if n_nodes < 2 || max_rounds < 1 || trials < 100 {
        return LeaderVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut total_rounds: u64 = 0;
    let mut successes = 0u32;
    for _ in 0..trials {
        let mut alive = n_nodes;
        let mut rounds = 0u32;
        while alive > 1 && rounds < max_rounds {
            // Each alive node flips a coin; survivors are ones that flipped heads.
            let mut survivors = 0u32;
            for _ in 0..alive {
                if (lcg(&mut state) >> 32) & 1 == 1 {
                    survivors += 1;
                }
            }
            if survivors == 0 {
                // All tails; everyone stays alive (reset).
                survivors = alive;
            }
            alive = survivors;
            rounds += 1;
        }
        if alive == 1 {
            successes += 1;
        }
        total_rounds += rounds as u64;
    }
    LeaderVerdict::Ok {
        mean_rounds: (total_rounds / trials as u64) as u32,
        success_pct_x100: ((successes as f64 / trials as f64) * 10000.0) as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_leader_election_random")?;

    println!("n=10: {:?}", simulate(10, 50, 1000, 42));
    println!("invalid: {:?}", simulate(1, 50, 1000, 42));
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
    fn invalid_single_node() {
        assert_eq!(simulate(1, 50, 1000, 42), LeaderVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_rounds() {
        assert_eq!(simulate(10, 0, 1000, 42), LeaderVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(simulate(10, 50, 50, 42), LeaderVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(10, 50, 500, 42);
        let b = simulate(10, 50, 500, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn small_n_high_success() {
        let v = simulate(2, 50, 1000, 42);
        if let LeaderVerdict::Ok {
            success_pct_x100, ..
        } = v
        {
            assert!(success_pct_x100 > 9000);
        }
    }

    #[test]
    fn larger_n_more_rounds() {
        let small = simulate(4, 50, 500, 42);
        let large = simulate(50, 100, 500, 42);
        if let (
            LeaderVerdict::Ok { mean_rounds: s, .. },
            LeaderVerdict::Ok { mean_rounds: l, .. },
        ) = (small, large)
        {
            assert!(l >= s);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(2, 1, 100, 42);
        assert!(matches!(v, LeaderVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(10, 50, 10_000, 42);
        assert!(matches!(v, LeaderVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_both_valid() {
        let a = simulate(10, 50, 500, 42);
        let b = simulate(10, 50, 500, 999);
        assert!(matches!(a, LeaderVerdict::Ok { .. }));
        assert!(matches!(b, LeaderVerdict::Ok { .. }));
    }

    #[test]
    fn rounds_le_max() {
        let v = simulate(10, 50, 500, 42);
        if let LeaderVerdict::Ok { mean_rounds, .. } = v {
            assert!(mean_rounds <= 50);
        }
    }

    #[test]
    fn success_pct_in_zero_one() {
        let v = simulate(10, 50, 500, 42);
        if let LeaderVerdict::Ok {
            success_pct_x100, ..
        } = v
        {
            assert!(success_pct_x100 <= 10000);
        }
    }
}
