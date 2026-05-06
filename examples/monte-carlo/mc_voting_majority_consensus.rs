//! # Monte-Carlo Voting Majority Consensus
//!
//! N voters each vote yes with probability `vote_yes_prob`. Sim
//! many elections; report fraction reaching majority (>n/2 yes).
//!
//! Demonstrates the **MC.80** recipe for PMAT-185 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Condorcet's Jury Theorem (1785); de Caritat, Essai sur
//!  l'application de l'analyse à la probabilité des décisions.
//!
//! Run with: cargo run --example mc_voting_majority_consensus
//!
//! Added by PMAT-185 (catalog 1288→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ConsensusVerdict {
    Ok {
        majority_rate: f64,
        unanimous_rate: f64,
    },
    InvalidConfig,
}

pub fn simulate(elections: u32, voters: u32, vote_yes_prob: f64, seed: u64) -> ConsensusVerdict {
    if elections == 0 || voters == 0 || !(0.0..=1.0).contains(&vote_yes_prob) {
        return ConsensusVerdict::InvalidConfig;
    }
    let threshold = voters / 2 + 1;
    let mut majorities: u32 = 0;
    let mut unanimous: u32 = 0;
    let mut rng_state = seed | 1;
    for _ in 0..elections {
        let mut yes: u32 = 0;
        for _ in 0..voters {
            let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
            if r < vote_yes_prob {
                yes += 1;
            }
        }
        if yes >= threshold {
            majorities += 1;
        }
        if yes == voters {
            unanimous += 1;
        }
    }
    ConsensusVerdict::Ok {
        majority_rate: f64::from(majorities) / f64::from(elections),
        unanimous_rate: f64::from(unanimous) / f64::from(elections),
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_voting_majority_consensus")?;

    println!("60% voters: {:?}", simulate(2000, 11, 0.6, 42));
    println!("split decision: {:?}", simulate(2000, 11, 0.5, 42));
    println!("invalid: {:?}", simulate(0, 11, 0.5, 42));
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
    fn high_vote_prob_high_majority() {
        let v = simulate(2000, 11, 0.9, 42);
        if let ConsensusVerdict::Ok { majority_rate, .. } = v {
            assert!(majority_rate > 0.95);
        }
    }

    #[test]
    fn low_vote_prob_low_majority() {
        let v = simulate(2000, 11, 0.1, 42);
        if let ConsensusVerdict::Ok { majority_rate, .. } = v {
            assert!(majority_rate < 0.05);
        }
    }

    #[test]
    fn split_around_half() {
        let v = simulate(5000, 11, 0.5, 42);
        if let ConsensusVerdict::Ok { majority_rate, .. } = v {
            assert!((majority_rate - 0.5).abs() < 0.15);
        }
    }

    #[test]
    fn invalid_zero_elections() {
        assert_eq!(simulate(0, 11, 0.5, 42), ConsensusVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_voters() {
        assert_eq!(simulate(100, 0, 0.5, 42), ConsensusVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_prob_out_of_range() {
        assert_eq!(simulate(100, 11, 1.5, 42), ConsensusVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 11, 0.5, 42);
        let b = simulate(500, 11, 0.5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn unanimous_rare_under_split() {
        let v = simulate(2000, 11, 0.5, 42);
        if let ConsensusVerdict::Ok { unanimous_rate, .. } = v {
            assert!(unanimous_rate < 0.01);
        }
    }

    #[test]
    fn larger_n_sharper_consensus() {
        // Condorcet's theorem: with p>0.5, majority probability → 1 as N → ∞.
        let small = simulate(2000, 5, 0.6, 42);
        let large = simulate(2000, 51, 0.6, 42);
        if let (
            ConsensusVerdict::Ok {
                majority_rate: s, ..
            },
            ConsensusVerdict::Ok {
                majority_rate: l, ..
            },
        ) = (small, large)
        {
            assert!(l >= s);
        }
    }

    #[test]
    fn always_yes_unanimous_one() {
        let v = simulate(100, 5, 1.0, 42);
        if let ConsensusVerdict::Ok {
            majority_rate,
            unanimous_rate,
        } = v
        {
            assert!((majority_rate - 1.0).abs() < 1e-9);
            assert!((unanimous_rate - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn always_no_majority_zero() {
        let v = simulate(100, 5, 0.0, 42);
        if let ConsensusVerdict::Ok { majority_rate, .. } = v {
            assert_eq!(majority_rate, 0.0);
        }
    }

    #[test]
    fn single_voter_rate_equals_prob() {
        let v = simulate(2000, 1, 0.7, 42);
        if let ConsensusVerdict::Ok { majority_rate, .. } = v {
            assert!((majority_rate - 0.7).abs() < 0.05);
        }
    }
}
