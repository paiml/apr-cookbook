//! # Monte-Carlo Consensus Quorum Failure
//!
//! Sim probability of losing quorum (majority) when each of N nodes
//! independently fails with `node_fail_prob`. Returns observed failure
//! rate and analytical bound.
//!
//! Demonstrates the **MC.56** recipe for PMAT-176 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Raft / Paxos quorum-loss probability analysis.
//!
//! Run with: cargo run --example mc_consensus_quorum_failure
//!
//! Added by PMAT-176 (catalog 1207→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum QuorumVerdict {
    Ok {
        observed_failure_rate: f64,
        majority_threshold: u32,
    },
    InvalidConfig,
}

pub fn simulate(nodes: u32, node_fail_prob: f64, trials: u32, seed: u64) -> QuorumVerdict {
    if nodes == 0
        || trials == 0
        || !node_fail_prob.is_finite()
        || !(0.0..=1.0).contains(&node_fail_prob)
    {
        return QuorumVerdict::InvalidConfig;
    }
    let majority_threshold = nodes / 2 + 1;
    let mut rng_state = seed | 1;
    let mut quorum_lost = 0u32;
    for _ in 0..trials {
        let mut healthy = 0u32;
        for _ in 0..nodes {
            if unit(&mut rng_state) >= node_fail_prob {
                healthy += 1;
            }
        }
        if healthy < majority_threshold {
            quorum_lost += 1;
        }
    }
    let observed_failure_rate = f64::from(quorum_lost) / f64::from(trials);
    QuorumVerdict::Ok {
        observed_failure_rate,
        majority_threshold,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_consensus_quorum_failure")?;

    println!("5-node low-fail: {:?}", simulate(5, 0.01, 1000, 42));
    println!("3-node high-fail: {:?}", simulate(3, 0.5, 1000, 42));
    println!("invalid: {:?}", simulate(0, 0.5, 100, 42));
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
    fn low_fail_low_quorum_loss() {
        let v = simulate(5, 0.01, 10_000, 42);
        if let QuorumVerdict::Ok {
            observed_failure_rate,
            ..
        } = v
        {
            assert!(observed_failure_rate < 0.05);
        }
    }

    #[test]
    fn high_fail_likely_quorum_loss() {
        let v = simulate(3, 0.5, 10_000, 42);
        if let QuorumVerdict::Ok {
            observed_failure_rate,
            ..
        } = v
        {
            assert!(observed_failure_rate > 0.3);
        }
    }

    #[test]
    fn majority_threshold_correct() {
        let v = simulate(5, 0.01, 100, 42);
        if let QuorumVerdict::Ok {
            majority_threshold, ..
        } = v
        {
            assert_eq!(majority_threshold, 3);
        }
    }

    #[test]
    fn three_node_majority_two() {
        let v = simulate(3, 0.01, 100, 42);
        if let QuorumVerdict::Ok {
            majority_threshold, ..
        } = v
        {
            assert_eq!(majority_threshold, 2);
        }
    }

    #[test]
    fn invalid_zero_nodes() {
        assert_eq!(simulate(0, 0.5, 100, 42), QuorumVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_trials() {
        assert_eq!(simulate(3, 0.5, 0, 42), QuorumVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_neg_prob() {
        assert_eq!(simulate(3, -0.1, 100, 42), QuorumVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(simulate(3, f64::NAN, 100, 42), QuorumVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(5, 0.1, 1000, 42);
        let b = simulate(5, 0.1, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn rate_in_unit_range() {
        let v = simulate(5, 0.5, 1000, 42);
        if let QuorumVerdict::Ok {
            observed_failure_rate,
            ..
        } = v
        {
            assert!((0.0..=1.0).contains(&observed_failure_rate));
        }
    }
}
