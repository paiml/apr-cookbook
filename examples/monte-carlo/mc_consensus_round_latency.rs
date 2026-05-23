//! # Monte-Carlo Consensus Round Latency
//!
//! Sim a Raft/Paxos-style consensus run: each round, a leader sends
//! AppendEntries; needs ⌈n/2⌉+1 acks before commit. Each follower has
//! independent failure prob. Reports mean rounds to commit.
//!
//! Demonstrates the **MC.73** recipe for PMAT-183 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Ongaro & Ousterhout, In Search of an Understandable
//!  Consensus Algorithm (Raft, USENIX ATC 2014).
//!
//! Run with: cargo run --example mc_consensus_round_latency
//!
//! Added by PMAT-183 (catalog 1270→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ConsensusVerdict {
    Ok {
        mean_rounds: f64,
        max_rounds: u32,
        commits: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    decisions: u32,
    nodes: u32,
    failure_prob: f64,
    max_rounds_per_decision: u32,
    seed: u64,
) -> ConsensusVerdict {
    if decisions == 0
        || nodes < 3
        || max_rounds_per_decision == 0
        || !(0.0..=1.0).contains(&failure_prob)
    {
        return ConsensusVerdict::InvalidConfig;
    }
    let quorum = nodes / 2 + 1;
    let mut total_rounds: u64 = 0;
    let mut max_rounds: u32 = 0;
    let mut commits: u32 = 0;
    let mut rng_state = seed | 1;
    for _ in 0..decisions {
        let mut rounds: u32 = 0;
        let mut committed = false;
        for _ in 0..max_rounds_per_decision {
            rounds += 1;
            let mut acks: u32 = 1; // leader counts itself
            for _ in 0..(nodes - 1) {
                let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
                if r >= failure_prob {
                    acks += 1;
                }
            }
            if acks >= quorum {
                committed = true;
                break;
            }
        }
        total_rounds += u64::from(rounds);
        if rounds > max_rounds {
            max_rounds = rounds;
        }
        if committed {
            commits += 1;
        }
    }
    ConsensusVerdict::Ok {
        mean_rounds: total_rounds as f64 / f64::from(decisions),
        max_rounds,
        commits,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_consensus_round_latency")?;

    println!("healthy: {:?}", simulate(1000, 5, 0.05, 10, 42));
    println!("flaky: {:?}", simulate(1000, 5, 0.6, 10, 42));
    println!("invalid: {:?}", simulate(0, 5, 0.05, 10, 42));
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
    fn healthy_cluster_one_round() {
        let v = simulate(1000, 5, 0.0, 10, 42);
        if let ConsensusVerdict::Ok {
            mean_rounds,
            commits,
            ..
        } = v
        {
            assert!((mean_rounds - 1.0).abs() < 0.01);
            assert_eq!(commits, 1000);
        }
    }

    #[test]
    fn high_failure_more_rounds() {
        let lo = simulate(1000, 5, 0.05, 10, 42);
        let hi = simulate(1000, 5, 0.6, 10, 42);
        if let (
            ConsensusVerdict::Ok { mean_rounds: l, .. },
            ConsensusVerdict::Ok { mean_rounds: h, .. },
        ) = (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn invalid_zero_decisions() {
        assert_eq!(
            simulate(0, 5, 0.05, 10, 42),
            ConsensusVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_too_few_nodes() {
        assert_eq!(
            simulate(100, 2, 0.05, 10, 42),
            ConsensusVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_max_rounds() {
        assert_eq!(
            simulate(100, 5, 0.05, 0, 42),
            ConsensusVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_prob_out_of_range() {
        assert_eq!(
            simulate(100, 5, 1.5, 10, 42),
            ConsensusVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 5, 0.1, 10, 42);
        let b = simulate(500, 5, 0.1, 10, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn commits_le_decisions() {
        let v = simulate(1000, 5, 0.7, 5, 42);
        if let ConsensusVerdict::Ok { commits, .. } = v {
            assert!(commits <= 1000);
        }
    }

    #[test]
    fn max_rounds_le_max_per_decision() {
        let v = simulate(1000, 5, 0.5, 7, 42);
        if let ConsensusVerdict::Ok { max_rounds, .. } = v {
            assert!(max_rounds <= 7);
        }
    }

    #[test]
    fn larger_cluster_more_resilient() {
        let small = simulate(1000, 3, 0.4, 10, 42);
        let large = simulate(1000, 7, 0.4, 10, 42);
        if let (ConsensusVerdict::Ok { commits: s, .. }, ConsensusVerdict::Ok { commits: l, .. }) =
            (small, large)
        {
            assert!(l >= s);
        }
    }

    #[test]
    fn always_failing_no_commits() {
        let v = simulate(100, 5, 1.0, 10, 42);
        if let ConsensusVerdict::Ok { commits, .. } = v {
            assert_eq!(commits, 0);
        }
    }
}
