//! # Monte-Carlo Quorum Read Consistency
//!
//! Sim Dynamo-style quorum reads: N replicas, R reads, W writes.
//! Replicas may be stale with prob `staleness_prob`. Reports
//! fraction of reads returning a non-stale answer (R+W>N invariant).
//!
//! Demonstrates the **MC.85** recipe for PMAT-187 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: DeCandia et al., Dynamo SOSP 2007 §3.3 (R+W>N quorum
//!  protocol).
//!
//! Run with: cargo run --example mc_quorum_read_consistency
//!
//! Added by PMAT-187 (catalog 1306→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum QuorumVerdict {
    Ok {
        consistent_reads: u32,
        stale_reads: u32,
        consistency_rate: f64,
    },
    InvalidConfig,
}

#[allow(clippy::too_many_arguments)]
pub fn simulate(
    reads: u32,
    n_replicas: u32,
    r_quorum: u32,
    staleness_prob: f64,
    seed: u64,
) -> QuorumVerdict {
    if reads == 0
        || n_replicas == 0
        || r_quorum == 0
        || r_quorum > n_replicas
        || !(0.0..=1.0).contains(&staleness_prob)
    {
        return QuorumVerdict::InvalidConfig;
    }
    let mut consistent = 0u32;
    let mut stale = 0u32;
    let mut rng_state = seed | 1;
    for _ in 0..reads {
        // Sample n_replicas independently; pick first r_quorum non-stale.
        let mut fresh_seen = 0u32;
        let mut found = false;
        for _ in 0..n_replicas {
            let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
            if r >= staleness_prob {
                fresh_seen += 1;
                if fresh_seen >= r_quorum {
                    found = true;
                    break;
                }
            }
        }
        if found {
            consistent += 1;
        } else {
            stale += 1;
        }
    }
    QuorumVerdict::Ok {
        consistent_reads: consistent,
        stale_reads: stale,
        consistency_rate: f64::from(consistent) / f64::from(reads),
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_quorum_read_consistency")?;

    println!("3 of 5 fresh: {:?}", simulate(2000, 5, 3, 0.1, 42));
    println!("strong quorum: {:?}", simulate(2000, 5, 5, 0.1, 42));
    println!("invalid: {:?}", simulate(0, 5, 3, 0.1, 42));
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
    fn low_staleness_high_consistency() {
        let v = simulate(2000, 5, 3, 0.05, 42);
        if let QuorumVerdict::Ok {
            consistency_rate, ..
        } = v
        {
            assert!(consistency_rate > 0.95);
        }
    }

    #[test]
    fn high_staleness_low_consistency() {
        let v = simulate(2000, 5, 3, 0.9, 42);
        if let QuorumVerdict::Ok {
            consistency_rate, ..
        } = v
        {
            assert!(consistency_rate < 0.10);
        }
    }

    #[test]
    fn invalid_zero_reads() {
        assert_eq!(simulate(0, 5, 3, 0.1, 42), QuorumVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_replicas() {
        assert_eq!(simulate(100, 0, 3, 0.1, 42), QuorumVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_quorum_gt_replicas() {
        assert_eq!(simulate(100, 3, 5, 0.1, 42), QuorumVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_prob_out_of_range() {
        assert_eq!(simulate(100, 5, 3, 1.5, 42), QuorumVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 5, 3, 0.2, 42);
        let b = simulate(500, 5, 3, 0.2, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn rate_in_unit_range() {
        let v = simulate(500, 5, 3, 0.2, 42);
        if let QuorumVerdict::Ok {
            consistency_rate, ..
        } = v
        {
            assert!((0.0..=1.0).contains(&consistency_rate));
        }
    }

    #[test]
    fn full_quorum_strictest() {
        let lax = simulate(2000, 5, 1, 0.1, 42);
        let strict = simulate(2000, 5, 5, 0.1, 42);
        if let (
            QuorumVerdict::Ok {
                consistency_rate: l,
                ..
            },
            QuorumVerdict::Ok {
                consistency_rate: s,
                ..
            },
        ) = (lax, strict)
        {
            assert!(l >= s);
        }
    }

    #[test]
    fn no_staleness_full_consistency() {
        let v = simulate(1000, 5, 3, 0.0, 42);
        if let QuorumVerdict::Ok {
            consistency_rate, ..
        } = v
        {
            assert!((consistency_rate - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn full_staleness_zero_consistency() {
        let v = simulate(100, 5, 3, 1.0, 42);
        if let QuorumVerdict::Ok {
            consistency_rate, ..
        } = v
        {
            assert_eq!(consistency_rate, 0.0);
        }
    }
}
