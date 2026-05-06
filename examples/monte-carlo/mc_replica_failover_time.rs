//! # Monte-Carlo Replica Failover Time
//!
//! Sim time-to-failover when primary fails: detection delay + leader-
//! election time + drain time. Returns mean / p99 failover.
//!
//! Demonstrates the **MC.39** recipe for PMAT-170 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Raft + sentinel-style failover protocols.
//!
//! Run with: cargo run --example mc_replica_failover_time
//!
//! Added by PMAT-170 (catalog 1153→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FailoverVerdict {
    Ok {
        mean_failover_ms: f64,
        p99_failover_ms: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    detection_window_ms: f64,
    election_round_ms: f64,
    drain_ms: f64,
    samples: u32,
    seed: u64,
) -> FailoverVerdict {
    if !detection_window_ms.is_finite()
        || detection_window_ms <= 0.0
        || !election_round_ms.is_finite()
        || election_round_ms <= 0.0
        || !drain_ms.is_finite()
        || drain_ms < 0.0
        || samples == 0
    {
        return FailoverVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut times: Vec<f64> = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        let detection = unit(&mut rng_state) * detection_window_ms;
        // Elections may need multiple rounds.
        let mut election = 0.0;
        loop {
            election += election_round_ms;
            if unit(&mut rng_state) > 0.3 || election > 5.0 * election_round_ms {
                break;
            }
        }
        let total = detection + election + drain_ms;
        times.push(total);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = times.iter().sum::<f64>() / f64::from(samples);
    let p99 = times[((samples as f64 * 0.99) as usize).min(samples as usize - 1)];
    FailoverVerdict::Ok {
        mean_failover_ms: mean,
        p99_failover_ms: p99,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_replica_failover_time")?;

    println!("fast: {:?}", simulate(500.0, 100.0, 50.0, 1000, 42));
    println!("slow: {:?}", simulate(5000.0, 1000.0, 500.0, 1000, 42));
    println!("invalid: {:?}", simulate(-1.0, 100.0, 50.0, 100, 42));
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
    fn p99_above_mean() {
        let v = simulate(500.0, 100.0, 50.0, 1000, 42);
        if let FailoverVerdict::Ok {
            mean_failover_ms,
            p99_failover_ms,
        } = v
        {
            assert!(p99_failover_ms >= mean_failover_ms);
        }
    }

    #[test]
    fn slower_settings_higher_mean() {
        let fast = simulate(500.0, 100.0, 50.0, 1000, 42);
        let slow = simulate(5000.0, 1000.0, 500.0, 1000, 42);
        if let (
            FailoverVerdict::Ok {
                mean_failover_ms: f,
                ..
            },
            FailoverVerdict::Ok {
                mean_failover_ms: s,
                ..
            },
        ) = (fast, slow)
        {
            assert!(s > f);
        }
    }

    #[test]
    fn invalid_zero_detection() {
        assert_eq!(
            simulate(0.0, 100.0, 50.0, 100, 42),
            FailoverVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_election() {
        assert_eq!(
            simulate(500.0, 0.0, 50.0, 100, 42),
            FailoverVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_neg_drain() {
        assert_eq!(
            simulate(500.0, 100.0, -1.0, 100, 42),
            FailoverVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(
            simulate(500.0, 100.0, 50.0, 0, 42),
            FailoverVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(f64::NAN, 100.0, 50.0, 100, 42),
            FailoverVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(500.0, 100.0, 50.0, 1000, 42);
        let b = simulate(500.0, 100.0, 50.0, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn mean_at_least_zero() {
        let v = simulate(500.0, 100.0, 50.0, 100, 42);
        if let FailoverVerdict::Ok {
            mean_failover_ms, ..
        } = v
        {
            assert!(mean_failover_ms >= 0.0);
        }
    }

    #[test]
    fn zero_drain_works() {
        let v = simulate(500.0, 100.0, 0.0, 100, 42);
        assert!(matches!(v, FailoverVerdict::Ok { .. }));
    }
}
