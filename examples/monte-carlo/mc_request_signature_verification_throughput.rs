//! # Monte-Carlo Request Signature Verification Throughput
//!
//! Sim N concurrent verifiers checking request signatures. Each
//! verification takes random time; reports throughput (verified/s)
//! given total budget.
//!
//! Demonstrates the **MC.98** recipe for PMAT-191 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: M/M/c queueing model (Kendall 1953); load-balancer
//!  throughput analysis.
//!
//! Run with: cargo run --example mc_request_signature_verification_throughput
//!
//! Added by PMAT-191 (catalog 1342→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum VerifyVerdict {
    Ok {
        verified_count: u32,
        rps: f64,
        avg_latency_us: f64,
    },
    InvalidConfig,
}

pub fn simulate(workers: u32, duration_us: u32, avg_verify_us: u32, seed: u64) -> VerifyVerdict {
    if workers == 0 || duration_us == 0 || avg_verify_us == 0 {
        return VerifyVerdict::InvalidConfig;
    }
    let mut worker_busy_until: Vec<u32> = vec![0; workers as usize];
    let mut verified: u32 = 0;
    let mut total_latency: u64 = 0;
    let mut rng_state = seed | 1;
    let mut now: u32 = 0;
    while now < duration_us {
        // Find first available worker.
        let mut next_free = u32::MAX;
        let mut next_idx = 0usize;
        for (i, t) in worker_busy_until.iter().enumerate() {
            if *t < next_free {
                next_free = *t;
                next_idx = i;
            }
        }
        if next_free > now {
            now = next_free;
        }
        if now >= duration_us {
            break;
        }
        let latency = 1 + ((lcg(&mut rng_state) >> 32) as u32) % (2 * avg_verify_us);
        worker_busy_until[next_idx] = now + latency;
        verified += 1;
        total_latency += u64::from(latency);
    }
    let secs = f64::from(duration_us) / 1_000_000.0;
    let rps = f64::from(verified) / secs;
    let avg_latency_us = if verified == 0 {
        0.0
    } else {
        total_latency as f64 / f64::from(verified)
    };
    VerifyVerdict::Ok {
        verified_count: verified,
        rps,
        avg_latency_us,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_request_signature_verification_throughput")?;

    println!("1 worker: {:?}", simulate(1, 1_000_000, 100, 42));
    println!("8 workers: {:?}", simulate(8, 1_000_000, 100, 42));
    println!("invalid: {:?}", simulate(0, 1_000_000, 100, 42));
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
    fn more_workers_more_throughput() {
        let one = simulate(1, 100_000, 100, 42);
        let eight = simulate(8, 100_000, 100, 42);
        if let (
            VerifyVerdict::Ok {
                verified_count: o, ..
            },
            VerifyVerdict::Ok {
                verified_count: e, ..
            },
        ) = (one, eight)
        {
            assert!(e > o);
        }
    }

    #[test]
    fn invalid_zero_workers() {
        assert_eq!(simulate(0, 100_000, 100, 42), VerifyVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_duration() {
        assert_eq!(simulate(1, 0, 100, 42), VerifyVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_avg_latency() {
        assert_eq!(simulate(1, 100_000, 0, 42), VerifyVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(4, 100_000, 100, 42);
        let b = simulate(4, 100_000, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn longer_latency_lower_throughput() {
        let fast = simulate(4, 100_000, 10, 42);
        let slow = simulate(4, 100_000, 1000, 42);
        if let (
            VerifyVerdict::Ok {
                verified_count: f, ..
            },
            VerifyVerdict::Ok {
                verified_count: s, ..
            },
        ) = (fast, slow)
        {
            assert!(f > s);
        }
    }

    #[test]
    fn verified_le_max_possible() {
        let v = simulate(4, 100_000, 100, 42);
        if let VerifyVerdict::Ok { verified_count, .. } = v {
            // Max ≈ 4 workers * 100_000us / 100us = 4000.
            assert!(verified_count <= 5000);
        }
    }

    #[test]
    fn rps_finite() {
        let v = simulate(4, 100_000, 100, 42);
        if let VerifyVerdict::Ok { rps, .. } = v {
            assert!(rps.is_finite());
        }
    }

    #[test]
    fn avg_latency_in_realistic_range() {
        let v = simulate(4, 100_000, 100, 42);
        if let VerifyVerdict::Ok { avg_latency_us, .. } = v {
            // Latency in [1, 200] → avg in (0, 200].
            assert!(avg_latency_us > 0.0 && avg_latency_us <= 200.0);
        }
    }

    #[test]
    fn small_window_works() {
        let v = simulate(1, 1_000, 100, 42);
        assert!(matches!(v, VerifyVerdict::Ok { .. }));
    }

    #[test]
    fn many_workers_handled() {
        let v = simulate(64, 100_000, 100, 42);
        assert!(matches!(v, VerifyVerdict::Ok { .. }));
    }
}
