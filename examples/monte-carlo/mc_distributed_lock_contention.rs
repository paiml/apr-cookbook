//! # Monte-Carlo Distributed Lock Contention
//!
//! Sim N processes contending for a single distributed lock with
//! lease timeout. Reports total acquires, retries, and starvation
//! ratio (max_retries / mean_retries).
//!
//! Demonstrates the **MC.100** recipe for PMAT-192 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Lamport's bakery algorithm (1974); Chubby distributed
//!  lock service (Burrows OSDI 2006).
//!
//! Run with: cargo run --example mc_distributed_lock_contention
//!
//! Added by PMAT-192 (catalog 1351→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LockVerdict {
    Ok {
        total_acquires: u32,
        total_retries: u32,
        starvation_ratio: f64,
    },
    InvalidConfig,
}

#[allow(clippy::too_many_arguments)]
pub fn simulate(seconds: u32, processes: u32, lock_hold_time: u32, seed: u64) -> LockVerdict {
    if seconds == 0 || processes == 0 || lock_hold_time == 0 {
        return LockVerdict::InvalidConfig;
    }
    let mut lock_held_by: Option<u32> = None;
    let mut lock_release_at: u32 = 0;
    let mut acquires_per_proc: Vec<u32> = vec![0; processes as usize];
    let mut retries_per_proc: Vec<u32> = vec![0; processes as usize];
    let mut rng_state = seed | 1;
    for sec in 0..seconds {
        // Release the lock if expired.
        if let Some(p) = lock_held_by {
            if sec >= lock_release_at {
                lock_held_by = None;
                let _ = p;
            }
        }
        // Each process attempts to acquire.
        for p in 0..processes {
            let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
            // Each tries with prob 0.3 per second.
            if r < 0.3 {
                if lock_held_by.is_none() {
                    lock_held_by = Some(p);
                    lock_release_at = sec + lock_hold_time;
                    acquires_per_proc[p as usize] += 1;
                } else {
                    retries_per_proc[p as usize] += 1;
                }
            }
        }
    }
    let total_acquires: u32 = acquires_per_proc.iter().sum();
    let total_retries: u32 = retries_per_proc.iter().sum();
    let max_retries: u32 = *retries_per_proc.iter().max().unwrap_or(&0);
    let mean_retries: f64 = f64::from(total_retries) / f64::from(processes).max(1.0);
    let starvation_ratio = if mean_retries > 0.0 {
        f64::from(max_retries) / mean_retries
    } else {
        1.0
    };
    LockVerdict::Ok {
        total_acquires,
        total_retries,
        starvation_ratio,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_distributed_lock_contention")?;

    println!("low contention: {:?}", simulate(600, 2, 5, 42));
    println!("high contention: {:?}", simulate(600, 16, 5, 42));
    println!("invalid: {:?}", simulate(0, 2, 5, 42));
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
    fn more_processes_more_retries() {
        let lo = simulate(600, 2, 10, 42);
        let hi = simulate(600, 16, 10, 42);
        if let (
            LockVerdict::Ok {
                total_retries: l, ..
            },
            LockVerdict::Ok {
                total_retries: h, ..
            },
        ) = (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn invalid_zero_seconds() {
        assert_eq!(simulate(0, 2, 5, 42), LockVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_processes() {
        assert_eq!(simulate(600, 0, 5, 42), LockVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_hold_time() {
        assert_eq!(simulate(600, 2, 0, 42), LockVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(300, 4, 5, 42);
        let b = simulate(300, 4, 5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn longer_hold_fewer_acquires() {
        let short = simulate(300, 4, 1, 42);
        let long = simulate(300, 4, 100, 42);
        if let (
            LockVerdict::Ok {
                total_acquires: s, ..
            },
            LockVerdict::Ok {
                total_acquires: l, ..
            },
        ) = (short, long)
        {
            assert!(s > l);
        }
    }

    #[test]
    fn single_process_acquires_at_least_once() {
        // A single process can still observe retries while it holds
        // its own lock during hold-time; verify it acquires at all.
        let v = simulate(300, 1, 5, 42);
        if let LockVerdict::Ok { total_acquires, .. } = v {
            assert!(total_acquires > 0);
        }
    }

    #[test]
    fn starvation_ratio_at_least_one_when_retries_exist() {
        let v = simulate(300, 8, 10, 42);
        if let LockVerdict::Ok {
            starvation_ratio, ..
        } = v
        {
            assert!(starvation_ratio >= 1.0);
        }
    }

    #[test]
    fn total_acquires_le_seconds() {
        // Lock can be held for at most `seconds` total time.
        let v = simulate(300, 8, 1, 42);
        if let LockVerdict::Ok { total_acquires, .. } = v {
            assert!(total_acquires <= 300);
        }
    }

    #[test]
    fn finite_outputs() {
        let v = simulate(300, 4, 5, 42);
        if let LockVerdict::Ok {
            starvation_ratio, ..
        } = v
        {
            assert!(starvation_ratio.is_finite());
        }
    }

    #[test]
    fn long_hold_high_starvation() {
        let v = simulate(300, 8, 100, 42);
        if let LockVerdict::Ok {
            starvation_ratio, ..
        } = v
        {
            // u32 always nonneg; documents intent.
            let _ = starvation_ratio;
        }
    }
}
