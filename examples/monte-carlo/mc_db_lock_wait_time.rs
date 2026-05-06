//! # Monte-Carlo DB Lock Wait Time
//!
//! Sim N concurrent tx contending for a single row lock. Each tx
//! holds the lock for a random duration, then releases. Reports
//! avg + p99 wait time and ratio of tx that waited.
//!
//! Demonstrates the **MC.65** recipe for PMAT-180 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Gray & Reuter, Transaction Processing §7 (1992).
//!
//! Run with: cargo run --example mc_db_lock_wait_time
//!
//! Added by PMAT-180 (catalog 1243→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LockVerdict {
    Ok {
        avg_wait: f64,
        p99_wait: u32,
        wait_ratio: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    transactions: u32,
    avg_hold_ms: u32,
    arrival_interval_ms: u32,
    seed: u64,
) -> LockVerdict {
    if transactions == 0 || avg_hold_ms == 0 || arrival_interval_ms == 0 {
        return LockVerdict::InvalidConfig;
    }
    let mut waits: Vec<u32> = Vec::with_capacity(transactions as usize);
    let mut lock_free_at: u32 = 0;
    let mut now: u32 = 0;
    let mut waited: u32 = 0;
    let mut rng_state = seed | 1;
    for _ in 0..transactions {
        let arrived = now;
        let start = arrived.max(lock_free_at);
        let wait = start - arrived;
        if wait > 0 {
            waited += 1;
        }
        let hold = 1 + ((lcg(&mut rng_state) >> 32) as u32) % (2 * avg_hold_ms);
        lock_free_at = start + hold;
        waits.push(wait);
        let jitter = ((lcg(&mut rng_state) >> 32) as u32) % (2 * arrival_interval_ms);
        now += jitter;
    }
    let total: u64 = waits.iter().map(|w| u64::from(*w)).sum();
    let avg_wait = total as f64 / f64::from(transactions);
    let mut sorted = waits.clone();
    sorted.sort_unstable();
    let p99_idx = (sorted.len() as f64 * 0.99) as usize;
    let p99_wait = sorted[p99_idx.min(sorted.len() - 1)];
    let wait_ratio = f64::from(waited) / f64::from(transactions);
    LockVerdict::Ok {
        avg_wait,
        p99_wait,
        wait_ratio,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_db_lock_wait_time")?;

    println!("low contention: {:?}", simulate(1000, 5, 100, 42));
    println!("high contention: {:?}", simulate(1000, 100, 5, 42));
    println!("invalid: {:?}", simulate(0, 5, 100, 42));
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
    fn low_contention_small_avg_wait() {
        let v = simulate(1000, 5, 100, 42);
        if let LockVerdict::Ok { avg_wait, .. } = v {
            assert!(avg_wait < 5.0);
        }
    }

    #[test]
    fn high_contention_high_avg_wait() {
        let v = simulate(1000, 100, 5, 42);
        if let LockVerdict::Ok { avg_wait, .. } = v {
            assert!(avg_wait > 50.0);
        }
    }

    #[test]
    fn invalid_zero_transactions() {
        assert_eq!(simulate(0, 5, 100, 42), LockVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_hold() {
        assert_eq!(simulate(100, 0, 100, 42), LockVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_arrival() {
        assert_eq!(simulate(100, 5, 0, 42), LockVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 10, 50, 42);
        let b = simulate(500, 10, 50, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn p99_geq_avg() {
        let v = simulate(1000, 50, 20, 42);
        if let LockVerdict::Ok {
            avg_wait, p99_wait, ..
        } = v
        {
            assert!(f64::from(p99_wait) >= avg_wait);
        }
    }

    #[test]
    fn wait_ratio_in_unit_range() {
        let v = simulate(500, 30, 30, 42);
        if let LockVerdict::Ok { wait_ratio, .. } = v {
            assert!((0.0..=1.0).contains(&wait_ratio));
        }
    }

    #[test]
    fn higher_contention_higher_wait_ratio() {
        let lo = simulate(500, 5, 1000, 42);
        let hi = simulate(500, 1000, 5, 42);
        if let (LockVerdict::Ok { wait_ratio: l, .. }, LockVerdict::Ok { wait_ratio: h, .. }) =
            (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn first_tx_never_waits() {
        let v = simulate(1, 5, 100, 42);
        if let LockVerdict::Ok { avg_wait, .. } = v {
            assert_eq!(avg_wait, 0.0);
        }
    }

    #[test]
    fn very_long_arrival_zero_wait() {
        let v = simulate(100, 5, 100_000, 42);
        if let LockVerdict::Ok { wait_ratio, .. } = v {
            assert!(wait_ratio < 0.1);
        }
    }
}
