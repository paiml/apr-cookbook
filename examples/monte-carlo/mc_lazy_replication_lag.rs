//! # Monte-Carlo Lazy Replication Lag
//!
//! Sim primary/replica lag with jittered sync intervals. Reports
//! mean/p99 lag samples — useful for SLA setting.
//!
//! Demonstrates the **MC.86** recipe for PMAT-187 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PostgreSQL streaming replication docs §27.2; Brewer's
//!  CAP theorem (2000) — eventual consistency tradeoff.
//!
//! Run with: cargo run --example mc_lazy_replication_lag
//!
//! Added by PMAT-187 (catalog 1306→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LagVerdict {
    Ok {
        mean_lag_ms: f64,
        p99_lag_ms: u32,
        max_lag_ms: u32,
    },
    InvalidConfig,
}

pub fn simulate(samples: u32, avg_sync_interval_ms: u32, jitter_pct: u32, seed: u64) -> LagVerdict {
    if samples == 0 || avg_sync_interval_ms == 0 || jitter_pct > 100 {
        return LagVerdict::InvalidConfig;
    }
    let mut lags: Vec<u32> = Vec::with_capacity(samples as usize);
    let mut rng_state = seed | 1;
    let span = avg_sync_interval_ms * jitter_pct / 100;
    let lo = avg_sync_interval_ms.saturating_sub(span);
    let hi_extra = 2 * span;
    for _ in 0..samples {
        let drift = (lcg(&mut rng_state) >> 32) as u32 % (hi_extra.max(1));
        let lag = lo + drift;
        lags.push(lag);
    }
    lags.sort_unstable();
    let total: u64 = lags.iter().map(|l| u64::from(*l)).sum();
    let mean_lag_ms = total as f64 / f64::from(samples);
    let p99_idx = (lags.len() as f64 * 0.99) as usize;
    let p99_lag_ms = lags[p99_idx.min(lags.len() - 1)];
    let max_lag_ms = *lags.last().unwrap_or(&0);
    LagVerdict::Ok {
        mean_lag_ms,
        p99_lag_ms,
        max_lag_ms,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_lazy_replication_lag")?;

    println!("low jitter: {:?}", simulate(2000, 100, 10, 42));
    println!("high jitter: {:?}", simulate(2000, 100, 50, 42));
    println!("invalid: {:?}", simulate(0, 100, 10, 42));
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
    fn mean_near_avg_sync() {
        let v = simulate(5000, 100, 10, 42);
        if let LagVerdict::Ok { mean_lag_ms, .. } = v {
            // Mean ≈ avg_sync_interval since jitter is symmetric.
            assert!((mean_lag_ms - 100.0).abs() < 20.0);
        }
    }

    #[test]
    fn higher_jitter_higher_max() {
        let lo = simulate(2000, 100, 5, 42);
        let hi = simulate(2000, 100, 50, 42);
        if let (LagVerdict::Ok { max_lag_ms: l, .. }, LagVerdict::Ok { max_lag_ms: h, .. }) =
            (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(simulate(0, 100, 10, 42), LagVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_interval() {
        assert_eq!(simulate(100, 0, 10, 42), LagVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_jitter_above_100() {
        assert_eq!(simulate(100, 100, 200, 42), LagVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 100, 10, 42);
        let b = simulate(500, 100, 10, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn p99_le_max() {
        let v = simulate(500, 100, 50, 42);
        if let LagVerdict::Ok {
            p99_lag_ms,
            max_lag_ms,
            ..
        } = v
        {
            assert!(p99_lag_ms <= max_lag_ms);
        }
    }

    #[test]
    fn mean_le_p99() {
        let v = simulate(500, 100, 50, 42);
        if let LagVerdict::Ok {
            mean_lag_ms,
            p99_lag_ms,
            ..
        } = v
        {
            assert!(mean_lag_ms <= f64::from(p99_lag_ms));
        }
    }

    #[test]
    fn zero_jitter_constant_lag() {
        let v = simulate(100, 100, 0, 42);
        if let LagVerdict::Ok {
            mean_lag_ms,
            max_lag_ms,
            ..
        } = v
        {
            assert!((mean_lag_ms - 100.0).abs() < 1e-9);
            assert_eq!(max_lag_ms, 100);
        }
    }

    #[test]
    fn larger_avg_higher_mean() {
        let small = simulate(1000, 50, 10, 42);
        let big = simulate(1000, 500, 10, 42);
        if let (LagVerdict::Ok { mean_lag_ms: s, .. }, LagVerdict::Ok { mean_lag_ms: b, .. }) =
            (small, big)
        {
            assert!(b > s);
        }
    }

    #[test]
    fn single_sample_works() {
        let v = simulate(1, 100, 10, 42);
        assert!(matches!(v, LagVerdict::Ok { .. }));
    }
}
