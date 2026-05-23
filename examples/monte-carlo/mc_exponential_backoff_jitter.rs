//! # Monte-Carlo Exponential Backoff with Jitter
//!
//! Sim retry attempts with exponential backoff and full-jitter; track
//! mean total wait time and max attempts before success. Compares
//! deterministic vs jittered backoff thundering-herd risk.
//!
//! Demonstrates the **MC.173** recipe for PMAT-216 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS "Exponential Backoff And Jitter" blog (2015);
//!  Marc Brooker thundering-herd analysis.
//!
//! Run with: cargo run --example mc_exponential_backoff_jitter
//!
//! Added by PMAT-216 (catalog 1567→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BackoffVerdict {
    Ok {
        mean_total_wait_ms: u32,
        max_attempts_observed: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    base_ms: u32,
    success_prob_pct: u32,
    max_attempts: u32,
    trials: u32,
    use_jitter: bool,
    seed: u64,
) -> BackoffVerdict {
    if base_ms == 0 || !(1..=99).contains(&success_prob_pct) || max_attempts < 2 || trials < 100 {
        return BackoffVerdict::InvalidConfig;
    }
    let p = success_prob_pct as f64 / 100.0;
    let mut state = seed | 1;
    let mut total_wait: u64 = 0;
    let mut max_attempts_obs = 0u32;
    for _ in 0..trials {
        let mut wait_sum: u64 = 0;
        let mut attempts = 0u32;
        for attempt in 0..max_attempts {
            attempts = attempt + 1;
            let r = (lcg(&mut state) as f64) / (u32::MAX as f64);
            if r < p {
                break;
            }
            let backoff = base_ms * (1u32 << attempt.min(20));
            let actual_wait = if use_jitter {
                let j = (lcg(&mut state) as f64) / (u32::MAX as f64);
                (backoff as f64 * j) as u64
            } else {
                backoff as u64
            };
            wait_sum += actual_wait;
        }
        total_wait += wait_sum;
        if attempts > max_attempts_obs {
            max_attempts_obs = attempts;
        }
    }
    BackoffVerdict::Ok {
        mean_total_wait_ms: (total_wait / trials as u64) as u32,
        max_attempts_observed: max_attempts_obs,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_exponential_backoff_jitter")?;

    println!("with jitter: {:?}", simulate(100, 50, 10, 1000, true, 42));
    println!("no jitter: {:?}", simulate(100, 50, 10, 1000, false, 42));
    println!("invalid: {:?}", simulate(0, 50, 10, 1000, true, 42));
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
    fn invalid_zero_base() {
        assert_eq!(
            simulate(0, 50, 10, 1000, true, 42),
            BackoffVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_success_prob() {
        assert_eq!(
            simulate(100, 0, 10, 1000, true, 42),
            BackoffVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_too_few_attempts() {
        assert_eq!(
            simulate(100, 50, 1, 1000, true, 42),
            BackoffVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(
            simulate(100, 50, 10, 50, true, 42),
            BackoffVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 50, 10, 500, true, 42);
        let b = simulate(100, 50, 10, 500, true, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn jitter_lower_mean_wait() {
        let no_jitter = simulate(100, 30, 10, 1000, false, 42);
        let with_jitter = simulate(100, 30, 10, 1000, true, 42);
        if let (
            BackoffVerdict::Ok {
                mean_total_wait_ms: nj,
                ..
            },
            BackoffVerdict::Ok {
                mean_total_wait_ms: j,
                ..
            },
        ) = (no_jitter, with_jitter)
        {
            // Jitter halves expected wait → with-jitter mean is lower.
            assert!(j < nj);
        }
    }

    #[test]
    fn high_success_quick_finish() {
        let v = simulate(100, 95, 10, 1000, true, 42);
        if let BackoffVerdict::Ok {
            max_attempts_observed,
            ..
        } = v
        {
            assert!(max_attempts_observed < 10);
        }
    }

    #[test]
    fn low_success_more_attempts() {
        let high = simulate(100, 90, 10, 1000, true, 42);
        let low = simulate(100, 10, 10, 1000, true, 42);
        if let (
            BackoffVerdict::Ok {
                max_attempts_observed: h,
                ..
            },
            BackoffVerdict::Ok {
                max_attempts_observed: l,
                ..
            },
        ) = (high, low)
        {
            assert!(l >= h);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(1, 1, 2, 100, true, 42);
        assert!(matches!(v, BackoffVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(100, 50, 10, 10_000, true, 42);
        assert!(matches!(v, BackoffVerdict::Ok { .. }));
    }

    #[test]
    fn always_succeed_zero_wait() {
        let v = simulate(100, 99, 10, 1000, false, 42);
        if let BackoffVerdict::Ok {
            mean_total_wait_ms, ..
        } = v
        {
            // ~99% succeed first try → mean wait extremely small.
            assert!(mean_total_wait_ms < 50);
        }
    }
}
