//! # Monte-Carlo Token Streaming Jitter
//!
//! Sim per-token inter-arrival jitter in an LLM stream. Returns
//! observed mean inter-token gap, p99, and rate of gaps over a
//! "stall" threshold.
//!
//! Demonstrates the **MC.42** recipe for PMAT-171 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: streaming SSE inter-token timing.
//!
//! Run with: cargo run --example mc_token_streaming_jitter
//!
//! Added by PMAT-171 (catalog 1162→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum JitterVerdict {
    Ok {
        mean_gap_ms: f64,
        p99_gap_ms: f64,
        stall_rate: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    base_gap_ms: f64,
    jitter_spread_ms: f64,
    stall_threshold_ms: f64,
    n_tokens: u32,
    seed: u64,
) -> JitterVerdict {
    if !base_gap_ms.is_finite()
        || base_gap_ms <= 0.0
        || !jitter_spread_ms.is_finite()
        || jitter_spread_ms < 0.0
        || !stall_threshold_ms.is_finite()
        || stall_threshold_ms <= 0.0
        || n_tokens == 0
    {
        return JitterVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut gaps: Vec<f64> = Vec::with_capacity(n_tokens as usize);
    let mut stalls = 0u32;
    for _ in 0..n_tokens {
        let jitter = (unit(&mut rng_state) - 0.5) * 2.0 * jitter_spread_ms;
        let gap = (base_gap_ms + jitter).max(0.0);
        if gap > stall_threshold_ms {
            stalls += 1;
        }
        gaps.push(gap);
    }
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = gaps.iter().sum::<f64>() / f64::from(n_tokens);
    let p99 = gaps[((n_tokens as f64 * 0.99) as usize).min(n_tokens as usize - 1)];
    let stall_rate = f64::from(stalls) / f64::from(n_tokens);
    JitterVerdict::Ok {
        mean_gap_ms: mean,
        p99_gap_ms: p99,
        stall_rate,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_token_streaming_jitter")?;

    println!("smooth: {:?}", simulate(20.0, 5.0, 100.0, 1000, 42));
    println!("jittery: {:?}", simulate(20.0, 100.0, 100.0, 1000, 42));
    println!("invalid: {:?}", simulate(0.0, 5.0, 100.0, 100, 42));
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
    fn smooth_low_stall() {
        let v = simulate(20.0, 5.0, 100.0, 10_000, 42);
        if let JitterVerdict::Ok { stall_rate, .. } = v {
            assert!(stall_rate < 0.001);
        }
    }

    #[test]
    fn jittery_high_p99() {
        let v = simulate(20.0, 50.0, 100.0, 10_000, 42);
        if let JitterVerdict::Ok { p99_gap_ms, .. } = v {
            assert!(p99_gap_ms > 50.0);
        }
    }

    #[test]
    fn invalid_zero_base() {
        assert_eq!(
            simulate(0.0, 5.0, 100.0, 100, 42),
            JitterVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_neg_jitter() {
        assert_eq!(
            simulate(20.0, -1.0, 100.0, 100, 42),
            JitterVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_stall() {
        assert_eq!(
            simulate(20.0, 5.0, 0.0, 100, 42),
            JitterVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_tokens() {
        assert_eq!(
            simulate(20.0, 5.0, 100.0, 0, 42),
            JitterVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(f64::NAN, 5.0, 100.0, 100, 42),
            JitterVerdict::InvalidConfig
        );
    }

    #[test]
    fn no_jitter_constant_mean() {
        let v = simulate(20.0, 0.0, 100.0, 1000, 42);
        if let JitterVerdict::Ok { mean_gap_ms, .. } = v {
            assert!((mean_gap_ms - 20.0).abs() < 1e-9);
        }
    }

    #[test]
    fn stall_rate_in_unit_range() {
        let v = simulate(20.0, 50.0, 100.0, 1000, 42);
        if let JitterVerdict::Ok { stall_rate, .. } = v {
            assert!((0.0..=1.0).contains(&stall_rate));
        }
    }

    #[test]
    fn deterministic() {
        let a = simulate(20.0, 5.0, 100.0, 1000, 42);
        let b = simulate(20.0, 5.0, 100.0, 1000, 42);
        assert_eq!(a, b);
    }
}
