//! # Monte-Carlo Token Throughput Simulator
//!
//! Simulate token-stream throughput: requests with variable token
//! counts batched by max_batch and processed at a fixed tokens/sec
//! rate. Returns mean tokens/sec observed.
//!
//! Demonstrates the **MC.10** recipe for PMAT-161 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vLLM batching throughput model.
//!
//! Run with: cargo run --example mc_token_throughput_sim
//!
//! Added by PMAT-161 (catalog 1072→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ThroughputSimVerdict {
    Ok {
        mean_tokens_per_sec: f64,
        total_tokens: u64,
        total_secs: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    num_requests: u32,
    mean_tokens_per_request: u32,
    max_batch_tokens: u32,
    rate_tokens_per_sec: f64,
    seed: u64,
) -> ThroughputSimVerdict {
    if num_requests == 0
        || mean_tokens_per_request == 0
        || max_batch_tokens == 0
        || !rate_tokens_per_sec.is_finite()
        || rate_tokens_per_sec <= 0.0
    {
        return ThroughputSimVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut total_tokens: u64 = 0;
    let mut total_secs: f64 = 0.0;
    let mut batch_tokens: u32 = 0;
    for _ in 0..num_requests {
        // Vary tokens between 50%..150% of mean.
        let request_tokens =
            (f64::from(mean_tokens_per_request) * (0.5 + unit(&mut rng_state))) as u32;
        if batch_tokens + request_tokens > max_batch_tokens {
            total_secs += f64::from(batch_tokens) / rate_tokens_per_sec;
            total_tokens += u64::from(batch_tokens);
            batch_tokens = request_tokens;
        } else {
            batch_tokens += request_tokens;
        }
    }
    if batch_tokens > 0 {
        total_secs += f64::from(batch_tokens) / rate_tokens_per_sec;
        total_tokens += u64::from(batch_tokens);
    }
    let mean_tokens_per_sec = if total_secs > 0.0 {
        total_tokens as f64 / total_secs
    } else {
        0.0
    };
    ThroughputSimVerdict::Ok {
        mean_tokens_per_sec,
        total_tokens,
        total_secs,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_token_throughput_sim")?;

    println!("typical: {:?}", simulate(1000, 100, 4096, 200.0, 42));
    println!("invalid: {:?}", simulate(0, 100, 4096, 200.0, 42));
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
    fn typical_yields_throughput() {
        let v = simulate(1000, 100, 4096, 200.0, 42);
        if let ThroughputSimVerdict::Ok {
            mean_tokens_per_sec,
            ..
        } = v
        {
            // Mean throughput should be near rate (200) since tokens
            // generate at that rate.
            assert!(mean_tokens_per_sec >= 150.0);
            assert!(mean_tokens_per_sec <= 250.0);
        }
    }

    #[test]
    fn zero_requests_invalid() {
        assert_eq!(
            simulate(0, 100, 4096, 200.0, 42),
            ThroughputSimVerdict::InvalidConfig
        );
    }

    #[test]
    fn zero_mean_tokens_invalid() {
        assert_eq!(
            simulate(100, 0, 4096, 200.0, 42),
            ThroughputSimVerdict::InvalidConfig
        );
    }

    #[test]
    fn zero_batch_invalid() {
        assert_eq!(
            simulate(100, 100, 0, 200.0, 42),
            ThroughputSimVerdict::InvalidConfig
        );
    }

    #[test]
    fn zero_rate_invalid() {
        assert_eq!(
            simulate(100, 100, 4096, 0.0, 42),
            ThroughputSimVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_rate_invalid() {
        assert_eq!(
            simulate(100, 100, 4096, f64::NAN, 42),
            ThroughputSimVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = simulate(100, 100, 4096, 200.0, 42);
        let b = simulate(100, 100, 4096, 200.0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn total_tokens_positive() {
        let v = simulate(10, 100, 4096, 200.0, 42);
        if let ThroughputSimVerdict::Ok { total_tokens, .. } = v {
            assert!(total_tokens > 0);
        }
    }

    #[test]
    fn total_secs_positive() {
        let v = simulate(10, 100, 4096, 200.0, 42);
        if let ThroughputSimVerdict::Ok { total_secs, .. } = v {
            assert!(total_secs > 0.0);
        }
    }

    #[test]
    fn higher_rate_higher_throughput() {
        let slow = simulate(100, 100, 4096, 100.0, 42);
        let fast = simulate(100, 100, 4096, 1000.0, 42);
        if let (
            ThroughputSimVerdict::Ok {
                mean_tokens_per_sec: a,
                ..
            },
            ThroughputSimVerdict::Ok {
                mean_tokens_per_sec: b,
                ..
            },
        ) = (slow, fast)
        {
            assert!(b > a);
        }
    }
}
