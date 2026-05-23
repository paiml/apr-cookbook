//! # Monte-Carlo Token Compression Ratio
//!
//! Sim compression ratio of a tokenized stream as a function of
//! Shannon entropy. Returns observed mean ratio + 95th percentile.
//! Lower entropy → better compression.
//!
//! Demonstrates the **MC.55** recipe for PMAT-176 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Shannon source-coding theorem.
//!
//! Run with: cargo run --example mc_token_compress_ratio
//!
//! Added by PMAT-176 (catalog 1207→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CompressVerdict {
    Ok { mean_ratio: f64, p95_ratio: f64 },
    InvalidConfig,
}

pub fn simulate(
    entropy_bits_per_token: f64,
    block_size_tokens: u32,
    num_blocks: u32,
    seed: u64,
) -> CompressVerdict {
    if !entropy_bits_per_token.is_finite()
        || entropy_bits_per_token <= 0.0
        || entropy_bits_per_token > 16.0
        || block_size_tokens == 0
        || num_blocks == 0
    {
        return CompressVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut ratios: Vec<f64> = Vec::with_capacity(num_blocks as usize);
    let baseline_bits_per_token = 16.0;
    for _ in 0..num_blocks {
        // Add jitter ±20% to entropy.
        let jitter = (unit(&mut rng_state) - 0.5) * 0.4 * entropy_bits_per_token;
        let observed_entropy =
            (entropy_bits_per_token + jitter).clamp(0.1, baseline_bits_per_token);
        let ratio = observed_entropy / baseline_bits_per_token;
        ratios.push(ratio);
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean_ratio = ratios.iter().sum::<f64>() / f64::from(num_blocks);
    let p95_ratio = ratios[((num_blocks as f64) * 0.95) as usize];
    let _ = block_size_tokens; // Block size kept for API symmetry.
    CompressVerdict::Ok {
        mean_ratio,
        p95_ratio,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_token_compress_ratio")?;

    println!("low entropy: {:?}", simulate(2.0, 256, 1000, 42));
    println!("high entropy: {:?}", simulate(14.0, 256, 1000, 42));
    println!("invalid: {:?}", simulate(0.0, 256, 1000, 42));
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
    fn low_entropy_better_ratio() {
        let lo = simulate(2.0, 256, 1000, 42);
        let hi = simulate(14.0, 256, 1000, 42);
        if let (
            CompressVerdict::Ok { mean_ratio: l, .. },
            CompressVerdict::Ok { mean_ratio: h, .. },
        ) = (lo, hi)
        {
            assert!(l < h);
        }
    }

    #[test]
    fn p95_above_mean() {
        let v = simulate(8.0, 256, 1000, 42);
        if let CompressVerdict::Ok {
            mean_ratio,
            p95_ratio,
        } = v
        {
            assert!(p95_ratio >= mean_ratio);
        }
    }

    #[test]
    fn invalid_zero_entropy() {
        assert_eq!(simulate(0.0, 256, 1000, 42), CompressVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_high_entropy() {
        assert_eq!(
            simulate(20.0, 256, 1000, 42),
            CompressVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_block_size() {
        assert_eq!(simulate(8.0, 0, 1000, 42), CompressVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_blocks() {
        assert_eq!(simulate(8.0, 256, 0, 42), CompressVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(f64::NAN, 256, 1000, 42),
            CompressVerdict::InvalidConfig
        );
    }

    #[test]
    fn ratio_in_unit_range() {
        let v = simulate(8.0, 256, 1000, 42);
        if let CompressVerdict::Ok { mean_ratio, .. } = v {
            assert!((0.0..=1.0).contains(&mean_ratio));
        }
    }

    #[test]
    fn deterministic() {
        let a = simulate(8.0, 256, 100, 42);
        let b = simulate(8.0, 256, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn min_entropy_low_ratio() {
        let v = simulate(0.5, 256, 1000, 42);
        if let CompressVerdict::Ok { mean_ratio, .. } = v {
            assert!(mean_ratio < 0.1);
        }
    }
}
