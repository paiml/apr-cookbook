//! # Monte-Carlo Zipf's Law Word Frequency
//!
//! Sample words from a Zipf-distributed corpus (rank-frequency
//! ∝ 1/k^s) and verify that observed frequencies follow Zipf's law.
//! Returns the top word's count and the rank-2 ratio.
//!
//! Demonstrates the **MC.178** recipe for PMAT-218 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Zipf, "Human Behavior and the Principle of Least Effort"
//!  (1949); Mandelbrot generalized law (1953).
//!
//! Run with: cargo run --example mc_zipf_law_word_freq
//!
//! Added by PMAT-218 (catalog 1585→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ZipfVerdict {
    Ok {
        top_count: u32,
        rank_2_ratio_x100: u32,
    },
    InvalidConfig,
}

pub fn simulate(vocab_size: u32, samples: u32, exponent_x100: u32, seed: u64) -> ZipfVerdict {
    if vocab_size < 5 || samples < 1000 || !(50..=300).contains(&exponent_x100) {
        return ZipfVerdict::InvalidConfig;
    }
    let s = exponent_x100 as f64 / 100.0;
    let mut state = seed | 1;
    // Pre-compute CDF
    let weights: Vec<f64> = (1..=vocab_size).map(|k| 1.0 / (k as f64).powf(s)).collect();
    let total: f64 = weights.iter().sum();
    let mut cdf: Vec<f64> = Vec::with_capacity(vocab_size as usize);
    let mut acc = 0.0;
    for w in &weights {
        acc += w / total;
        cdf.push(acc);
    }
    let mut counts: Vec<u32> = vec![0; vocab_size as usize];
    for _ in 0..samples {
        let u = (lcg(&mut state) as f64) / (u32::MAX as f64);
        let idx = cdf.iter().position(|c| u < *c).unwrap_or(0);
        counts[idx] += 1;
    }
    let top = counts[0];
    let rank_2 = counts[1].max(1);
    let ratio = (top as f64 / rank_2 as f64 * 100.0) as u32;
    ZipfVerdict::Ok {
        top_count: top,
        rank_2_ratio_x100: ratio,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_zipf_law_word_freq")?;

    println!("s=1.0: {:?}", simulate(20, 10_000, 100, 42));
    println!("invalid: {:?}", simulate(2, 100, 100, 42));
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
    fn invalid_too_small_vocab() {
        assert_eq!(simulate(2, 1000, 100, 42), ZipfVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_samples() {
        assert_eq!(simulate(10, 100, 100, 42), ZipfVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_exponent_too_low() {
        assert_eq!(simulate(10, 1000, 49, 42), ZipfVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_exponent_too_high() {
        assert_eq!(simulate(10, 1000, 301, 42), ZipfVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(10, 1000, 100, 42);
        let b = simulate(10, 1000, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn rank_1_dominates() {
        // For s=1.0, rank-1 / rank-2 ≈ 2 → ratio ≈ 200.
        let v = simulate(20, 50_000, 100, 42);
        if let ZipfVerdict::Ok {
            rank_2_ratio_x100, ..
        } = v
        {
            // Allow wide band for finite-sample variance.
            assert!((150..=300).contains(&rank_2_ratio_x100));
        }
    }

    #[test]
    fn higher_exponent_more_skew() {
        let s_low = simulate(20, 50_000, 80, 42);
        let s_high = simulate(20, 50_000, 200, 42);
        if let (
            ZipfVerdict::Ok {
                rank_2_ratio_x100: l,
                ..
            },
            ZipfVerdict::Ok {
                rank_2_ratio_x100: h,
                ..
            },
        ) = (s_low, s_high)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn top_count_at_least_one() {
        let v = simulate(10, 1000, 100, 42);
        if let ZipfVerdict::Ok { top_count, .. } = v {
            assert!(top_count >= 1);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(5, 1000, 50, 42);
        assert!(matches!(v, ZipfVerdict::Ok { .. }));
    }

    #[test]
    fn many_samples_handled() {
        let v = simulate(20, 100_000, 100, 42);
        assert!(matches!(v, ZipfVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(10, 1000, 100, 42);
        let b = simulate(10, 1000, 100, 999);
        assert!(a != b);
    }

    #[test]
    fn top_count_le_samples() {
        let v = simulate(10, 1000, 100, 42);
        if let ZipfVerdict::Ok { top_count, .. } = v {
            assert!(top_count <= 1000);
        }
    }
}
