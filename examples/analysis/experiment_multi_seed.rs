//! # Recipe: Experiment — Multi-Seed with Confidence Intervals
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr experiment --seeds 0..10 --config variant_a --ci 0.95`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example experiment_multi_seed` exits 0
//! 2. [x] `cargo test --example experiment_multi_seed` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr experiment --seeds` in-process (no shell-out)
//! 10. [x] Unit tests cover mean, stddev, 95% CI bounds
//!
//! ## Learning Objective
//! Demonstrates a multi-seed experiment: runs N seeds, computes per-seed scores,
//! and reports mean ± 95% confidence interval using the normal approximation
//! (t-table value 1.96 for large N). Highlights why single-seed RL/training
//! results are unreliable.
//!
//! ## Run Command
//! ```bash
//! cargo run --example experiment_multi_seed
//! ```
//!
//! ## References
//! - Henderson, P. et al. (2018). *Deep Reinforcement Learning that Matters*. AAAI. arXiv:1709.06560

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::json;

#[derive(Debug, Clone, Copy)]
struct SeedResult {
    seed: u64,
    score: f64,
}

#[derive(Debug, Clone, Copy)]
struct CiSummary {
    mean: f64,
    stddev: f64,
    ci_half_width: f64,
    lower: f64,
    upper: f64,
    n: usize,
}

fn simulate_training(seed: u64, n_steps: usize) -> f64 {
    // Deterministic per-seed training simulation: mean reward with noise.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cum = 0.0_f64;
    for _ in 0..n_steps {
        cum += rng.gen_range(-1.0..1.0) + 0.3;
    }
    cum / n_steps as f64
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Sample standard deviation (Bessel-corrected for n>=2; 0 otherwise).
fn stddev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let var: f64 = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    var.sqrt()
}

/// Normal-approximation 95% confidence interval (z=1.96).
fn ci_95(xs: &[f64]) -> CiSummary {
    let m = mean(xs);
    let s = stddev(xs);
    let n = xs.len();
    let half = if n == 0 {
        0.0
    } else {
        1.96 * s / (n as f64).sqrt()
    };
    CiSummary {
        mean: m,
        stddev: s,
        ci_half_width: half,
        lower: m - half,
        upper: m + half,
        n,
    }
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("experiment_multi_seed")?;
    println!("=== Recipe: {} ===", ctx.name());

    let seeds: Vec<u64> = (100..120).collect();
    let n_steps = 500;

    let results: Vec<SeedResult> = seeds
        .iter()
        .map(|&s| SeedResult {
            seed: s,
            score: simulate_training(s, n_steps),
        })
        .collect();

    println!("\n--- Per-seed scores ---");
    for r in &results {
        println!("  seed={:>4} score={:.4}", r.seed, r.score);
    }

    let scores: Vec<f64> = results.iter().map(|r| r.score).collect();
    let summary = ci_95(&scores);

    println!("\n--- 95% CI summary ---");
    println!("n       = {}", summary.n);
    println!("mean    = {:.4}", summary.mean);
    println!("stddev  = {:.4}", summary.stddev);
    println!(
        "CI 95%  = [{:.4}, {:.4}] (±{:.4})",
        summary.lower, summary.upper, summary.ci_half_width
    );

    let report = json!({
        "recipe": ctx.name(),
        "n_steps": n_steps,
        "results": results.iter().map(|r| json!({
            "seed": r.seed,
            "score": r.score,
        })).collect::<Vec<_>>(),
        "summary": {
            "n": summary.n,
            "mean": summary.mean,
            "stddev": summary.stddev,
            "ci_half_width": summary.ci_half_width,
            "lower": summary.lower,
            "upper": summary.upper,
        },
    });
    let out = ctx.path("multi-seed.json");
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out, bytes)?;

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_empty_zero() {
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn stddev_of_constant_is_zero() {
        assert_eq!(stddev(&[3.0, 3.0, 3.0]), 0.0);
    }

    #[test]
    fn stddev_small_sample() {
        let v = stddev(&[1.0, 3.0]);
        assert!((v - (2.0_f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn ci_95_contains_mean() {
        let s = ci_95(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(s.lower <= s.mean);
        assert!(s.mean <= s.upper);
    }

    #[test]
    fn ci_95_width_shrinks_with_more_samples() {
        // Same distribution, more samples => tighter CI.
        let small: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let large: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();
        let s1 = ci_95(&small);
        let s2 = ci_95(&large);
        assert!(s2.ci_half_width < s1.ci_half_width);
    }

    #[test]
    fn simulate_training_is_deterministic() {
        let a = simulate_training(42, 100);
        let b = simulate_training(42, 100);
        assert!((a - b).abs() < 1e-12);
    }
}
