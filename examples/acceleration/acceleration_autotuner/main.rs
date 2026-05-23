#![allow(unused_imports)]
//! # Recipe: Acceleration Autotuner
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/avx512-matmul-v1.yaml
//! **Category**: Acceleration
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] 10 tests
//!
//! ## Learning Objective
//! Search for optimal kernel configurations (tile size, unroll factor,
//! vectorization width) for matrix multiply on a given hardware target
//! using exhaustive, random, and Bayesian-inspired search strategies.
//!
//! ## Run Command
//! ```bash
//! cargo run --example acceleration_autotuner
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr bench model.apr          # APR native format
//! apr bench model.gguf         # GGUF (llama.cpp compatible)
//! apr bench model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Hennessy, J. & Patterson, D. (2017). *Computer Architecture: A Quantitative Approach*. DOI: 10.1016/C2012-0-01712-X

use apr_cookbook::prelude::*;
use rand::Rng;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("acceleration_autotuner")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Auto-tuning matrix multiply kernel configurations");
    println!();

    // Build full tuning space.
    let space = build_tuning_space();
    println!("Tuning space: {} configurations", space.len());
    println!("  Tile sizes:   {:?}", TILE_SIZES);
    println!("  Unroll factors: {:?}", UNROLL_FACTORS);
    println!("  Vec widths:   {:?}", VEC_WIDTHS);
    println!("  Problem: {}x{} * {}x{} (FP32)", M, K, K, N);
    println!();

    // -----------------------------------------------------------------------
    // 1. Exhaustive search
    // -----------------------------------------------------------------------
    let exhaustive = search_exhaustive(&space)?;
    let optimal_gflops = exhaustive.best_gflops;

    // -----------------------------------------------------------------------
    // 2. Random search (10 trials)
    // -----------------------------------------------------------------------
    let random = search_random(&space, ctx.rng(), 10)?;

    // -----------------------------------------------------------------------
    // 3. Bayesian-inspired search (10 trials)
    // -----------------------------------------------------------------------
    let bayesian = search_bayesian(&space, ctx.rng(), 10)?;

    // -----------------------------------------------------------------------
    // Print full results table (all configs ranked by GFLOPS)
    // -----------------------------------------------------------------------
    let mut all_estimates: Vec<CostEstimate> = space.iter().map(estimate_cost).collect();
    all_estimates.sort_by(|a, b| {
        b.gflops
            .partial_cmp(&a.gflops)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let worst_gflops = all_estimates.last().map_or(0.0, |e| e.gflops);

    println!("All Configurations (ranked by estimated GFLOPS):");
    println!("{:-<90}", "");
    println!(
        "{:<4} {:<14} {:>10} {:>14} {:>14} {:>12} {:>10}",
        "Rank", "Config", "GFLOPS", "Compute Ops", "Latency ns", "Mem Traffic", "Speedup"
    );
    println!("{:-<90}", "");

    for (rank, est) in all_estimates.iter().enumerate() {
        let speedup = if worst_gflops > 0.0 {
            est.gflops / worst_gflops
        } else {
            1.0
        };
        println!(
            "{:<4} {:<14} {:>10.2} {:>14.0} {:>14.0} {:>10.0} B {:>8.2}x",
            rank + 1,
            est.config.label(),
            est.gflops,
            est.compute_ops,
            est.estimated_latency_ns,
            est.memory_traffic_bytes,
            speedup,
        );
    }
    println!("{:-<90}", "");
    println!();

    // -----------------------------------------------------------------------
    // Strategy comparison
    // -----------------------------------------------------------------------
    let strategies = [&exhaustive, &random, &bayesian];

    println!("Strategy Comparison:");
    println!("{:-<72}", "");
    println!(
        "{:<12} {:>14} {:>14} {:>12} {:>12}",
        "Strategy", "Best GFLOPS", "Best Config", "Tried", "Efficiency"
    );
    println!("{:-<72}", "");

    for s in &strategies {
        println!(
            "{:<12} {:>12.2} {:>14} {:>12} {:>10.1}%",
            s.strategy,
            s.best_gflops,
            s.best_config.label(),
            s.configs_tried,
            s.efficiency * 100.0,
        );
    }
    println!("{:-<72}", "");
    println!();

    // Record metrics.
    ctx.record_metric("total_configs", space.len() as i64);
    ctx.record_float_metric("optimal_gflops", optimal_gflops);
    ctx.record_float_metric("random_efficiency", random.efficiency);
    ctx.record_float_metric("bayesian_efficiency", bayesian.efficiency);
    ctx.record_string_metric("best_config", exhaustive.best_config.label());

    println!(
        "Optimal config: {} ({:.2} GFLOPS)",
        exhaustive.best_config.label(),
        optimal_gflops,
    );
    println!(
        "Random found:   {} ({:.2} GFLOPS, {:.1}% of optimal)",
        random.best_config.label(),
        random.best_gflops,
        random.efficiency * 100.0,
    );
    println!(
        "Bayesian found: {} ({:.2} GFLOPS, {:.1}% of optimal)",
        bayesian.best_config.label(),
        bayesian.best_gflops,
        bayesian.efficiency * 100.0,
    );
    println!();

    ctx.report()?;
    println!("\n[SUCCESS] Auto-tuner search complete.");

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuning_space_size() {
        let space = build_tuning_space();
        assert_eq!(space.len(), 48);
    }

    #[test]
    fn test_tuning_space_all_unique() {
        let space = build_tuning_space();
        for (i, a) in space.iter().enumerate() {
            for (j, b) in space.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        (a.tile_size, a.unroll_factor, a.vec_width),
                        (b.tile_size, b.unroll_factor, b.vec_width),
                        "Duplicate config at indices {i} and {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_cost_estimate_positive() {
        let cfg = TuneConfig {
            tile_size: 64,
            unroll_factor: 4,
            vec_width: 8,
        };
        let est = estimate_cost(&cfg);
        assert!(est.gflops > 0.0, "GFLOPS must be positive");
        assert!(est.memory_traffic_bytes > 0.0);
        assert!(est.compute_ops > 0.0);
        assert!(est.estimated_latency_ns > 0.0);
    }

    #[test]
    fn test_larger_tile_more_efficient() {
        let small = estimate_cost(&TuneConfig {
            tile_size: 16,
            unroll_factor: 1,
            vec_width: 4,
        });
        let large = estimate_cost(&TuneConfig {
            tile_size: 128,
            unroll_factor: 1,
            vec_width: 4,
        });
        // Larger tile should yield lower memory traffic (better cache reuse).
        assert!(
            large.memory_traffic_bytes < small.memory_traffic_bytes,
            "Larger tile should reduce memory traffic"
        );
    }

    #[test]
    fn test_wider_vec_more_gflops() {
        let narrow = estimate_cost(&TuneConfig {
            tile_size: 64,
            unroll_factor: 1,
            vec_width: 4,
        });
        let wide = estimate_cost(&TuneConfig {
            tile_size: 64,
            unroll_factor: 1,
            vec_width: 16,
        });
        // Wider vectorization should reduce compute cost, potentially more GFLOPS.
        assert!(
            wide.compute_ops < narrow.compute_ops,
            "Wider vector should reduce compute ops"
        );
    }

    #[test]
    fn test_exhaustive_finds_optimal() {
        let space = build_tuning_space();
        let result = search_exhaustive(&space).expect("exhaustive search failed");
        assert_eq!(result.configs_tried, 48);
        assert!((result.efficiency - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_random_search_bounded() {
        let space = build_tuning_space();
        let mut ctx = RecipeContext::new("test_random_search").expect("context failed");
        let result = search_random(&space, ctx.rng(), 10).expect("random search failed");
        assert_eq!(result.configs_tried, 10);
        assert!(result.efficiency > 0.0);
        assert!(result.efficiency <= 1.0);
    }

    #[test]
    fn test_bayesian_search_bounded() {
        let space = build_tuning_space();
        let mut ctx = RecipeContext::new("test_bayesian_search").expect("context failed");
        let result = search_bayesian(&space, ctx.rng(), 10).expect("bayesian search failed");
        assert_eq!(result.configs_tried, 10);
        assert!(result.efficiency > 0.0);
        assert!(result.efficiency <= 1.0);
    }

    #[test]
    fn test_config_label_format() {
        let cfg = TuneConfig {
            tile_size: 32,
            unroll_factor: 2,
            vec_width: 8,
        };
        assert_eq!(cfg.label(), "T32/U2/V8");
    }

    #[test]
    fn test_empty_space_errors() {
        let empty: Vec<TuneConfig> = vec![];
        assert!(search_exhaustive(&empty).is_err());

        let mut ctx = RecipeContext::new("test_empty_space").expect("context failed");
        assert!(search_random(&empty, ctx.rng(), 5).is_err());
        assert!(search_bayesian(&empty, ctx.rng(), 5).is_err());
    }
}
