//! # Recipe: Acceleration Autotuner
//!
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

use apr_cookbook::prelude::*;
use rand::Rng;

// ---------------------------------------------------------------------------
// Tuning space constants
// ---------------------------------------------------------------------------

const TILE_SIZES: [u32; 4] = [16, 32, 64, 128];
const UNROLL_FACTORS: [u32; 4] = [1, 2, 4, 8];
const VEC_WIDTHS: [u32; 3] = [4, 8, 16];

/// Problem dimensions for the matrix multiply kernel (M x K) * (K x N).
const M: u64 = 1024;
const K: u64 = 1024;
const N: u64 = 1024;

/// Simulated hardware parameters.
const BYTES_PER_ELEMENT: f64 = 4.0; // FP32
const PEAK_BANDWIDTH_GB_S: f64 = 50.0; // GB/s memory bandwidth
const PEAK_GFLOPS: f64 = 256.0; // GFLOPS compute throughput

// ---------------------------------------------------------------------------
// Key types
// ---------------------------------------------------------------------------

/// A single kernel configuration to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TuneConfig {
    tile_size: u32,
    unroll_factor: u32,
    vec_width: u32,
}

impl TuneConfig {
    fn label(&self) -> String {
        format!(
            "T{}/U{}/V{}",
            self.tile_size, self.unroll_factor, self.vec_width
        )
    }
}

/// Cost model output for a given configuration.
#[derive(Debug, Clone)]
struct CostEstimate {
    config: TuneConfig,
    memory_traffic_bytes: f64,
    compute_ops: f64,
    estimated_latency_ns: f64,
    gflops: f64,
}

/// Summary of one search strategy run.
#[derive(Debug, Clone)]
struct SearchResult {
    strategy: String,
    best_config: TuneConfig,
    best_gflops: f64,
    configs_tried: usize,
    efficiency: f64,
}

// ---------------------------------------------------------------------------
// Configuration space
// ---------------------------------------------------------------------------

/// Build the full Cartesian product of tuning parameters (48 configs).
fn build_tuning_space() -> Vec<TuneConfig> {
    let mut configs =
        Vec::with_capacity(TILE_SIZES.len() * UNROLL_FACTORS.len() * VEC_WIDTHS.len());
    for &tile in &TILE_SIZES {
        for &unroll in &UNROLL_FACTORS {
            for &vec_w in &VEC_WIDTHS {
                configs.push(TuneConfig {
                    tile_size: tile,
                    unroll_factor: unroll,
                    vec_width: vec_w,
                });
            }
        }
    }
    configs
}

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

/// Estimate performance of a configuration using a roofline-style cost model.
///
/// - Memory traffic = (M*K + K*N + M*N) * bytes_per_element / tile_efficiency
/// - Compute ops    = 2*M*K*N / (unroll_factor * vec_width)
/// - Latency        = max(memory_time, compute_time)
fn estimate_cost(cfg: &TuneConfig) -> CostEstimate {
    // Tile efficiency: larger tiles reuse more data in cache.
    // Model: efficiency = log2(tile_size) / log2(max_tile).
    let tile_efficiency = (f64::from(cfg.tile_size)).log2() / (128.0_f64).log2();

    let raw_traffic = (M * K + K * N + M * N) as f64 * BYTES_PER_ELEMENT;
    let memory_traffic_bytes = raw_traffic / tile_efficiency;

    let raw_ops = 2.0 * M as f64 * K as f64 * N as f64;
    let compute_ops = raw_ops / f64::from(cfg.unroll_factor * cfg.vec_width);

    // Convert bandwidth / flops to nanoseconds.
    let bandwidth_bytes_per_ns = PEAK_BANDWIDTH_GB_S; // GB/s == bytes/ns
    let peak_flops_per_ns = PEAK_GFLOPS; // GFLOPS == ops/ns (in giga-units)

    let memory_time_ns = memory_traffic_bytes / bandwidth_bytes_per_ns;
    let compute_time_ns = compute_ops / peak_flops_per_ns;

    let estimated_latency_ns = memory_time_ns.max(compute_time_ns);

    // GFLOPS achieved = raw_flops / latency_seconds / 1e9.
    // latency_seconds = estimated_latency_ns * 1e-9, so:
    // gflops = raw_ops / (latency_ns * 1e-9) / 1e9 = raw_ops / latency_ns.
    let gflops = raw_ops / estimated_latency_ns;

    CostEstimate {
        config: *cfg,
        memory_traffic_bytes,
        compute_ops,
        estimated_latency_ns,
        gflops,
    }
}

// ---------------------------------------------------------------------------
// Search strategies
// ---------------------------------------------------------------------------

/// Exhaustive search: evaluate every configuration.
fn search_exhaustive(space: &[TuneConfig]) -> Result<SearchResult> {
    let mut best: Option<CostEstimate> = None;

    for cfg in space {
        let est = estimate_cost(cfg);
        let is_better = best.as_ref().map_or(true, |prev| est.gflops > prev.gflops);
        if is_better {
            best = Some(est);
        }
    }

    let best = best.ok_or_else(|| CookbookError::invalid_format("empty tuning space"))?;

    Ok(SearchResult {
        strategy: "Exhaustive".into(),
        best_config: best.config,
        best_gflops: best.gflops,
        configs_tried: space.len(),
        efficiency: 1.0, // by definition: exhaustive finds the optimum
    })
}

/// Random search: sample `n_trials` configs uniformly at random.
fn search_random(
    space: &[TuneConfig],
    rng: &mut impl Rng,
    n_trials: usize,
) -> Result<SearchResult> {
    if space.is_empty() {
        return Err(CookbookError::invalid_format("empty tuning space"));
    }

    let mut best: Option<CostEstimate> = None;

    for _ in 0..n_trials {
        let idx = rng.gen_range(0..space.len());
        let est = estimate_cost(&space[idx]);
        let is_better = best.as_ref().map_or(true, |prev| est.gflops > prev.gflops);
        if is_better {
            best = Some(est);
        }
    }

    let best = best.ok_or_else(|| CookbookError::invalid_format("no trials executed"))?;

    // Compute efficiency relative to the global optimum.
    let optimal = find_optimal_gflops(space);
    let efficiency = if optimal > 0.0 {
        best.gflops / optimal
    } else {
        0.0
    };

    Ok(SearchResult {
        strategy: "Random".into(),
        best_config: best.config,
        best_gflops: best.gflops,
        configs_tried: n_trials,
        efficiency,
    })
}

/// Bayesian-inspired search: use the best-so-far config to bias sampling
/// toward nearby configurations (same tile size or same vec width).
fn search_bayesian(
    space: &[TuneConfig],
    rng: &mut impl Rng,
    n_trials: usize,
) -> Result<SearchResult> {
    if space.is_empty() {
        return Err(CookbookError::invalid_format("empty tuning space"));
    }

    // Start with a random config.
    let first_idx = rng.gen_range(0..space.len());
    let mut best = estimate_cost(&space[first_idx]);
    let mut tried = 1_usize;

    for _ in 1..n_trials {
        // With 60% probability, pick a neighbor (shares tile or vec width).
        let candidate_idx = if rng.gen_bool(0.6) {
            pick_neighbor(space, &best.config, rng)
        } else {
            rng.gen_range(0..space.len())
        };

        let est = estimate_cost(&space[candidate_idx]);
        tried += 1;

        if est.gflops > best.gflops {
            best = est;
        }
    }

    let optimal = find_optimal_gflops(space);
    let efficiency = if optimal > 0.0 {
        best.gflops / optimal
    } else {
        0.0
    };

    Ok(SearchResult {
        strategy: "Bayesian".into(),
        best_config: best.config,
        best_gflops: best.gflops,
        configs_tried: tried,
        efficiency,
    })
}

/// Pick a configuration that shares at least one parameter with `current`.
fn pick_neighbor(space: &[TuneConfig], current: &TuneConfig, rng: &mut impl Rng) -> usize {
    let neighbors: Vec<usize> = space
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            c.tile_size == current.tile_size
                || c.vec_width == current.vec_width
                || c.unroll_factor == current.unroll_factor
        })
        .map(|(i, _)| i)
        .collect();

    if neighbors.is_empty() {
        rng.gen_range(0..space.len())
    } else {
        neighbors[rng.gen_range(0..neighbors.len())]
    }
}

/// Find the best GFLOPS across the entire space (used for efficiency calc).
fn find_optimal_gflops(space: &[TuneConfig]) -> f64 {
    space
        .iter()
        .map(|c| estimate_cost(c).gflops)
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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
