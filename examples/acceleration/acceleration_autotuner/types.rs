#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use rand::Rng;

// ---------------------------------------------------------------------------
// Tuning space constants
// ---------------------------------------------------------------------------

pub const TILE_SIZES: [u32; 4] = [16, 32, 64, 128];
pub const UNROLL_FACTORS: [u32; 4] = [1, 2, 4, 8];
pub const VEC_WIDTHS: [u32; 3] = [4, 8, 16];

/// Problem dimensions for the matrix multiply kernel (M x K) * (K x N).
pub const M: u64 = 1024;
pub const K: u64 = 1024;
pub const N: u64 = 1024;

/// Simulated hardware parameters.
pub const BYTES_PER_ELEMENT: f64 = 4.0; // FP32
pub const PEAK_BANDWIDTH_GB_S: f64 = 50.0; // GB/s memory bandwidth
pub const PEAK_GFLOPS: f64 = 256.0; // GFLOPS compute throughput

// ---------------------------------------------------------------------------
// Key types
// ---------------------------------------------------------------------------

/// A single kernel configuration to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TuneConfig {
    pub tile_size: u32,
    pub unroll_factor: u32,
    pub vec_width: u32,
}

impl TuneConfig {
    pub fn label(&self) -> String {
        format!(
            "T{}/U{}/V{}",
            self.tile_size, self.unroll_factor, self.vec_width
        )
    }
}

/// Cost model output for a given configuration.
#[derive(Debug, Clone)]
pub struct CostEstimate {
    pub config: TuneConfig,
    pub memory_traffic_bytes: f64,
    pub compute_ops: f64,
    pub estimated_latency_ns: f64,
    pub gflops: f64,
}

/// Summary of one search strategy run.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub strategy: String,
    pub best_config: TuneConfig,
    pub best_gflops: f64,
    pub configs_tried: usize,
    pub efficiency: f64,
}

// ---------------------------------------------------------------------------
// Configuration space
// ---------------------------------------------------------------------------

/// Build the full Cartesian product of tuning parameters (48 configs).
pub fn build_tuning_space() -> Vec<TuneConfig> {
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

// Estimate performance of a configuration using a roofline-style cost model.
//
// - Memory traffic = (M*K + K*N + M*N) * bytes_per_element / tile_efficiency
// - Compute ops    = 2*M*K*N / (unroll_factor * vec_width)
/// - Latency        = max(memory_time, compute_time)
pub fn estimate_cost(cfg: &TuneConfig) -> CostEstimate {
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
pub fn search_exhaustive(space: &[TuneConfig]) -> Result<SearchResult> {
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
pub fn search_random(
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

// Bayesian-inspired search: use the best-so-far config to bias sampling
/// toward nearby configurations (same tile size or same vec width).
pub fn search_bayesian(
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
pub fn pick_neighbor(space: &[TuneConfig], current: &TuneConfig, rng: &mut impl Rng) -> usize {
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
pub fn find_optimal_gflops(space: &[TuneConfig]) -> f64 {
    space
        .iter()
        .map(|c| estimate_cost(c).gflops)
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
