//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

// ============================================================================
// Constants
// ============================================================================

/// Number of inference iterations to simulate.
pub const NUM_ITERATIONS: usize = 10;

/// Tokens produced per iteration (simulated generation).
pub const TOKENS_PER_ITERATION: usize = 128;

/// Total time budget per iteration in milliseconds.
pub const BUDGET_MS: f64 = 100.0;

/// Brick names representing transformer layers.
pub const BRICK_NAMES: [&str; 4] = ["attention", "ffn", "normalization", "embedding"];

/// Base latency in milliseconds for each brick (attention, ffn, norm, embed).
pub const BRICK_BASE_MS: [f64; 4] = [35.0, 25.0, 5.0, 10.0];

/// Jitter coefficient (fraction of base latency).
pub const JITTER_FRACTION: f64 = 0.15;

// ============================================================================
// Domain Types
// ============================================================================

/// Timing data for a single brick within one iteration.
#[derive(Debug, Clone)]
pub struct BrickTiming {
    pub name: String,
    pub elapsed_ms: f64,
}

/// Aggregate score for a single brick across all iterations.
#[derive(Debug, Clone)]
pub struct BrickScore {
    pub name: String,
    pub mean_ms: f64,
    pub p95_ms: f64,
    pub pct_budget: f64,
}

/// Detected hardware information.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub cpu: String,
    pub memory_gb: f64,
    pub gpu: Option<String>,
}

/// Per-iteration statistics.
#[derive(Debug, Clone)]
pub struct IterationStats {
    pub iteration: usize,
    pub total_ms: f64,
    pub tokens: usize,
}

/// Complete headless report.
#[derive(Debug, Clone)]
pub struct HeadlessReport {
    pub model_name: String,
    pub iterations: usize,
    pub throughput_tok_s: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub bricks: Vec<BrickScore>,
    pub hardware: HardwareInfo,
}

// ============================================================================
// Percentile Computation
// ============================================================================

// Compute the p-th percentile (0-100) from a sorted slice of f64 values.
//
// Uses linear interpolation between adjacent ranks.
/// Returns 0.0 for empty input.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;

    let lower_val = sorted[lower.min(sorted.len() - 1)];
    let upper_val = sorted[upper.min(sorted.len() - 1)];
    lower_val + frac * (upper_val - lower_val)
}

/// Sort a slice and return a new sorted vector.
pub fn sorted_copy(values: &[f64]) -> Vec<f64> {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

// ============================================================================
// Simulation Helpers
// ============================================================================

// Simulate brick timings for a single inference iteration.
//
/// Each brick gets its base latency plus deterministic jitter from the RNG.
pub fn simulate_iteration(rng: &mut impl Rng, iteration: usize) -> Vec<BrickTiming> {
    BRICK_NAMES
        .iter()
        .zip(BRICK_BASE_MS.iter())
        .map(|(&name, &base_ms)| {
            let jitter = rng.gen_range(-JITTER_FRACTION..JITTER_FRACTION) * base_ms;
            // Slight degradation over iterations (cache pressure simulation)
            let degradation = 1.0 + (iteration as f64 * 0.005);
            let elapsed_ms = (base_ms * degradation + jitter).max(0.1);
            BrickTiming {
                name: name.to_string(),
                elapsed_ms,
            }
        })
        .collect()
}

/// Build hardware info from deterministic values (no actual detection).
pub fn build_hardware_info() -> HardwareInfo {
    HardwareInfo {
        cpu: "AMD EPYC 7763 64-Core".to_string(),
        memory_gb: 256.0,
        gpu: Some("NVIDIA A100 80GB".to_string()),
    }
}

// ============================================================================
// Aggregation
// ============================================================================

/// Compute brick scores from collected timings across all iterations.
pub fn compute_brick_scores(all_timings: &[Vec<BrickTiming>]) -> Vec<BrickScore> {
    BRICK_NAMES
        .iter()
        .map(|&name| {
            let values: Vec<f64> = all_timings
                .iter()
                .flat_map(|iter_timings| {
                    iter_timings
                        .iter()
                        .filter(|t| t.name == name)
                        .map(|t| t.elapsed_ms)
                })
                .collect();

            let mean_ms = compute_mean(&values);
            let sorted = sorted_copy(&values);
            let p95_ms = percentile(&sorted, 95.0);
            let pct_budget = (mean_ms / BUDGET_MS) * 100.0;

            BrickScore {
                name: name.to_string(),
                mean_ms,
                p95_ms,
                pct_budget,
            }
        })
        .collect()
}

/// Compute the arithmetic mean of a slice.
pub fn compute_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values.iter().sum();
    sum / values.len() as f64
}

/// Compute throughput in tokens per second from iteration stats.
pub fn compute_throughput(stats: &[IterationStats]) -> f64 {
    if stats.is_empty() {
        return 0.0;
    }
    let total_tokens: usize = stats.iter().map(|s| s.tokens).sum();
    let total_time_s: f64 = stats.iter().map(|s| s.total_ms).sum::<f64>() / 1000.0;
    if total_time_s <= 0.0 {
        return 0.0;
    }
    total_tokens as f64 / total_time_s
}

/// Build the final headless report from all collected data.
pub fn build_report(
    stats: &[IterationStats],
    brick_scores: Vec<BrickScore>,
    hardware: HardwareInfo,
) -> HeadlessReport {
    let latencies: Vec<f64> = stats.iter().map(|s| s.total_ms).collect();
    let sorted_lat = sorted_copy(&latencies);

    HeadlessReport {
        model_name: "llama-7b.apr".to_string(),
        iterations: stats.len(),
        throughput_tok_s: compute_throughput(stats),
        latency_p50_ms: percentile(&sorted_lat, 50.0),
        latency_p95_ms: percentile(&sorted_lat, 95.0),
        latency_p99_ms: percentile(&sorted_lat, 99.0),
        bricks: brick_scores,
        hardware,
    }
}

// ============================================================================
// Display Helpers
// ============================================================================

/// Print the hardware info section.
pub fn print_hardware(hw: &HardwareInfo) {
    println!("1. Hardware Inventory");
    println!("   -----------------------------------------------");
    println!("   CPU:    {}", hw.cpu);
    println!("   Memory: {:.0} GB", hw.memory_gb);
    match &hw.gpu {
        Some(gpu) => println!("   GPU:    {}", gpu),
        None => println!("   GPU:    (none)"),
    }
    println!();
}

/// Print per-iteration simulation results.
pub fn print_iterations(all_timings: &[Vec<BrickTiming>], stats: &[IterationStats]) {
    println!("2. Simulated Inference ({} iterations)", stats.len());
    println!("   -----------------------------------------------");
    for (timings, stat) in all_timings.iter().zip(stats.iter()) {
        let brick_summary: Vec<String> = timings
            .iter()
            .map(|t| format!("{}={:.1}ms", t.name, t.elapsed_ms))
            .collect();
        println!(
            "   iter {:>2}: total={:.1}ms  [{}]",
            stat.iteration,
            stat.total_ms,
            brick_summary.join(", ")
        );
    }
    println!();
}

/// Print brick score table.
pub fn print_brick_scores(scores: &[BrickScore]) {
    println!("3. Brick Score Aggregation");
    println!("   -----------------------------------------------");
    println!(
        "   {:>15}  {:>8}  {:>8}  {:>10}",
        "Brick", "Mean(ms)", "P95(ms)", "Budget(%)"
    );
    for score in scores {
        println!(
            "   {:>15}  {:>8.2}  {:>8.2}  {:>9.1}%",
            score.name, score.mean_ms, score.p95_ms, score.pct_budget
        );
    }
    println!();
}

/// Print throughput and latency percentiles.
pub fn print_throughput_latency(report: &HeadlessReport) {
    println!("4. Throughput and Latency Percentiles");
    println!("   -----------------------------------------------");
    println!("   Throughput: {:.1} tokens/sec", report.throughput_tok_s);
    println!("   Latency p50:  {:.2} ms", report.latency_p50_ms);
    println!("   Latency p95:  {:.2} ms", report.latency_p95_ms);
    println!("   Latency p99:  {:.2} ms", report.latency_p99_ms);
    println!();
}

/// Print budget utilization breakdown.
pub fn print_budget_utilization(scores: &[BrickScore]) {
    println!("5. Performance Budget Utilization");
    println!("   -----------------------------------------------");
    println!("   Budget per iteration: {:.0} ms", BUDGET_MS);
    let total_pct: f64 = scores.iter().map(|s| s.pct_budget).sum();
    for score in scores {
        let bar_len = (score.pct_budget / 2.0).round() as usize;
        let bar: String = "#".repeat(bar_len.min(50));
        println!(
            "   {:>15}: {:<50} {:.1}%",
            score.name, bar, score.pct_budget
        );
    }
    println!("   {:>15}: {:.1}%", "TOTAL", total_pct);
    let status = if total_pct <= 100.0 {
        "WITHIN BUDGET"
    } else {
        "OVER BUDGET"
    };
    println!("   Status: {}", status);
    println!();
}

/// Print the final structured headless report.
pub fn print_headless_report(report: &HeadlessReport) {
    println!("6. Headless Report Summary");
    println!("   -----------------------------------------------");
    println!("   model_name:       {}", report.model_name);
    println!("   iterations:       {}", report.iterations);
    println!("   throughput_tok_s: {:.1}", report.throughput_tok_s);
    println!("   latency_p50_ms:  {:.2}", report.latency_p50_ms);
    println!("   latency_p95_ms:  {:.2}", report.latency_p95_ms);
    println!("   latency_p99_ms:  {:.2}", report.latency_p99_ms);
    println!("   hardware:");
    println!("     cpu:       {}", report.hardware.cpu);
    println!("     memory_gb: {:.0}", report.hardware.memory_gb);
    match &report.hardware.gpu {
        Some(gpu) => println!("     gpu:       {}", gpu),
        None => println!("     gpu:       null"),
    }
    println!("   bricks:");
    for brick in &report.bricks {
        println!(
            "     - name: {:>15}, mean_ms: {:>6.2}, p95_ms: {:>6.2}, pct_budget: {:>5.1}%",
            brick.name, brick.mean_ms, brick.p95_ms, brick.pct_budget
        );
    }
}

// ============================================================================
// Main
// ============================================================================
