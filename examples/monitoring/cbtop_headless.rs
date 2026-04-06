//! # Headless Performance Monitoring Report (cbtop)
//!
//! **CLI equivalent:** `apr cbtop --headless --json`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Demonstrates headless inference monitoring: per-brick (layer) timing
//! collection, hardware inventory, throughput computation, latency percentiles,
//! and performance budget utilization -- all without a TUI.
//!
//! ## Sections
//! 1. Hardware inventory -- detect CPU, memory, optional GPU
//! 2. Simulated inference -- 10 iterations with per-brick timing
//! 3. Brick score aggregation -- mean, p95, budget percentage per brick
//! 4. Throughput and latency percentiles -- tokens/sec, p50, p95, p99
//! 5. Budget utilization -- which bricks consume the most of the time budget
//! 6. Headless report -- structured summary of all collected metrics
//!
//! ## Architecture
//!
//! ```text
//! +-------------------------------------------------------------------+
//! |               cbtop Headless Monitor                               |
//! +-------------------------------------------------------------------+
//! |                                                                   |
//! |  Inference Loop -------> BrickTimings -------> BrickScores        |
//! |    (10 iterations)           |                  (mean, p95, pct)  |
//! |                              v                                    |
//! |                        IterationStats -------> Percentiles        |
//! |                              |                  (p50, p95, p99)   |
//! |                              v                                    |
//! |                        HeadlessReport                             |
//! |                              |                                    |
//! |  HardwareInfo ---------------+                                    |
//! |    (cpu, mem, gpu)           |                                    |
//! |                              v                                    |
//! |                        Structured Output                          |
//! +-------------------------------------------------------------------+
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo run --example cbtop_headless
//! ```
//!
//! ## Recipe Metadata
//!
//! - **Category**: Monitoring
//! - **Complexity**: Intermediate
//! - **Dependencies**: apr_cookbook (RecipeContext, rand)
//! - **IIUR**: Isolated, Idempotent, Useful, Reproducible
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;
use rand::Rng;

// ============================================================================
// Constants
// ============================================================================

/// Number of inference iterations to simulate.
const NUM_ITERATIONS: usize = 10;

/// Tokens produced per iteration (simulated generation).
const TOKENS_PER_ITERATION: usize = 128;

/// Total time budget per iteration in milliseconds.
const BUDGET_MS: f64 = 100.0;

/// Brick names representing transformer layers.
const BRICK_NAMES: [&str; 4] = ["attention", "ffn", "normalization", "embedding"];

/// Base latency in milliseconds for each brick (attention, ffn, norm, embed).
const BRICK_BASE_MS: [f64; 4] = [35.0, 25.0, 5.0, 10.0];

/// Jitter coefficient (fraction of base latency).
const JITTER_FRACTION: f64 = 0.15;

// ============================================================================
// Domain Types
// ============================================================================

/// Timing data for a single brick within one iteration.
#[derive(Debug, Clone)]
struct BrickTiming {
    name: String,
    elapsed_ms: f64,
}

/// Aggregate score for a single brick across all iterations.
#[derive(Debug, Clone)]
struct BrickScore {
    name: String,
    mean_ms: f64,
    p95_ms: f64,
    pct_budget: f64,
}

/// Detected hardware information.
#[derive(Debug, Clone)]
struct HardwareInfo {
    cpu: String,
    memory_gb: f64,
    gpu: Option<String>,
}

/// Per-iteration statistics.
#[derive(Debug, Clone)]
struct IterationStats {
    iteration: usize,
    total_ms: f64,
    tokens: usize,
}

/// Complete headless report.
#[derive(Debug, Clone)]
struct HeadlessReport {
    model_name: String,
    iterations: usize,
    throughput_tok_s: f64,
    latency_p50_ms: f64,
    latency_p95_ms: f64,
    latency_p99_ms: f64,
    bricks: Vec<BrickScore>,
    hardware: HardwareInfo,
}

// ============================================================================
// Percentile Computation
// ============================================================================

/// Compute the p-th percentile (0-100) from a sorted slice of f64 values.
///
/// Uses linear interpolation between adjacent ranks.
/// Returns 0.0 for empty input.
fn percentile(sorted: &[f64], p: f64) -> f64 {
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
fn sorted_copy(values: &[f64]) -> Vec<f64> {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

// ============================================================================
// Simulation Helpers
// ============================================================================

/// Simulate brick timings for a single inference iteration.
///
/// Each brick gets its base latency plus deterministic jitter from the RNG.
fn simulate_iteration(rng: &mut impl Rng, iteration: usize) -> Vec<BrickTiming> {
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
fn build_hardware_info() -> HardwareInfo {
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
fn compute_brick_scores(all_timings: &[Vec<BrickTiming>]) -> Vec<BrickScore> {
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
fn compute_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values.iter().sum();
    sum / values.len() as f64
}

/// Compute throughput in tokens per second from iteration stats.
fn compute_throughput(stats: &[IterationStats]) -> f64 {
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
fn build_report(
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
fn print_hardware(hw: &HardwareInfo) {
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
fn print_iterations(all_timings: &[Vec<BrickTiming>], stats: &[IterationStats]) {
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
fn print_brick_scores(scores: &[BrickScore]) {
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
fn print_throughput_latency(report: &HeadlessReport) {
    println!("4. Throughput and Latency Percentiles");
    println!("   -----------------------------------------------");
    println!("   Throughput: {:.1} tokens/sec", report.throughput_tok_s);
    println!("   Latency p50:  {:.2} ms", report.latency_p50_ms);
    println!("   Latency p95:  {:.2} ms", report.latency_p95_ms);
    println!("   Latency p99:  {:.2} ms", report.latency_p99_ms);
    println!();
}

/// Print budget utilization breakdown.
fn print_budget_utilization(scores: &[BrickScore]) {
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
fn print_headless_report(report: &HeadlessReport) {
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

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("cbtop_headless")?;

    println!("=== cbtop Headless Performance Monitor ===\n");

    // Section 1: Hardware inventory
    let hardware = build_hardware_info();
    print_hardware(&hardware);

    // Section 2: Simulate inference iterations
    let mut all_timings: Vec<Vec<BrickTiming>> = Vec::with_capacity(NUM_ITERATIONS);
    let mut stats: Vec<IterationStats> = Vec::with_capacity(NUM_ITERATIONS);

    for i in 0..NUM_ITERATIONS {
        let timings = simulate_iteration(ctx.rng(), i);
        let total_ms: f64 = timings.iter().map(|t| t.elapsed_ms).sum();
        stats.push(IterationStats {
            iteration: i,
            total_ms,
            tokens: TOKENS_PER_ITERATION,
        });
        all_timings.push(timings);
    }

    print_iterations(&all_timings, &stats);

    // Section 3: Brick score aggregation
    let brick_scores = compute_brick_scores(&all_timings);
    print_brick_scores(&brick_scores);

    // Section 4: Throughput and latency percentiles
    let report = build_report(&stats, brick_scores.clone(), hardware);
    print_throughput_latency(&report);

    // Section 5: Budget utilization
    print_budget_utilization(&brick_scores);

    // Section 6: Headless report
    print_headless_report(&report);

    // Record metrics
    ctx.record_float_metric("throughput_tok_s", report.throughput_tok_s);
    ctx.record_float_metric("latency_p50_ms", report.latency_p50_ms);
    ctx.record_float_metric("latency_p95_ms", report.latency_p95_ms);
    ctx.record_float_metric("latency_p99_ms", report.latency_p99_ms);
    ctx.record_metric("iterations", report.iterations as i64);

    println!();
    ctx.report()?;
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn test_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    // -- Brick score computation --

    #[test]
    fn test_brick_score_computation_means() {
        let timings = vec![
            vec![
                BrickTiming {
                    name: "attention".to_string(),
                    elapsed_ms: 30.0,
                },
                BrickTiming {
                    name: "ffn".to_string(),
                    elapsed_ms: 20.0,
                },
            ],
            vec![
                BrickTiming {
                    name: "attention".to_string(),
                    elapsed_ms: 40.0,
                },
                BrickTiming {
                    name: "ffn".to_string(),
                    elapsed_ms: 24.0,
                },
            ],
        ];
        // Only compute for brick names in BRICK_NAMES
        let scores = compute_brick_scores(&timings);
        let attn = scores.iter().find(|s| s.name == "attention");
        assert!(attn.is_some(), "attention brick should be present");
        let attn = attn.expect("checked above");
        let expected_mean = (30.0 + 40.0) / 2.0;
        assert!(
            (attn.mean_ms - expected_mean).abs() < 0.01,
            "attention mean should be {}, got {}",
            expected_mean,
            attn.mean_ms,
        );
    }

    #[test]
    fn test_brick_score_pct_budget() {
        let timings = vec![vec![
            BrickTiming {
                name: "attention".to_string(),
                elapsed_ms: 50.0,
            },
            BrickTiming {
                name: "ffn".to_string(),
                elapsed_ms: 25.0,
            },
            BrickTiming {
                name: "normalization".to_string(),
                elapsed_ms: 5.0,
            },
            BrickTiming {
                name: "embedding".to_string(),
                elapsed_ms: 10.0,
            },
        ]];
        let scores = compute_brick_scores(&timings);
        let attn = scores
            .iter()
            .find(|s| s.name == "attention")
            .expect("present");
        // 50.0 / 100.0 * 100 = 50%
        assert!(
            (attn.pct_budget - 50.0).abs() < 0.01,
            "attention budget should be 50%, got {}",
            attn.pct_budget,
        );
    }

    // -- Throughput calculation --

    #[test]
    fn test_throughput_calculation() {
        let stats = vec![
            IterationStats {
                iteration: 0,
                total_ms: 100.0,
                tokens: 128,
            },
            IterationStats {
                iteration: 1,
                total_ms: 100.0,
                tokens: 128,
            },
        ];
        let throughput = compute_throughput(&stats);
        // 256 tokens / 0.2 sec = 1280 tok/s
        assert!(
            (throughput - 1280.0).abs() < 0.1,
            "throughput should be 1280.0, got {}",
            throughput,
        );
    }

    #[test]
    fn test_throughput_zero_time() {
        let stats = vec![IterationStats {
            iteration: 0,
            total_ms: 0.0,
            tokens: 128,
        }];
        let throughput = compute_throughput(&stats);
        assert!(
            (throughput - 0.0).abs() < 0.01,
            "throughput should be 0 for zero time, got {}",
            throughput,
        );
    }

    // -- Percentile math --

    #[test]
    fn test_percentile_sorted_values() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p50 = percentile(&values, 50.0);
        assert!(
            (p50 - 5.5).abs() < 0.01,
            "p50 of 1..10 should be 5.5, got {}",
            p50,
        );
        let p0 = percentile(&values, 0.0);
        assert!((p0 - 1.0).abs() < 0.01, "p0 should be 1.0, got {}", p0,);
        let p100 = percentile(&values, 100.0);
        assert!(
            (p100 - 10.0).abs() < 0.01,
            "p100 should be 10.0, got {}",
            p100,
        );
    }

    #[test]
    fn test_percentile_monotonicity() {
        let values = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 50.0, 100.0];
        let p50 = percentile(&values, 50.0);
        let p95 = percentile(&values, 95.0);
        let p99 = percentile(&values, 99.0);
        assert!(p50 <= p95, "p50={} should be <= p95={}", p50, p95);
        assert!(p95 <= p99, "p95={} should be <= p99={}", p95, p99);
    }

    // -- Budget utilization --

    #[test]
    fn test_budget_utilization_sums_correctly() {
        let timings = vec![vec![
            BrickTiming {
                name: "attention".to_string(),
                elapsed_ms: 35.0,
            },
            BrickTiming {
                name: "ffn".to_string(),
                elapsed_ms: 25.0,
            },
            BrickTiming {
                name: "normalization".to_string(),
                elapsed_ms: 5.0,
            },
            BrickTiming {
                name: "embedding".to_string(),
                elapsed_ms: 10.0,
            },
        ]];
        let scores = compute_brick_scores(&timings);
        let total_pct: f64 = scores.iter().map(|s| s.pct_budget).sum();
        // (35+25+5+10) / 100 * 100 = 75%
        assert!(
            (total_pct - 75.0).abs() < 0.01,
            "total budget should be 75%, got {}",
            total_pct,
        );
    }

    // -- Report generation --

    #[test]
    fn test_report_generation_fields() {
        let stats = vec![
            IterationStats {
                iteration: 0,
                total_ms: 80.0,
                tokens: 128,
            },
            IterationStats {
                iteration: 1,
                total_ms: 85.0,
                tokens: 128,
            },
            IterationStats {
                iteration: 2,
                total_ms: 90.0,
                tokens: 128,
            },
        ];
        let scores = vec![BrickScore {
            name: "attention".to_string(),
            mean_ms: 35.0,
            p95_ms: 38.0,
            pct_budget: 35.0,
        }];
        let hw = build_hardware_info();
        let report = build_report(&stats, scores, hw);

        assert_eq!(report.model_name, "llama-7b.apr");
        assert_eq!(report.iterations, 3);
        assert!(report.throughput_tok_s > 0.0);
        assert!(report.latency_p50_ms > 0.0);
        assert!(report.latency_p50_ms <= report.latency_p95_ms);
        assert!(report.latency_p95_ms <= report.latency_p99_ms);
    }

    // -- Hardware info defaults --

    #[test]
    fn test_hardware_info_defaults() {
        let hw = build_hardware_info();
        assert!(!hw.cpu.is_empty(), "CPU should not be empty");
        assert!(hw.memory_gb > 0.0, "memory should be positive");
        assert!(hw.gpu.is_some(), "default hardware should have GPU");
    }

    #[test]
    fn test_hardware_info_no_gpu() {
        let hw = HardwareInfo {
            cpu: "test-cpu".to_string(),
            memory_gb: 16.0,
            gpu: None,
        };
        assert!(hw.gpu.is_none());
        assert_eq!(hw.cpu, "test-cpu");
    }

    // -- Edge cases --

    #[test]
    fn test_zero_iterations_throughput() {
        let stats: Vec<IterationStats> = vec![];
        let throughput = compute_throughput(&stats);
        assert!(
            (throughput - 0.0).abs() < 0.01,
            "throughput should be 0 with no iterations, got {}",
            throughput,
        );
    }

    #[test]
    fn test_percentile_empty_slice() {
        let values: Vec<f64> = vec![];
        assert!(
            (percentile(&values, 50.0) - 0.0).abs() < 0.01,
            "percentile of empty should be 0",
        );
    }

    #[test]
    fn test_percentile_single_value() {
        let values = vec![42.0];
        assert!(
            (percentile(&values, 50.0) - 42.0).abs() < 0.01,
            "p50 of single value should be that value",
        );
        assert!(
            (percentile(&values, 99.0) - 42.0).abs() < 0.01,
            "p99 of single value should be that value",
        );
    }

    #[test]
    fn test_single_brick_score() {
        let timings = vec![vec![BrickTiming {
            name: "attention".to_string(),
            elapsed_ms: 42.0,
        }]];
        let scores = compute_brick_scores(&timings);
        let attn = scores
            .iter()
            .find(|s| s.name == "attention")
            .expect("present");
        assert!(
            (attn.mean_ms - 42.0).abs() < 0.01,
            "single timing mean should be 42.0, got {}",
            attn.mean_ms,
        );
        assert!(
            (attn.p95_ms - 42.0).abs() < 0.01,
            "single timing p95 should be 42.0, got {}",
            attn.p95_ms,
        );
    }

    // -- Simulation determinism --

    #[test]
    fn test_simulation_deterministic() {
        let mut rng1 = test_rng();
        let mut rng2 = test_rng();
        let t1 = simulate_iteration(&mut rng1, 0);
        let t2 = simulate_iteration(&mut rng2, 0);
        assert_eq!(t1.len(), t2.len());
        for (a, b) in t1.iter().zip(t2.iter()) {
            assert_eq!(a.name, b.name);
            assert!(
                (a.elapsed_ms - b.elapsed_ms).abs() < 1e-10,
                "same seed should produce identical timings",
            );
        }
    }

    #[test]
    fn test_simulate_iteration_produces_all_bricks() {
        let mut rng = test_rng();
        let timings = simulate_iteration(&mut rng, 0);
        assert_eq!(timings.len(), BRICK_NAMES.len());
        for (timing, &expected_name) in timings.iter().zip(BRICK_NAMES.iter()) {
            assert_eq!(timing.name, expected_name);
            assert!(timing.elapsed_ms > 0.0, "timing should be positive");
        }
    }

    #[test]
    fn test_compute_mean_basic() {
        assert!((compute_mean(&[10.0, 20.0, 30.0]) - 20.0).abs() < 0.01);
        assert!((compute_mean(&[]) - 0.0).abs() < 0.01);
        assert!((compute_mean(&[7.0]) - 7.0).abs() < 0.01);
    }
}
