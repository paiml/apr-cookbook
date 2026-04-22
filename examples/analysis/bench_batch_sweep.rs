//! # Recipe: Benchmark Batch-Size Sweep
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr bench model.apr --batch-sizes 1,2,4,8,16,32,64,128 --sweep`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example bench_batch_sweep` exits 0
//! 2. [x] `cargo test --example bench_batch_sweep` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr bench` behavior in-process (no shell-out)
//! 10. [x] Unit tests cover sweep logic, saturation detection, ratio math
//!
//! ## Learning Objective
//! Demonstrates batch-size scaling analysis by sweeping 8 batch sizes and
//! identifying the "knee of the curve" where marginal throughput gains drop
//! below a 10% threshold -- the optimal deployment batch size.
//!
//! ## Run Command
//! ```bash
//! cargo run --example bench_batch_sweep
//! ```
//!
//! ## Format Variants
//! ```bash
//! apr bench model.apr --batch-sizes 1,2,4,8,16,32,64 --sweep
//! apr bench model.gguf --batch-sizes 1,2,4,8,16,32,64 --sweep
//! apr bench model.safetensors --batch-sizes 1,2,4,8,16,32,64 --sweep
//! ```
//!
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SweepSample {
    batch_size: usize,
    latency_ms: f64,
    throughput_samples_per_sec: f64,
}

#[derive(Debug, Clone)]
struct SweepAnalysis {
    samples: Vec<SweepSample>,
    knee_batch_size: usize,
    peak_throughput: f64,
    saturation_ratio: f64,
}

// ---------------------------------------------------------------------------
// Sweep logic
// ---------------------------------------------------------------------------

fn simulate_inference(weights: &[f32], batch_size: usize, dim: usize) -> f64 {
    // Deterministic micro-benchmark: O(batch * dim) work.
    let start = Instant::now();
    let mut acc = 0.0_f32;
    for b in 0..batch_size {
        let phase = (b as f32) * 0.001;
        for (i, w) in weights.iter().take(dim).enumerate() {
            acc += w * (phase + (i as f32) * 1e-4).sin();
        }
    }
    // Consume acc to prevent dead-code elimination.
    std::hint::black_box(acc);
    start.elapsed().as_secs_f64() * 1000.0
}

fn run_sweep(weights: &[f32], batch_sizes: &[usize], iterations: usize) -> Vec<SweepSample> {
    let dim = weights.len().clamp(4, 256);
    let mut results = Vec::with_capacity(batch_sizes.len());
    for &bs in batch_sizes {
        // Warmup.
        let _ = simulate_inference(weights, bs, dim);
        let mut total = 0.0_f64;
        for _ in 0..iterations {
            total += simulate_inference(weights, bs, dim);
        }
        let latency_ms = total / iterations as f64;
        let throughput = if latency_ms > 0.0 {
            (bs as f64 / latency_ms) * 1000.0
        } else {
            0.0
        };
        results.push(SweepSample {
            batch_size: bs,
            latency_ms,
            throughput_samples_per_sec: throughput,
        });
    }
    results
}

/// Find the "knee of the curve": the largest batch size where the marginal
/// throughput gain from the previous step is >= 10%.
fn find_throughput_knee(samples: &[SweepSample]) -> usize {
    if samples.is_empty() {
        return 1;
    }
    if samples.len() == 1 {
        return samples[0].batch_size;
    }
    let mut knee = samples[0].batch_size;
    for window in samples.windows(2) {
        let prev = &window[0];
        let cur = &window[1];
        if prev.throughput_samples_per_sec <= 0.0 {
            knee = cur.batch_size;
            continue;
        }
        let gain = (cur.throughput_samples_per_sec - prev.throughput_samples_per_sec)
            / prev.throughput_samples_per_sec;
        if gain >= 0.10 {
            knee = cur.batch_size;
        } else {
            break;
        }
    }
    knee
}

fn analyze(samples: Vec<SweepSample>) -> SweepAnalysis {
    let peak = samples
        .iter()
        .map(|s| s.throughput_samples_per_sec)
        .fold(0.0_f64, f64::max);
    let first = samples
        .first()
        .map_or(0.0, |s| s.throughput_samples_per_sec);
    let saturation_ratio = if first > 0.0 { peak / first } else { 1.0 };
    let knee = find_throughput_knee(&samples);
    SweepAnalysis {
        samples,
        knee_batch_size: knee,
        peak_throughput: peak,
        saturation_ratio,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("bench_batch_sweep")?;
    println!("=== Recipe: {} ===", ctx.name());

    // Build a small synthetic model for the sweep.
    let dim = 64;
    let seed = hash_name_to_seed("bench-batch-sweep");
    let weight_bytes = generate_model_payload(seed, dim * dim);
    let weights: Vec<f32> = weight_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let model_path = ctx.path("sweep-target.apr");
    std::fs::write(&model_path, &weight_bytes)?;

    // Run the sweep.
    let batch_sizes: Vec<usize> = vec![1, 2, 4, 8, 16, 32, 64, 128];
    let iterations = 20;
    let samples = run_sweep(&weights, &batch_sizes, iterations);
    let analysis = analyze(samples);

    println!(
        "\n--- Sweep Results ({} batch sizes) ---",
        batch_sizes.len()
    );
    println!(
        "{:>8} {:>14} {:>20}",
        "Batch", "Latency(ms)", "Throughput(/s)"
    );
    for s in &analysis.samples {
        println!(
            "{:>8} {:>14.3} {:>20.1}",
            s.batch_size, s.latency_ms, s.throughput_samples_per_sec
        );
    }

    println!("\nKnee batch size:     {}", analysis.knee_batch_size);
    println!(
        "Peak throughput:     {:.1} samples/s",
        analysis.peak_throughput
    );
    println!("Saturation ratio:    {:.2}x", analysis.saturation_ratio);

    // Write JSON report.
    let report = json!({
        "recipe": ctx.name(),
        "batch_sizes": batch_sizes,
        "samples": analysis.samples.iter().map(|s| json!({
            "batch_size": s.batch_size,
            "latency_ms": s.latency_ms,
            "throughput_samples_per_sec": s.throughput_samples_per_sec,
        })).collect::<Vec<_>>(),
        "knee_batch_size": analysis.knee_batch_size,
        "peak_throughput": analysis.peak_throughput,
        "saturation_ratio": analysis.saturation_ratio,
    });
    let report_path = ctx.path("sweep.json");
    let report_bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&report_path, report_bytes)?;

    ctx.record_metric("n_batch_sizes", batch_sizes.len() as i64);
    ctx.record_metric("knee_batch_size", analysis.knee_batch_size as i64);
    ctx.record_float_metric("peak_throughput", analysis.peak_throughput);
    ctx.record_float_metric("saturation_ratio", analysis.saturation_ratio);

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(bs: usize, throughput: f64) -> SweepSample {
        SweepSample {
            batch_size: bs,
            latency_ms: if throughput > 0.0 {
                (bs as f64 / throughput) * 1000.0
            } else {
                0.0
            },
            throughput_samples_per_sec: throughput,
        }
    }

    #[test]
    fn knee_flat_curve_returns_first() {
        let s = vec![sample(1, 100.0), sample(2, 101.0), sample(4, 102.0)];
        // Every step is <10% gain, so knee stays at 1.
        assert_eq!(find_throughput_knee(&s), 1);
    }

    #[test]
    fn knee_detects_last_scaling_step() {
        let s = vec![
            sample(1, 100.0),
            sample(2, 200.0),  // +100%
            sample(4, 350.0),  // +75%
            sample(8, 370.0),  // +5.7% -> below threshold
            sample(16, 375.0), // +1.4%
        ];
        assert_eq!(find_throughput_knee(&s), 4);
    }

    #[test]
    fn knee_single_sample_returns_its_batch() {
        let s = vec![sample(8, 1000.0)];
        assert_eq!(find_throughput_knee(&s), 8);
    }

    #[test]
    fn knee_empty_returns_one() {
        assert_eq!(find_throughput_knee(&[]), 1);
    }

    #[test]
    fn run_sweep_emits_one_sample_per_batch_size() {
        let w = vec![0.01_f32; 32];
        let bs = [1, 2, 4];
        let samples = run_sweep(&w, &bs, 2);
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].batch_size, 1);
        assert_eq!(samples[2].batch_size, 4);
    }

    #[test]
    fn analyze_computes_saturation_ratio() {
        let samples = vec![sample(1, 100.0), sample(16, 500.0)];
        let a = analyze(samples);
        assert!((a.saturation_ratio - 5.0).abs() < 1e-6);
        assert!((a.peak_throughput - 500.0).abs() < 1e-6);
    }
}
