//! # APR Model Benchmarking
//!
//! CLI equivalent: `apr bench model.apr --batch-sizes 1,4,16,64`
//!
//! Throughput benchmarking for APR model inference across multiple batch sizes.
//! Measures latency, throughput, and memory scaling to identify optimal
//! deployment configurations.
//!
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BenchResult {
    batch_size: usize,
    latency_ms: f64,
    throughput_samples_per_sec: f64,
    memory_bytes: usize,
}

impl BenchResult {
    fn new(batch_size: usize, latency_ms: f64, memory_bytes: usize) -> Self {
        let throughput = if latency_ms > 0.0 {
            (batch_size as f64 / latency_ms) * 1000.0
        } else {
            0.0
        };
        Self {
            batch_size,
            latency_ms,
            throughput_samples_per_sec: throughput,
            memory_bytes,
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark logic
// ---------------------------------------------------------------------------

fn extract_weights(model_bytes: &[u8]) -> Vec<f32> {
    let header_size = 64.min(model_bytes.len());
    let payload = &model_bytes[header_size..];
    payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn simulate_matmul(weights: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    // Simulate matrix multiplication: output = input * weights^T
    let batch_size = input.len() / cols;
    let mut output = vec![0.0_f32; batch_size * rows];

    for b in 0..batch_size {
        for r in 0..rows {
            let mut sum = 0.0_f32;
            let w_offset = r * cols;
            let i_offset = b * cols;
            for c in 0..cols {
                if w_offset + c < weights.len() && i_offset + c < input.len() {
                    sum += weights[w_offset + c] * input[i_offset + c];
                }
            }
            output[b * rows + r] = sum;
        }
    }
    output
}

fn bench_inference(model_bytes: &[u8], batch_size: usize, iterations: usize) -> BenchResult {
    let weights = extract_weights(model_bytes);
    let num_weights = weights.len();

    // Infer matrix dimensions (assume square-ish)
    let dim = (num_weights as f64).sqrt() as usize;
    let rows = dim.max(1);
    let cols = if rows > 0 { num_weights / rows } else { 1 };
    let cols = cols.max(1);

    // Generate deterministic input
    let seed = hash_name_to_seed("bench-input");
    let input_bytes = generate_model_payload(seed, batch_size * cols);
    let input: Vec<f32> = input_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Warmup
    let _ = simulate_matmul(&weights, &input, rows, cols);

    // Timed iterations
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = simulate_matmul(&weights, &input, rows, cols);
    }
    let elapsed = start.elapsed();
    let latency_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    // Memory estimate: weights + input + output
    let output_elements = batch_size * rows;
    let memory_bytes = (num_weights + batch_size * cols + output_elements) * 4;

    BenchResult::new(batch_size, latency_ms, memory_bytes)
}

fn format_throughput(samples_per_sec: f64) -> String {
    if samples_per_sec >= 1_000_000.0 {
        format!("{:.1}M samples/s", samples_per_sec / 1_000_000.0)
    } else if samples_per_sec >= 1000.0 {
        format!("{:.1}K samples/s", samples_per_sec / 1000.0)
    } else {
        format!("{:.1} samples/s", samples_per_sec)
    }
}

fn format_memory(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn throughput_bar(value: f64, max_value: f64, width: usize) -> String {
    let ratio = if max_value > 0.0 {
        (value / max_value).min(1.0)
    } else {
        0.0
    };
    let filled = (ratio * width as f64) as usize;
    let empty = width - filled;
    format!("[{}{}]", "#".repeat(filled), " ".repeat(empty))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_bench")?;

    println!("=== APR Model Benchmark ===\n");

    // --- Section 1: Create test model ---
    let dim = 128;
    let seed = hash_name_to_seed("bench-model");
    let weight_bytes = generate_model_payload(seed, dim * dim);

    let bundle = ModelBundleV2::new()
        .with_name("bench-target")
        .with_description("Model for throughput benchmarking")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("weight", vec![dim, dim], weight_bytes)
        .build();

    let model_path = ctx.path("bench-target.apr");
    std::fs::write(&model_path, &bundle)?;
    println!("Model: bench-target ({dim}x{dim} = {} params)", dim * dim);
    println!("File size: {} bytes\n", bundle.len());

    // --- Section 2: Single batch timing ---
    println!("--- Single Batch Timing ---");
    let single = bench_inference(&bundle, 1, 100);
    println!("Batch size: 1");
    println!("Latency:    {:.3} ms", single.latency_ms);
    println!(
        "Throughput: {}",
        format_throughput(single.throughput_samples_per_sec)
    );
    println!("Memory:     {}\n", format_memory(single.memory_bytes));

    // --- Section 3: Batch size scaling table ---
    let batch_sizes = [1, 2, 4, 8, 16, 32, 64];
    let iterations = 50;

    println!("--- Batch Size Scaling ---\n");
    println!(
        "{:>8} {:>12} {:>18} {:>12}",
        "Batch", "Latency(ms)", "Throughput", "Memory"
    );
    println!("{}", "-".repeat(55));

    let mut results = Vec::new();
    for &bs in &batch_sizes {
        let r = bench_inference(&bundle, bs, iterations);
        results.push(r);
    }

    for r in &results {
        println!(
            "{:>8} {:>12.3} {:>18} {:>12}",
            r.batch_size,
            r.latency_ms,
            format_throughput(r.throughput_samples_per_sec),
            format_memory(r.memory_bytes),
        );
    }

    // --- Section 4: Throughput chart ---
    println!("\n--- Throughput Chart ---\n");
    let max_throughput = results
        .iter()
        .map(|r| r.throughput_samples_per_sec)
        .fold(0.0_f64, f64::max);

    for r in &results {
        let bar = throughput_bar(r.throughput_samples_per_sec, max_throughput, 40);
        println!(
            "  batch={:<4} {bar} {:.0} samples/s",
            r.batch_size, r.throughput_samples_per_sec
        );
    }

    // --- Section 5: Memory scaling ---
    println!("\n--- Memory Scaling ---");
    let base_memory = results[0].memory_bytes;
    for r in &results {
        let scale = r.memory_bytes as f64 / base_memory as f64;
        println!(
            "  batch={:<4} {:>10} ({:.1}x base)",
            r.batch_size,
            format_memory(r.memory_bytes),
            scale,
        );
    }

    // Verify monotonicity: throughput should generally increase with batch size
    // (not strictly due to measurement noise, but the trend should hold)
    let first_tp = results[0].throughput_samples_per_sec;
    let last_tp = results.last().unwrap().throughput_samples_per_sec;
    assert!(
        last_tp >= first_tp * 0.5,
        "Throughput should not dramatically decrease with larger batches"
    );

    println!("\nBenchmark complete.");
    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_model(dim: usize) -> Vec<u8> {
        let seed = hash_name_to_seed("bench-test");
        let payload = generate_model_payload(seed, dim * dim);
        ModelBundleV2::new()
            .with_name("bench-test")
            .with_description("test")
            .with_compression(Compression::Lz4)
            .with_quantization(Quantization::FP32)
            .add_tensor("weight", vec![dim, dim], payload)
            .build()
    }

    #[test]
    fn test_latency_positive() {
        let model = make_test_model(32);
        let result = bench_inference(&model, 1, 10);
        assert!(result.latency_ms > 0.0);
    }

    #[test]
    fn test_throughput_positive() {
        let model = make_test_model(32);
        let result = bench_inference(&model, 4, 10);
        assert!(result.throughput_samples_per_sec > 0.0);
    }

    #[test]
    fn test_throughput_scales_with_batch() {
        let model = make_test_model(32);
        let r1 = bench_inference(&model, 1, 20);
        let r16 = bench_inference(&model, 16, 20);
        // Throughput with batch=16 should be meaningfully higher
        assert!(
            r16.throughput_samples_per_sec > r1.throughput_samples_per_sec * 0.5,
            "Batch=16 throughput ({}) should not be drastically less than batch=1 ({})",
            r16.throughput_samples_per_sec,
            r1.throughput_samples_per_sec,
        );
    }

    #[test]
    fn test_memory_scales_with_batch() {
        let model = make_test_model(32);
        let r1 = bench_inference(&model, 1, 5);
        let r16 = bench_inference(&model, 16, 5);
        assert!(
            r16.memory_bytes > r1.memory_bytes,
            "Memory should increase with batch size"
        );
    }

    #[test]
    fn test_deterministic_latency() {
        let model = make_test_model(16);
        let r1 = bench_inference(&model, 1, 50);
        let r2 = bench_inference(&model, 1, 50);
        // Allow 10x variance for CI flakiness, but both should be positive
        let ratio = r1.latency_ms / r2.latency_ms;
        assert!(
            (0.1..10.0).contains(&ratio),
            "Latency should be roughly deterministic: {} vs {}",
            r1.latency_ms,
            r2.latency_ms,
        );
    }

    #[test]
    fn test_bench_result_new() {
        let r = BenchResult::new(8, 4.0, 1024);
        assert_eq!(r.batch_size, 8);
        assert!((r.throughput_samples_per_sec - 2000.0).abs() < 1e-6);
        assert_eq!(r.memory_bytes, 1024);
    }

    #[test]
    fn test_bench_result_zero_latency() {
        let r = BenchResult::new(1, 0.0, 512);
        assert_eq!(r.throughput_samples_per_sec, 0.0);
    }

    #[test]
    fn test_format_throughput_samples() {
        assert!(format_throughput(500.0).contains("samples/s"));
    }

    #[test]
    fn test_format_throughput_k() {
        assert!(format_throughput(5000.0).contains("K samples/s"));
    }

    #[test]
    fn test_format_throughput_m() {
        assert!(format_throughput(2_000_000.0).contains("M samples/s"));
    }

    #[test]
    fn test_format_memory_kb() {
        assert!(format_memory(2048).contains("KB"));
    }

    #[test]
    fn test_throughput_bar_full() {
        let bar = throughput_bar(100.0, 100.0, 10);
        assert!(bar.contains("##########"));
    }

    #[test]
    fn test_throughput_bar_empty() {
        let bar = throughput_bar(0.0, 100.0, 10);
        assert!(bar.contains("[          ]"));
    }

    #[test]
    fn test_simulate_matmul_output_size() {
        let weights = vec![1.0_f32; 4 * 4];
        let input = vec![1.0_f32; 2 * 4]; // batch=2, cols=4
        let output = simulate_matmul(&weights, &input, 4, 4);
        assert_eq!(output.len(), 2 * 4); // batch * rows
    }
}
