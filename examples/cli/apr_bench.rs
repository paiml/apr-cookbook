//! Benchmark APR model inference performance.
//!
//! This CLI tool measures inference latency and throughput for
//! APR models, helping identify performance characteristics.
//!
//! # Run
//!
//! ```bash
//! cargo run --example apr_bench --release -- --help
//! cargo run --example apr_bench --release -- --demo --iterations 1000
//! ```

use apr_cookbook::bundle::{BundledModel, ModelBundle};
use apr_cookbook::Result;
use clap::Parser;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "apr-bench")]
#[command(about = "Benchmark APR model inference")]
#[command(version)]
struct Args {
    /// Path to the APR model file
    #[arg(value_name = "FILE")]
    path: Option<PathBuf>,

    /// Number of iterations
    #[arg(short, long, default_value = "1000")]
    iterations: usize,

    /// Warmup iterations
    #[arg(short, long, default_value = "100")]
    warmup: usize,

    /// Batch size for inference
    #[arg(short, long, default_value = "1")]
    batch_size: usize,

    /// Use demo mode with sample model
    #[arg(long)]
    demo: bool,
}

/// Benchmark results
struct BenchResult {
    iterations: usize,
    total_time: Duration,
    mean_latency: Duration,
    min_latency: Duration,
    max_latency: Duration,
    p50_latency: Duration,
    p99_latency: Duration,
    throughput: f64,
}

impl BenchResult {
    fn display(&self) {
        println!("=== Benchmark Results ===\n");

        println!("Iterations:    {}", self.iterations);
        println!("Total time:    {:?}", self.total_time);
        println!();

        println!("Latency:");
        println!("  Mean:        {:?}", self.mean_latency);
        println!("  Min:         {:?}", self.min_latency);
        println!("  Max:         {:?}", self.max_latency);
        println!("  P50:         {:?}", self.p50_latency);
        println!("  P99:         {:?}", self.p99_latency);
        println!();

        println!("Throughput:    {:.2} inferences/sec", self.throughput);
    }
}

fn create_demo_model(size: usize) -> Vec<u8> {
    ModelBundle::new()
        .with_name("benchmark-model")
        .with_payload(vec![0u8; size])
        .build()
}

/// Simulated inference function
fn simulate_inference(model: &BundledModel, _batch_size: usize) {
    // Simulate some work proportional to model size
    let bytes = model.as_bytes();
    let _sum: u64 = bytes.iter().take(100).map(|&b| u64::from(b)).sum();
}

fn run_benchmark(model: &BundledModel, args: &Args) -> BenchResult {
    let mut latencies = Vec::with_capacity(args.iterations);

    // Warmup
    for _ in 0..args.warmup {
        simulate_inference(model, args.batch_size);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..args.iterations {
        let iter_start = Instant::now();
        simulate_inference(model, args.batch_size);
        latencies.push(iter_start.elapsed());
    }
    let total_time = start.elapsed();

    // Calculate statistics
    latencies.sort();

    let sum: Duration = latencies.iter().sum();
    let mean_latency = sum / args.iterations as u32;
    let min_latency = latencies[0];
    let max_latency = latencies[latencies.len() - 1];
    let p50_latency = latencies[args.iterations / 2];
    let p99_latency = latencies[(args.iterations * 99) / 100];
    let throughput = args.iterations as f64 / total_time.as_secs_f64();

    BenchResult {
        iterations: args.iterations,
        total_time,
        mean_latency,
        min_latency,
        max_latency,
        p50_latency,
        p99_latency,
        throughput,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== APR Cookbook: Model Benchmark ===\n");

    let model_bytes = match args.path.as_ref() {
        Some(path) if !args.demo => {
            println!("Model: {}\n", path.display());
            // In production: std::fs::read(path)?
            create_demo_model(10000)
        }
        _ => {
            if !args.demo {
                println!("No file specified. Running in demo mode.\n");
            }
            create_demo_model(10000)
        }
    };

    let model = BundledModel::from_bytes(&model_bytes)?;

    println!("Model size:    {} bytes", model.size());
    println!("Batch size:    {}", args.batch_size);
    println!("Warmup:        {} iterations", args.warmup);
    println!("Benchmark:     {} iterations\n", args.iterations);

    println!("Running benchmark...");
    let result = run_benchmark(&model, &args);
    println!();

    result.display();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_model_creation() {
        let bytes = create_demo_model(1000);
        let model = BundledModel::from_bytes(&bytes);
        assert!(model.is_ok());
    }

    #[test]
    fn test_benchmark_runs() {
        let bytes = create_demo_model(1000);
        let model = BundledModel::from_bytes(&bytes).unwrap();

        let args = Args {
            path: None,
            iterations: 10,
            warmup: 2,
            batch_size: 1,
            demo: true,
        };

        let result = run_benchmark(&model, &args);
        assert_eq!(result.iterations, 10);
        assert!(result.throughput > 0.0);
    }

    #[test]
    fn test_latency_ordering() {
        let bytes = create_demo_model(1000);
        let model = BundledModel::from_bytes(&bytes).unwrap();

        let args = Args {
            path: None,
            iterations: 100,
            warmup: 10,
            batch_size: 1,
            demo: true,
        };

        let result = run_benchmark(&model, &args);

        // Min <= P50 <= P99 <= Max
        assert!(result.min_latency <= result.p50_latency);
        assert!(result.p50_latency <= result.p99_latency);
        assert!(result.p99_latency <= result.max_latency);
    }

    #[test]
    fn test_cli_args_defaults() {
        let args = Args::try_parse_from(["apr-bench", "--demo"]).unwrap();
        assert_eq!(args.iterations, 1000);
        assert_eq!(args.warmup, 100);
        assert_eq!(args.batch_size, 1);
    }
}
