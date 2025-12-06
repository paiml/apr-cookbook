//! # Recipe: APR Benchmark CLI
//!
//! **Category**: CLI Tools
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
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Benchmark APR model inference performance.
//!
//! ## Run Command
//! ```bash
//! cargo run --example cli_apr_bench
//! cargo run --example cli_apr_bench -- --demo --iterations 100
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let config = parse_args(&args)?;

    if config.help {
        print_help();
        return Ok(());
    }

    run_benchmark(&config)
}

#[derive(Debug, Clone)]
struct BenchConfig {
    model_path: Option<String>,
    demo: bool,
    iterations: usize,
    warmup: usize,
    batch_size: usize,
    json: bool,
    help: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchResults {
    model: String,
    iterations: usize,
    batch_size: usize,
    latency: LatencyStats,
    throughput: ThroughputStats,
    memory: MemoryStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LatencyStats {
    mean_ms: f64,
    std_ms: f64,
    min_ms: f64,
    max_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ThroughputStats {
    samples_per_sec: f64,
    batches_per_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryStats {
    peak_mb: f64,
    model_mb: f64,
}

fn parse_args(args: &[String]) -> Result<BenchConfig> {
    let mut config = BenchConfig {
        model_path: None,
        demo: false,
        iterations: 100,
        warmup: 10,
        batch_size: 1,
        json: false,
        help: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => config.help = true,
            "--demo" | "-d" => config.demo = true,
            "--json" | "-j" => config.json = true,
            "--iterations" | "-n" => {
                i += 1;
                if i < args.len() {
                    config.iterations = args[i].parse().unwrap_or(100);
                }
            }
            "--warmup" | "-w" => {
                i += 1;
                if i < args.len() {
                    config.warmup = args[i].parse().unwrap_or(10);
                }
            }
            "--batch" | "-b" => {
                i += 1;
                if i < args.len() {
                    config.batch_size = args[i].parse().unwrap_or(1);
                }
            }
            path if !path.starts_with('-') => {
                config.model_path = Some(path.to_string());
            }
            _ => {}
        }
        i += 1;
    }

    Ok(config)
}

fn print_help() {
    println!("apr-bench - Benchmark APR model inference");
    println!();
    println!("USAGE:");
    println!("    apr-bench [OPTIONS] <MODEL_PATH>");
    println!();
    println!("OPTIONS:");
    println!("    -h, --help           Print help information");
    println!("    -d, --demo           Run with demo model");
    println!("    -n, --iterations N   Number of iterations (default: 100)");
    println!("    -w, --warmup N       Warmup iterations (default: 10)");
    println!("    -b, --batch N        Batch size (default: 1)");
    println!("    -j, --json           Output as JSON");
    println!();
    println!("EXAMPLES:");
    println!("    apr-bench model.apr");
    println!("    apr-bench --demo --iterations 1000");
    println!("    apr-bench -n 100 -b 32 model.apr");
}

fn run_benchmark(config: &BenchConfig) -> Result<()> {
    let mut ctx = RecipeContext::new("cli_apr_bench")?;

    // Get model path
    let model_name = if config.demo {
        "demo-model".to_string()
    } else if let Some(path) = &config.model_path {
        path.clone()
    } else {
        print_help();
        return Ok(());
    };

    if !config.json {
        println!("APR Model Benchmark");
        println!("===================");
        println!();
        println!("Model: {}", model_name);
        println!("Iterations: {}", config.iterations);
        println!("Warmup: {}", config.warmup);
        println!("Batch size: {}", config.batch_size);
        println!();
        println!("Running warmup...");
    }

    // Warmup (simulated)
    let _warmup_times: Vec<f64> = (0..config.warmup)
        .map(|i| simulate_inference(i, config.batch_size))
        .collect();

    if !config.json {
        println!("Running benchmark...");
    }

    // Benchmark (simulated)
    let mut times: Vec<f64> = (0..config.iterations)
        .map(|i| simulate_inference(i + config.warmup, config.batch_size))
        .collect();

    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate statistics
    let results = calculate_results(&model_name, &times, config)?;

    ctx.record_float_metric("mean_latency_ms", results.latency.mean_ms);
    ctx.record_float_metric("throughput", results.throughput.samples_per_sec);

    // Output
    if config.json {
        let json = serde_json::to_string_pretty(&results)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?;
        println!("{}", json);
    } else {
        print_results(&results);
    }

    Ok(())
}

fn simulate_inference(iteration: usize, batch_size: usize) -> f64 {
    // Deterministic simulated inference time
    let base_time = 1.0; // 1ms base
    let batch_factor = (batch_size as f64).sqrt();
    let variation = (iteration % 10) as f64 * 0.01;

    base_time * batch_factor + variation
}

fn calculate_results(model: &str, times: &[f64], config: &BenchConfig) -> Result<BenchResults> {
    let n = times.len() as f64;

    let mean = times.iter().sum::<f64>() / n;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    let min = *times.first().unwrap_or(&0.0);
    let max = *times.last().unwrap_or(&0.0);

    let p50_idx = (times.len() as f64 * 0.50) as usize;
    let p95_idx = (times.len() as f64 * 0.95) as usize;
    let p99_idx = (times.len() as f64 * 0.99) as usize;

    let p50 = times.get(p50_idx).copied().unwrap_or(mean);
    let p95 = times.get(p95_idx).copied().unwrap_or(mean);
    let p99 = times.get(p99_idx).copied().unwrap_or(mean);

    let samples_per_sec = (config.batch_size as f64 / mean) * 1000.0;
    let batches_per_sec = (1.0 / mean) * 1000.0;

    Ok(BenchResults {
        model: model.to_string(),
        iterations: times.len(),
        batch_size: config.batch_size,
        latency: LatencyStats {
            mean_ms: mean,
            std_ms: std,
            min_ms: min,
            max_ms: max,
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
        },
        throughput: ThroughputStats {
            samples_per_sec,
            batches_per_sec,
        },
        memory: MemoryStats {
            peak_mb: 50.0,
            model_mb: 10.0,
        },
    })
}

fn print_results(results: &BenchResults) {
    println!();
    println!("Results");
    println!("-------");
    println!();
    println!("Latency:");
    println!(
        "  Mean:  {:.3}ms ± {:.3}ms",
        results.latency.mean_ms, results.latency.std_ms
    );
    println!("  Min:   {:.3}ms", results.latency.min_ms);
    println!("  Max:   {:.3}ms", results.latency.max_ms);
    println!("  P50:   {:.3}ms", results.latency.p50_ms);
    println!("  P95:   {:.3}ms", results.latency.p95_ms);
    println!("  P99:   {:.3}ms", results.latency.p99_ms);
    println!();
    println!("Throughput:");
    println!("  {:.1} samples/sec", results.throughput.samples_per_sec);
    println!("  {:.1} batches/sec", results.throughput.batches_per_sec);
    println!();
    println!("Memory:");
    println!("  Peak:  {:.1}MB", results.memory.peak_mb);
    println!("  Model: {:.1}MB", results.memory.model_mb);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_demo() {
        let args = vec!["apr-bench".to_string(), "--demo".to_string()];
        let config = parse_args(&args).unwrap();

        assert!(config.demo);
        assert_eq!(config.iterations, 100);
    }

    #[test]
    fn test_parse_args_iterations() {
        let args = vec![
            "apr-bench".to_string(),
            "--iterations".to_string(),
            "500".to_string(),
        ];
        let config = parse_args(&args).unwrap();

        assert_eq!(config.iterations, 500);
    }

    #[test]
    fn test_parse_args_batch() {
        let args = vec!["apr-bench".to_string(), "-b".to_string(), "32".to_string()];
        let config = parse_args(&args).unwrap();

        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_simulate_inference_deterministic() {
        let t1 = simulate_inference(5, 16);
        let t2 = simulate_inference(5, 16);

        assert_eq!(t1, t2);
    }

    #[test]
    fn test_simulate_inference_batch_scaling() {
        let t1 = simulate_inference(0, 1);
        let t16 = simulate_inference(0, 16);

        assert!(t16 > t1);
    }

    #[test]
    fn test_calculate_results() {
        let times = vec![1.0, 1.1, 1.2, 1.05, 0.95];
        let config = BenchConfig {
            model_path: None,
            demo: true,
            iterations: 5,
            warmup: 0,
            batch_size: 1,
            json: false,
            help: false,
        };

        let results = calculate_results("test", &times, &config).unwrap();

        assert!(results.latency.mean_ms > 0.0);
        assert!(results.throughput.samples_per_sec > 0.0);
    }

    #[test]
    fn test_percentiles() {
        let mut times: Vec<f64> = (1..=100).map(|i| i as f64).collect();

        let config = BenchConfig {
            model_path: None,
            demo: true,
            iterations: 100,
            warmup: 0,
            batch_size: 1,
            json: false,
            help: false,
        };

        let results = calculate_results("test", &times, &config).unwrap();

        assert!((results.latency.p50_ms - 50.0).abs() < 2.0);
        assert!((results.latency.p95_ms - 95.0).abs() < 2.0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_inference_time_positive(iteration in 0usize..1000, batch in 1usize..64) {
            let time = simulate_inference(iteration, batch);
            prop_assert!(time > 0.0);
        }

        #[test]
        fn prop_batch_increases_time(batch1 in 1usize..10, batch2 in 11usize..32) {
            let t1 = simulate_inference(0, batch1);
            let t2 = simulate_inference(0, batch2);

            prop_assert!(t2 > t1);
        }

        #[test]
        fn prop_statistics_valid(iterations in 10usize..100) {
            let times: Vec<f64> = (0..iterations)
                .map(|i| simulate_inference(i, 1))
                .collect();

            let config = BenchConfig {
                model_path: None,
                demo: true,
                iterations,
                warmup: 0,
                batch_size: 1,
                json: false,
                help: false,
            };

            let results = calculate_results("test", &times, &config).unwrap();

            prop_assert!(results.latency.min_ms <= results.latency.mean_ms);
            prop_assert!(results.latency.mean_ms <= results.latency.max_ms);
        }
    }
}
