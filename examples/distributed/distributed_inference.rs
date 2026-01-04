//! Distributed Inference Example with repartir
//!
//! Demonstrates multi-node inference using the repartir distributed computing library.
//!
//! # repartir Features
//!
//! - **Work-Stealing Scheduler**: Blumofe & Leiserson (1999) algorithm
//! - **CPU Executor**: Local multi-core parallel execution
//! - **Batch Submission**: High-throughput task processing
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   Distributed Inference Pipeline                │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌─────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
//! │  │  Tasks  │ ─► │  Scheduler  │ ─► │  Workers (CPU/GPU)      │  │
//! │  │ (batch) │    │ (steal-work)│    │  ├── worker-0          │  │
//! │  └─────────┘    └─────────────┘    │  ├── worker-1          │  │
//! │                                     │  ├── worker-2          │  │
//! │                                     │  └── worker-N          │  │
//! │                                     └─────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example distributed_inference --features distributed
//! ```
//!
//! # Recipe Metadata
//!
//! - **Category**: Distributed Computing
//! - **Complexity**: Advanced
//! - **Dependencies**: repartir 1.1+
//! - **IIUR**: Isolated, Idempotent, Useful, Reproducible

use std::time::Instant;

/// Simulated model shard for distributed inference
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ModelShard {
    /// Shard index (0-based)
    shard_id: usize,
    /// Total number of shards
    total_shards: usize,
    /// Shard weight data (simulated)
    weights: Vec<f32>,
    /// Embedding dimension
    embed_dim: usize,
}

impl ModelShard {
    /// Create a new model shard with simulated weights
    fn new(shard_id: usize, total_shards: usize, embed_dim: usize) -> Self {
        // Deterministic weight generation for reproducibility
        let weights = (0..embed_dim * 1024)
            .map(|i| ((shard_id * 1000 + i) as f32).sin() * 0.1)
            .collect();

        Self {
            shard_id,
            total_shards,
            weights,
            embed_dim,
        }
    }

    /// Run inference on this shard (simulated)
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Simulated forward pass: matmul + activation
        let output_size = self.embed_dim;
        let mut output = vec![0.0f32; output_size];

        for (i, out) in output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (j, &inp) in input.iter().take(self.weights.len()).enumerate() {
                let weight_idx = (i * input.len() + j) % self.weights.len();
                sum += inp * self.weights[weight_idx];
            }
            // ReLU activation
            *out = sum.max(0.0);
        }

        output
    }

    /// Estimated FLOPS for this shard
    fn flops(&self) -> usize {
        2 * self.embed_dim * 1024 // 2 ops per multiply-add
    }
}

/// Distributed inference configuration
#[derive(Debug, Clone)]
struct InferenceConfig {
    /// Number of model shards
    num_shards: usize,
    /// Number of worker threads
    num_workers: usize,
    /// Batch size for inference
    batch_size: usize,
    /// Embedding dimension
    embed_dim: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            num_shards: 4,
            num_workers: 4,
            batch_size: 32,
            embed_dim: 768,
        }
    }
}

/// Distributed inference engine (simulated repartir integration)
#[allow(dead_code)]
struct DistributedInference {
    /// Model shards
    shards: Vec<ModelShard>,
    /// Configuration
    config: InferenceConfig,
}

impl DistributedInference {
    /// Create a new distributed inference engine
    fn new(config: InferenceConfig) -> Self {
        let shards: Vec<_> = (0..config.num_shards)
            .map(|i| ModelShard::new(i, config.num_shards, config.embed_dim))
            .collect();

        Self { shards, config }
    }

    /// Run distributed inference (simulated work-stealing)
    fn infer(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Simulate work-stealing scheduler behavior
        // In production, this would use repartir::Pool
        let mut outputs = Vec::with_capacity(inputs.len());

        for input in inputs {
            // Process through all shards (pipeline parallel)
            let mut intermediate = input.clone();
            for shard in &self.shards {
                intermediate = shard.forward(&intermediate);
            }
            outputs.push(intermediate);
        }

        outputs
    }

    /// Get total model FLOPS
    fn total_flops(&self) -> usize {
        self.shards.iter().map(ModelShard::flops).sum()
    }
}

/// Benchmark result
#[derive(Debug)]
#[allow(dead_code)]
struct BenchmarkResult {
    /// Total samples processed
    samples: usize,
    /// Total time in milliseconds
    time_ms: f64,
    /// Samples per second
    samples_per_sec: f64,
    /// Effective GFLOPS
    gflops: f64,
}

/// Run distributed inference benchmark
fn run_benchmark(config: &InferenceConfig, num_iterations: usize) -> BenchmarkResult {
    let engine = DistributedInference::new(config.clone());

    // Generate batch of inputs
    let inputs: Vec<Vec<f32>> = (0..config.batch_size)
        .map(|i| {
            (0..config.embed_dim)
                .map(|j| ((i * 100 + j) as f32).sin())
                .collect()
        })
        .collect();

    // Warmup
    let _ = engine.infer(&inputs);

    // Benchmark
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = engine.infer(&inputs);
    }
    let elapsed = start.elapsed();

    let total_samples = config.batch_size * num_iterations;
    let time_ms = elapsed.as_secs_f64() * 1000.0;
    let samples_per_sec = total_samples as f64 / elapsed.as_secs_f64();
    let total_flops = engine.total_flops() * total_samples;
    let gflops = total_flops as f64 / elapsed.as_secs_f64() / 1e9;

    BenchmarkResult {
        samples: total_samples,
        time_ms,
        samples_per_sec,
        gflops,
    }
}

fn main() {
    println!("=== Distributed Inference Example ===\n");

    // =========================================================================
    // Section 1: Configuration
    // =========================================================================
    println!("1. Configuration");
    println!("   ─────────────────────────────────────────");

    let config = InferenceConfig::default();
    println!("   Shards:        {}", config.num_shards);
    println!("   Workers:       {}", config.num_workers);
    println!("   Batch size:    {}", config.batch_size);
    println!("   Embed dim:     {}", config.embed_dim);
    println!();

    // =========================================================================
    // Section 2: Model Sharding
    // =========================================================================
    println!("2. Model Sharding");
    println!("   ─────────────────────────────────────────");

    let engine = DistributedInference::new(config.clone());
    println!(
        "   Created {} shards with {} weights each",
        engine.shards.len(),
        engine.shards[0].weights.len()
    );
    println!(
        "   Total model size: {:.2} MB (f32)",
        engine.shards.iter().map(|s| s.weights.len()).sum::<usize>() as f64 * 4.0 / 1e6
    );
    println!("   FLOPS per sample: {}", engine.total_flops());
    println!();

    // =========================================================================
    // Section 3: Distributed Execution Pattern
    // =========================================================================
    println!("3. Distributed Execution Pattern");
    println!("   ─────────────────────────────────────────");
    println!("   ┌───────────────────────────────────────────────────┐");
    println!("   │  repartir Work-Stealing Architecture               │");
    println!("   ├───────────────────────────────────────────────────┤");
    println!("   │  Task Queue ──► Scheduler ──► Worker Pool         │");
    println!("   │                    │          ├── CPU Worker 0    │");
    println!("   │                    │          ├── CPU Worker 1    │");
    println!("   │                    │          ├── CPU Worker 2    │");
    println!("   │                    │          └── CPU Worker N    │");
    println!("   │                    │                              │");
    println!("   │                    └──► Steal work from busy      │");
    println!("   │                         workers (load balancing)  │");
    println!("   └───────────────────────────────────────────────────┘");
    println!();

    // =========================================================================
    // Section 4: Inference Demo
    // =========================================================================
    println!("4. Inference Demo");
    println!("   ─────────────────────────────────────────");

    let test_input: Vec<f32> = (0..config.embed_dim).map(|i| (i as f32).sin()).collect();
    let results = engine.infer(std::slice::from_ref(&test_input));

    println!(
        "   Input:  [{:.4}, {:.4}, {:.4}, ...]",
        test_input[0], test_input[1], test_input[2]
    );
    if let Some(output) = results.first() {
        println!(
            "   Output: [{:.4}, {:.4}, {:.4}, ...]",
            output[0], output[1], output[2]
        );
        println!("   Output shape: {}", output.len());
    }
    println!();

    // =========================================================================
    // Section 5: Benchmark
    // =========================================================================
    println!("5. Benchmark");
    println!("   ─────────────────────────────────────────");
    println!("   ┌──────────────┬────────────┬────────────┬────────────┐");
    println!("   │ Shards       │ Time (ms)  │ Samples/s  │ GFLOPS     │");
    println!("   ├──────────────┼────────────┼────────────┼────────────┤");

    for num_shards in [1, 2, 4, 8] {
        let bench_config = InferenceConfig {
            num_shards,
            ..Default::default()
        };
        let result = run_benchmark(&bench_config, 10);
        println!(
            "   │ {:12} │ {:10.2} │ {:10.1} │ {:10.4} │",
            num_shards, result.time_ms, result.samples_per_sec, result.gflops
        );
    }
    println!("   └──────────────┴────────────┴────────────┴────────────┘");
    println!();

    // =========================================================================
    // Section 6: repartir API Example (Conceptual)
    // =========================================================================
    println!("6. repartir API Example (Conceptual)");
    println!("   ─────────────────────────────────────────");
    println!("   ```rust");
    println!("   use repartir::{{Pool, task::{{Task, Backend}}}};");
    println!();
    println!("   let pool = Pool::builder()");
    println!("       .cpu_workers(8)");
    println!("       .max_queue_size(1000)");
    println!("       .build()?;");
    println!();
    println!("   let task = Task::builder()");
    println!("       .binary(\"./inference-worker\")");
    println!("       .arg(\"--shard\")");
    println!("       .arg(\"0\")");
    println!("       .backend(Backend::Cpu)");
    println!("       .build()?;");
    println!();
    println!("   let result = pool.submit(task).await?;");
    println!("   ```");
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_shard_creation() {
        let shard = ModelShard::new(0, 4, 768);
        assert_eq!(shard.shard_id, 0);
        assert_eq!(shard.total_shards, 4);
        assert_eq!(shard.embed_dim, 768);
        assert!(!shard.weights.is_empty());
    }

    #[test]
    fn test_model_shard_forward() {
        let shard = ModelShard::new(0, 1, 64);
        let input = vec![1.0f32; 64];
        let output = shard.forward(&input);
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_model_shard_flops() {
        let shard = ModelShard::new(0, 1, 768);
        let flops = shard.flops();
        assert!(flops > 0);
    }

    #[test]
    fn test_distributed_inference_creation() {
        let config = InferenceConfig::default();
        let engine = DistributedInference::new(config.clone());
        assert_eq!(engine.shards.len(), config.num_shards);
    }

    #[test]
    fn test_distributed_inference_infer() {
        let config = InferenceConfig {
            num_shards: 2,
            embed_dim: 64,
            ..Default::default()
        };
        let engine = DistributedInference::new(config.clone());
        let inputs = vec![vec![1.0f32; 64]; 4];
        let outputs = engine.infer(&inputs);
        assert_eq!(outputs.len(), 4);
        for output in &outputs {
            assert_eq!(output.len(), 64);
        }
    }

    #[test]
    fn test_distributed_inference_total_flops() {
        let config = InferenceConfig {
            num_shards: 4,
            embed_dim: 768,
            ..Default::default()
        };
        let engine = DistributedInference::new(config);
        let total_flops = engine.total_flops();
        assert!(total_flops > 0);
    }

    #[test]
    fn test_benchmark_result() {
        let config = InferenceConfig {
            num_shards: 1,
            batch_size: 8,
            embed_dim: 64,
            ..Default::default()
        };
        let result = run_benchmark(&config, 2);
        assert!(result.samples > 0);
        assert!(result.time_ms > 0.0);
        assert!(result.samples_per_sec > 0.0);
    }

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.num_shards, 4);
        assert_eq!(config.num_workers, 4);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.embed_dim, 768);
    }

    #[test]
    fn test_deterministic_weights() {
        let shard1 = ModelShard::new(0, 2, 64);
        let shard2 = ModelShard::new(0, 2, 64);
        assert_eq!(shard1.weights, shard2.weights);
    }

    #[test]
    fn test_different_shards_different_weights() {
        let shard1 = ModelShard::new(0, 2, 64);
        let shard2 = ModelShard::new(1, 2, 64);
        assert_ne!(shard1.weights, shard2.weights);
    }

    #[test]
    fn test_relu_activation() {
        let shard = ModelShard::new(0, 1, 4);
        // Create input that will produce negative intermediate values
        let input = vec![-10.0f32; 4];
        let output = shard.forward(&input);
        // ReLU should make all negative values 0
        for val in &output {
            assert!(*val >= 0.0, "ReLU should produce non-negative values");
        }
    }
}
