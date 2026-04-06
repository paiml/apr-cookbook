//! Distributed Model Sharding Example
//!
//! Demonstrates partitioning large models across multiple workers for distributed
//! inference. Covers layer-wise sharding, tensor-parallel sharding, pipeline
//! parallel execution, fault tolerance with shard reassignment, and memory
//! estimation per shard.
//!
//! # Sharding Strategies
//!
//! ```text
//! Layer-Wise:      [L0,L1] -> W0   [L2,L3] -> W1   [L4,L5] -> W2
//! Tensor-Parallel: [T0_a] -> W0    [T0_b] -> W1    [T0_c] -> W2
//! Pipeline:        W0 -> W1 -> W2  (sequential stage execution)
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example distributed_model_sharding
//! ```
//!
//! # Recipe Metadata
//!
//! - **Category**: Distributed Computing
//! - **Complexity**: Advanced
//! - **Dependencies**: std only
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
//! - Dean, J. et al. (2012). *Large Scale Distributed Deep Networks*. NeurIPS. arXiv:1206.5533

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A single layer in the model architecture.
#[derive(Debug, Clone)]
struct ModelLayer {
    name: String,
    param_count: usize,
    memory_mb: f64,
    compute_flops: u64,
}

impl ModelLayer {
    fn new(name: &str, param_count: usize, memory_mb: f64, compute_flops: u64) -> Self {
        Self {
            name: name.to_string(),
            param_count,
            memory_mb,
            compute_flops,
        }
    }
}

/// A compute worker that can host model shards.
#[derive(Debug, Clone)]
struct Worker {
    id: usize,
    capacity_mb: f64,
    available_mb: f64,
    assigned_shards: Vec<usize>,
}

impl Worker {
    fn new(id: usize, capacity_mb: f64) -> Self {
        Self {
            id,
            capacity_mb,
            available_mb: capacity_mb,
            assigned_shards: Vec::new(),
        }
    }

    /// Attempt to assign a shard. Returns true if the worker has capacity.
    fn assign_shard(&mut self, shard_idx: usize, memory_mb: f64) -> bool {
        if self.available_mb >= memory_mb {
            self.available_mb -= memory_mb;
            self.assigned_shards.push(shard_idx);
            true
        } else {
            false
        }
    }

    /// Remove a shard and reclaim memory (used in tests for verification).
    #[allow(dead_code)]
    fn remove_shard(&mut self, shard_idx: usize, memory_mb: f64) {
        if let Some(pos) = self.assigned_shards.iter().position(|&s| s == shard_idx) {
            self.assigned_shards.remove(pos);
            self.available_mb = (self.available_mb + memory_mb).min(self.capacity_mb);
        }
    }

    /// Utilization as a fraction of total capacity.
    fn utilization(&self) -> f64 {
        if self.capacity_mb == 0.0 {
            return 0.0;
        }
        1.0 - self.available_mb / self.capacity_mb
    }
}

/// A shard representing a subset of the model assigned to a worker.
#[derive(Debug, Clone)]
struct Shard {
    layer_indices: Vec<usize>,
    worker_id: usize,
    memory_requirement: f64,
}

/// Strategy for partitioning the model across workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShardingStrategy {
    LayerWise,
    TensorParallel,
    PipelineParallel,
}

impl ShardingStrategy {
    fn name(self) -> &'static str {
        match self {
            Self::LayerWise => "Layer-Wise",
            Self::TensorParallel => "Tensor-Parallel",
            Self::PipelineParallel => "Pipeline-Parallel",
        }
    }
}

/// A complete sharding plan describing how the model is distributed.
#[derive(Debug, Clone)]
struct ShardingPlan {
    strategy: ShardingStrategy,
    shards: Vec<Shard>,
    total_memory: f64,
    max_worker_load: f64,
}

impl ShardingPlan {
    /// Compute load imbalance: ratio of max worker load to ideal balanced load.
    fn load_imbalance(&self, num_workers: usize) -> f64 {
        if num_workers == 0 || self.total_memory == 0.0 {
            return 0.0;
        }
        let ideal = self.total_memory / num_workers as f64;
        self.max_worker_load / ideal
    }
}

/// Result of a simulated distributed forward pass.
#[derive(Debug)]
struct ForwardPassResult {
    /// Simulated output value per shard (used in tests for verification).
    #[allow(dead_code)]
    outputs: Vec<f64>,
    /// Total latency in simulated microseconds.
    total_latency_us: f64,
    /// Worker ID that was the pipeline bottleneck.
    bottleneck_worker: usize,
}

/// Deterministic hash-based pseudo-random value in [0, 1).
fn det_rand(seed: u64, index: u64) -> f64 {
    let mut h = DefaultHasher::new();
    (seed, index).hash(&mut h);
    h.finish() as f64 / u64::MAX as f64
}

/// Build a sample model architecture with the given number of layers.
fn build_model(num_layers: usize) -> Vec<ModelLayer> {
    (0..num_layers)
        .map(|i| {
            let base_params = 1_000_000 + i * 500_000;
            let memory = base_params as f64 * 4.0 / (1024.0 * 1024.0); // f32 bytes -> MB
            let flops = base_params as u64 * 2; // 2 FLOPS per param (approx)
            ModelLayer::new(&format!("layer_{i}"), base_params, memory, flops)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Sharding algorithms
// ---------------------------------------------------------------------------

/// Layer-wise sharding: assign consecutive layer groups to workers.
fn shard_layer_wise(layers: &[ModelLayer], num_workers: usize) -> ShardingPlan {
    let mut shards = Vec::new();
    let chunk_size = layers.len().div_ceil(num_workers);
    let mut max_load: f64 = 0.0;
    let mut total_memory: f64 = 0.0;

    for (worker_id, chunk) in layers.chunks(chunk_size).enumerate() {
        let indices: Vec<usize> =
            (worker_id * chunk_size..worker_id * chunk_size + chunk.len()).collect();
        let mem: f64 = chunk.iter().map(|l| l.memory_mb).sum();
        total_memory += mem;
        if mem > max_load {
            max_load = mem;
        }
        shards.push(Shard {
            layer_indices: indices,
            worker_id,
            memory_requirement: mem,
        });
    }

    ShardingPlan {
        strategy: ShardingStrategy::LayerWise,
        shards,
        total_memory,
        max_worker_load: max_load,
    }
}

/// Tensor-parallel sharding: split each layer evenly across workers.
fn shard_tensor_parallel(layers: &[ModelLayer], num_workers: usize) -> ShardingPlan {
    let mut shards = Vec::new();
    let mut max_load: f64 = 0.0;
    let mut total_memory: f64 = 0.0;
    let mut worker_loads = vec![0.0f64; num_workers];

    for (layer_idx, layer) in layers.iter().enumerate() {
        let mem_per_worker = layer.memory_mb / num_workers as f64;
        total_memory += layer.memory_mb;
        for (worker_id, load) in worker_loads.iter_mut().enumerate() {
            *load += mem_per_worker;
            shards.push(Shard {
                layer_indices: vec![layer_idx],
                worker_id,
                memory_requirement: mem_per_worker,
            });
        }
    }

    for &load in &worker_loads {
        if load > max_load {
            max_load = load;
        }
    }

    ShardingPlan {
        strategy: ShardingStrategy::TensorParallel,
        shards,
        total_memory,
        max_worker_load: max_load,
    }
}

/// Pipeline-parallel sharding: balanced assignment for sequential execution.
fn shard_pipeline_parallel(layers: &[ModelLayer], num_workers: usize) -> ShardingPlan {
    // Greedy balanced assignment: assign each layer to the least-loaded worker
    let mut worker_loads = vec![0.0f64; num_workers];
    let mut worker_layers: Vec<Vec<usize>> = vec![Vec::new(); num_workers];
    let mut total_memory: f64 = 0.0;

    for (i, layer) in layers.iter().enumerate() {
        // Find worker with minimum load
        let min_worker = worker_loads
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map_or(0, |(idx, _)| idx);
        worker_loads[min_worker] += layer.memory_mb;
        worker_layers[min_worker].push(i);
        total_memory += layer.memory_mb;
    }

    let max_load = worker_loads.iter().fold(0.0f64, |acc, &l| acc.max(l));

    let shards = worker_layers
        .into_iter()
        .enumerate()
        .filter(|(_, indices)| !indices.is_empty())
        .map(|(worker_id, indices)| {
            let mem: f64 = indices.iter().map(|&i| layers[i].memory_mb).sum();
            Shard {
                layer_indices: indices,
                worker_id,
                memory_requirement: mem,
            }
        })
        .collect();

    ShardingPlan {
        strategy: ShardingStrategy::PipelineParallel,
        shards,
        total_memory,
        max_worker_load: max_load,
    }
}

// ---------------------------------------------------------------------------
// Simulation helpers
// ---------------------------------------------------------------------------

/// Simulate a distributed forward pass given a sharding plan.
fn simulate_forward_pass(
    plan: &ShardingPlan,
    layers: &[ModelLayer],
    seed: u64,
) -> ForwardPassResult {
    let mut outputs = Vec::new();
    let mut worker_latencies: std::collections::HashMap<usize, f64> =
        std::collections::HashMap::new();

    for (shard_idx, shard) in plan.shards.iter().enumerate() {
        // Simulated compute time proportional to FLOPS
        let total_flops: u64 = shard
            .layer_indices
            .iter()
            .map(|&i| layers[i].compute_flops)
            .sum();
        let latency = total_flops as f64 / 1e6; // microseconds (1 TFLOPS assumed)
        let output = det_rand(seed, shard_idx as u64) * latency;
        outputs.push(output);

        let entry = worker_latencies.entry(shard.worker_id).or_insert(0.0);
        *entry += latency;
    }

    let (bottleneck_worker, &max_latency) = worker_latencies
        .iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap_or((&0, &0.0));

    ForwardPassResult {
        outputs,
        total_latency_us: max_latency,
        bottleneck_worker: *bottleneck_worker,
    }
}

/// Reassign shards from a failed worker to remaining healthy workers.
fn reassign_shards(
    plan: &ShardingPlan,
    failed_worker: usize,
    workers: &mut [Worker],
) -> Vec<(usize, usize)> {
    let mut reassignments = Vec::new();

    for (shard_idx, shard) in plan.shards.iter().enumerate() {
        if shard.worker_id != failed_worker {
            continue;
        }

        // Find the worker with the most available memory (excluding failed)
        let target = workers
            .iter()
            .filter(|w| w.id != failed_worker)
            .min_by(|a, b| {
                // Prefer higher available memory (sort descending)
                b.available_mb.partial_cmp(&a.available_mb).unwrap()
            })
            .map(|w| w.id);

        if let Some(target_id) = target {
            if workers[target_id].assign_shard(shard_idx, shard.memory_requirement) {
                reassignments.push((shard_idx, target_id));
            }
        }
    }

    reassignments
}

fn main() {
    println!("=== Distributed Model Sharding Example ===\n");

    // =========================================================================
    // Section 1: Define Model Architecture
    // =========================================================================
    println!("1. Model Architecture");
    println!("   ─────────────────────────────────────────");

    let layers = build_model(12);
    let total_params: usize = layers.iter().map(|l| l.param_count).sum();
    let total_memory: f64 = layers.iter().map(|l| l.memory_mb).sum();
    let total_flops: u64 = layers.iter().map(|l| l.compute_flops).sum();

    println!("   Layers:       {}", layers.len());
    println!("   Total params: {total_params}");
    println!("   Total memory: {total_memory:.2} MB");
    println!("   Total FLOPS:  {total_flops}");
    println!();
    println!("   ┌──────────┬────────────┬───────────┬──────────────┐");
    println!("   │ Layer    │ Params     │ Mem (MB)  │ FLOPS        │");
    println!("   ├──────────┼────────────┼───────────┼──────────────┤");
    for layer in &layers {
        println!(
            "   │ {:8} │ {:>10} │ {:>9.2} │ {:>12} │",
            layer.name, layer.param_count, layer.memory_mb, layer.compute_flops
        );
    }
    println!("   └──────────┴────────────┴───────────┴──────────────┘");
    println!();

    // =========================================================================
    // Section 2: Layer-Wise Sharding
    // =========================================================================
    println!("2. Layer-Wise Sharding");
    println!("   ─────────────────────────────────────────");

    let num_workers = 4;
    let plan_lw = shard_layer_wise(&layers, num_workers);

    println!("   Strategy:     {}", plan_lw.strategy.name());
    println!("   Workers:      {num_workers}");
    println!("   Shards:       {}", plan_lw.shards.len());
    println!("   Max load:     {:.2} MB", plan_lw.max_worker_load);
    println!(
        "   Imbalance:    {:.2}x",
        plan_lw.load_imbalance(num_workers)
    );
    println!();

    for shard in &plan_lw.shards {
        println!(
            "   Worker {} <- layers {:?} ({:.2} MB)",
            shard.worker_id, shard.layer_indices, shard.memory_requirement
        );
    }
    println!();

    // =========================================================================
    // Section 3: Tensor-Parallel Sharding
    // =========================================================================
    println!("3. Tensor-Parallel Sharding");
    println!("   ─────────────────────────────────────────");

    let plan_tp = shard_tensor_parallel(&layers, num_workers);

    println!("   Strategy:     {}", plan_tp.strategy.name());
    println!(
        "   Shards:       {} (layers x workers)",
        plan_tp.shards.len()
    );
    println!("   Max load:     {:.2} MB", plan_tp.max_worker_load);
    println!(
        "   Imbalance:    {:.2}x (perfectly balanced)",
        plan_tp.load_imbalance(num_workers)
    );
    println!();

    // =========================================================================
    // Section 4: Pipeline Parallel Execution Simulation
    // =========================================================================
    println!("4. Pipeline Parallel Execution");
    println!("   ─────────────────────────────────────────");

    let plan_pp = shard_pipeline_parallel(&layers, num_workers);

    println!("   Strategy:     {}", plan_pp.strategy.name());
    println!("   Max load:     {:.2} MB", plan_pp.max_worker_load);
    println!(
        "   Imbalance:    {:.2}x",
        plan_pp.load_imbalance(num_workers)
    );
    println!();

    let seed = 42;
    let result_lw = simulate_forward_pass(&plan_lw, &layers, seed);
    let result_tp = simulate_forward_pass(&plan_tp, &layers, seed);
    let result_pp = simulate_forward_pass(&plan_pp, &layers, seed);

    println!("   Forward pass simulation (seed={seed}):");
    println!("   ┌───────────────────┬──────────────┬─────────────────┐");
    println!("   │ Strategy          │ Latency (us) │ Bottleneck W#   │");
    println!("   ├───────────────────┼──────────────┼─────────────────┤");
    println!(
        "   │ {:17} │ {:>12.2} │ {:>15} │",
        "Layer-Wise", result_lw.total_latency_us, result_lw.bottleneck_worker
    );
    println!(
        "   │ {:17} │ {:>12.2} │ {:>15} │",
        "Tensor-Parallel", result_tp.total_latency_us, result_tp.bottleneck_worker
    );
    println!(
        "   │ {:17} │ {:>12.2} │ {:>15} │",
        "Pipeline-Parallel", result_pp.total_latency_us, result_pp.bottleneck_worker
    );
    println!("   └───────────────────┴──────────────┴─────────────────┘");
    println!();

    // =========================================================================
    // Section 5: Worker Failure and Shard Reassignment
    // =========================================================================
    println!("5. Fault Tolerance: Worker Failure");
    println!("   ─────────────────────────────────────────");

    let mut workers: Vec<Worker> = (0..num_workers)
        .map(|id| {
            let capacity = 2048.0 + det_rand(seed, id as u64 + 100) * 1024.0;
            Worker::new(id, capacity)
        })
        .collect();

    // Pre-assign layer-wise shards to workers
    for shard in &plan_lw.shards {
        workers[shard.worker_id].assign_shard(
            plan_lw
                .shards
                .iter()
                .position(|s| std::ptr::eq(s, shard))
                .unwrap_or(0),
            shard.memory_requirement,
        );
    }

    println!("   Before failure:");
    for w in &workers {
        println!(
            "   Worker {}: {:.1}% utilized, {} shards, {:.2}/{:.2} MB",
            w.id,
            w.utilization() * 100.0,
            w.assigned_shards.len(),
            w.capacity_mb - w.available_mb,
            w.capacity_mb,
        );
    }
    println!();

    let failed_worker = 1;
    println!("   >> Worker {failed_worker} FAILED <<\n");

    let reassignments = reassign_shards(&plan_lw, failed_worker, &mut workers);

    if reassignments.is_empty() {
        println!("   No shards could be reassigned (insufficient capacity).");
    } else {
        for (shard_idx, target) in &reassignments {
            println!("   Shard {shard_idx} reassigned to Worker {target}");
        }
    }
    println!();

    println!("   After reassignment:");
    for w in &workers {
        let status = if w.id == failed_worker {
            " [FAILED]"
        } else {
            ""
        };
        println!(
            "   Worker {}: {:.1}% utilized, {} shards{}",
            w.id,
            w.utilization() * 100.0,
            w.assigned_shards.len(),
            status,
        );
    }
    println!();

    // =========================================================================
    // Section 6: Sharding Strategy Comparison Summary
    // =========================================================================
    println!("6. Strategy Comparison Summary");
    println!("   ─────────────────────────────────────────");

    let plans = [
        (&plan_lw, &result_lw),
        (&plan_tp, &result_tp),
        (&plan_pp, &result_pp),
    ];

    println!("   ┌───────────────────┬───────┬───────────┬───────────┬──────────────┐");
    println!("   │ Strategy          │ Shards│ Max (MB)  │ Imbalance │ Latency (us) │");
    println!("   ├───────────────────┼───────┼───────────┼───────────┼──────────────┤");
    for (plan, result) in &plans {
        println!(
            "   │ {:17} │ {:>5} │ {:>9.2} │ {:>9.2}x │ {:>12.2} │",
            plan.strategy.name(),
            plan.shards.len(),
            plan.max_worker_load,
            plan.load_imbalance(num_workers),
            result.total_latency_us,
        );
    }
    println!("   └───────────────────┴───────┴───────────┴───────────┴──────────────┘");
    println!();

    println!("   Key insights:");
    println!("   - Tensor-Parallel achieves perfect memory balance across workers");
    println!("   - Pipeline-Parallel uses greedy balancing for near-optimal distribution");
    println!("   - Layer-Wise is simplest but may cause imbalance with uneven layers");
    println!("   - Fault tolerance requires spare capacity on remaining workers");
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_layer_creation() {
        let layer = ModelLayer::new("test", 1_000_000, 3.81, 2_000_000);
        assert_eq!(layer.name, "test");
        assert_eq!(layer.param_count, 1_000_000);
        assert!((layer.memory_mb - 3.81).abs() < 1e-6);
        assert_eq!(layer.compute_flops, 2_000_000);
    }

    #[test]
    fn test_worker_creation() {
        let worker = Worker::new(0, 2048.0);
        assert_eq!(worker.id, 0);
        assert!((worker.capacity_mb - 2048.0).abs() < 1e-6);
        assert!((worker.available_mb - 2048.0).abs() < 1e-6);
        assert!(worker.assigned_shards.is_empty());
    }

    #[test]
    fn test_worker_assign_shard_success() {
        let mut worker = Worker::new(0, 100.0);
        assert!(worker.assign_shard(0, 50.0));
        assert_eq!(worker.assigned_shards.len(), 1);
        assert!((worker.available_mb - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_worker_assign_shard_insufficient_capacity() {
        let mut worker = Worker::new(0, 10.0);
        assert!(!worker.assign_shard(0, 20.0));
        assert!(worker.assigned_shards.is_empty());
    }

    #[test]
    fn test_worker_remove_shard() {
        let mut worker = Worker::new(0, 100.0);
        worker.assign_shard(0, 30.0);
        worker.remove_shard(0, 30.0);
        assert!(worker.assigned_shards.is_empty());
        assert!((worker.available_mb - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_worker_utilization() {
        let mut worker = Worker::new(0, 100.0);
        assert!((worker.utilization()).abs() < 1e-6);
        worker.assign_shard(0, 75.0);
        assert!((worker.utilization() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_worker_utilization_zero_capacity() {
        let worker = Worker::new(0, 0.0);
        assert!((worker.utilization()).abs() < 1e-6);
    }

    #[test]
    fn test_build_model() {
        let layers = build_model(6);
        assert_eq!(layers.len(), 6);
        // Each successive layer should have more params
        for i in 1..layers.len() {
            assert!(layers[i].param_count > layers[i - 1].param_count);
        }
    }

    #[test]
    fn test_shard_layer_wise() {
        let layers = build_model(8);
        let plan = shard_layer_wise(&layers, 4);
        assert_eq!(plan.strategy, ShardingStrategy::LayerWise);
        assert_eq!(plan.shards.len(), 4);
        // All layers should be assigned
        let assigned: usize = plan.shards.iter().map(|s| s.layer_indices.len()).sum();
        assert_eq!(assigned, 8);
    }

    #[test]
    fn test_shard_tensor_parallel() {
        let layers = build_model(4);
        let plan = shard_tensor_parallel(&layers, 2);
        assert_eq!(plan.strategy, ShardingStrategy::TensorParallel);
        // 4 layers x 2 workers = 8 shards
        assert_eq!(plan.shards.len(), 8);
        // Imbalance should be 1.0 (perfectly balanced)
        assert!((plan.load_imbalance(2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_shard_pipeline_parallel() {
        let layers = build_model(6);
        let plan = shard_pipeline_parallel(&layers, 3);
        assert_eq!(plan.strategy, ShardingStrategy::PipelineParallel);
        // All layers should be covered
        let mut all_indices: Vec<usize> = plan
            .shards
            .iter()
            .flat_map(|s| s.layer_indices.clone())
            .collect();
        all_indices.sort_unstable();
        assert_eq!(all_indices, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_sharding_plan_load_imbalance() {
        let plan = ShardingPlan {
            strategy: ShardingStrategy::LayerWise,
            shards: vec![
                Shard {
                    layer_indices: vec![0],
                    worker_id: 0,
                    memory_requirement: 10.0,
                },
                Shard {
                    layer_indices: vec![1],
                    worker_id: 1,
                    memory_requirement: 30.0,
                },
            ],
            total_memory: 40.0,
            max_worker_load: 30.0,
        };
        // Ideal = 40 / 2 = 20, imbalance = 30 / 20 = 1.5
        assert!((plan.load_imbalance(2) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_sharding_plan_load_imbalance_zero() {
        let plan = ShardingPlan {
            strategy: ShardingStrategy::LayerWise,
            shards: Vec::new(),
            total_memory: 0.0,
            max_worker_load: 0.0,
        };
        assert!((plan.load_imbalance(0)).abs() < 1e-6);
    }

    #[test]
    fn test_simulate_forward_pass() {
        let layers = build_model(4);
        let plan = shard_layer_wise(&layers, 2);
        let result = simulate_forward_pass(&plan, &layers, 42);
        assert_eq!(result.outputs.len(), plan.shards.len());
        assert!(result.total_latency_us > 0.0);
    }

    #[test]
    fn test_simulate_forward_pass_deterministic() {
        let layers = build_model(4);
        let plan = shard_layer_wise(&layers, 2);
        let r1 = simulate_forward_pass(&plan, &layers, 42);
        let r2 = simulate_forward_pass(&plan, &layers, 42);
        assert_eq!(r1.outputs, r2.outputs);
        assert!((r1.total_latency_us - r2.total_latency_us).abs() < 1e-6);
    }

    #[test]
    fn test_reassign_shards_after_failure() {
        let layers = build_model(4);
        let plan = shard_layer_wise(&layers, 2);

        let mut workers = vec![Worker::new(0, 2048.0), Worker::new(1, 2048.0)];
        for (idx, shard) in plan.shards.iter().enumerate() {
            workers[shard.worker_id].assign_shard(idx, shard.memory_requirement);
        }

        let reassignments = reassign_shards(&plan, 1, &mut workers);
        // All shards from worker 1 should be reassigned to worker 0
        assert!(!reassignments.is_empty());
        for (_, target) in &reassignments {
            assert_ne!(*target, 1);
        }
    }

    #[test]
    fn test_det_rand_deterministic() {
        let a = det_rand(42, 0);
        let b = det_rand(42, 0);
        assert!((a - b).abs() < 1e-15);
    }

    #[test]
    fn test_det_rand_range() {
        for i in 0..100 {
            let val = det_rand(42, i);
            assert!(val >= 0.0);
            assert!(val < 1.0);
        }
    }

    #[test]
    fn test_strategy_name() {
        assert_eq!(ShardingStrategy::LayerWise.name(), "Layer-Wise");
        assert_eq!(ShardingStrategy::TensorParallel.name(), "Tensor-Parallel");
        assert_eq!(
            ShardingStrategy::PipelineParallel.name(),
            "Pipeline-Parallel"
        );
    }
}
