#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A single layer in the model architecture.
#[derive(Debug, Clone)]
pub struct ModelLayer {
    pub name: String,
    pub param_count: usize,
    pub memory_mb: f64,
    pub compute_flops: u64,
}

impl ModelLayer {
    pub fn new(name: &str, param_count: usize, memory_mb: f64, compute_flops: u64) -> Self {
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
pub struct Worker {
    pub id: usize,
    pub capacity_mb: f64,
    pub available_mb: f64,
    pub assigned_shards: Vec<usize>,
}

impl Worker {
    pub fn new(id: usize, capacity_mb: f64) -> Self {
        Self {
            id,
            capacity_mb,
            available_mb: capacity_mb,
            assigned_shards: Vec::new(),
        }
    }

    /// Attempt to assign a shard. Returns true if the worker has capacity.
    pub fn assign_shard(&mut self, shard_idx: usize, memory_mb: f64) -> bool {
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
    pub fn remove_shard(&mut self, shard_idx: usize, memory_mb: f64) {
        if let Some(pos) = self.assigned_shards.iter().position(|&s| s == shard_idx) {
            self.assigned_shards.remove(pos);
            self.available_mb = (self.available_mb + memory_mb).min(self.capacity_mb);
        }
    }

    /// Utilization as a fraction of total capacity.
    pub fn utilization(&self) -> f64 {
        if self.capacity_mb == 0.0 {
            return 0.0;
        }
        1.0 - self.available_mb / self.capacity_mb
    }
}

/// A shard representing a subset of the model assigned to a worker.
#[derive(Debug, Clone)]
pub struct Shard {
    pub layer_indices: Vec<usize>,
    pub worker_id: usize,
    pub memory_requirement: f64,
}

/// Strategy for partitioning the model across workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardingStrategy {
    LayerWise,
    TensorParallel,
    PipelineParallel,
}

impl ShardingStrategy {
    pub fn name(self) -> &'static str {
        match self {
            Self::LayerWise => "Layer-Wise",
            Self::TensorParallel => "Tensor-Parallel",
            Self::PipelineParallel => "Pipeline-Parallel",
        }
    }
}

/// A complete sharding plan describing how the model is distributed.
#[derive(Debug, Clone)]
pub struct ShardingPlan {
    pub strategy: ShardingStrategy,
    pub shards: Vec<Shard>,
    pub total_memory: f64,
    pub max_worker_load: f64,
}

impl ShardingPlan {
    /// Compute load imbalance: ratio of max worker load to ideal balanced load.
    pub fn load_imbalance(&self, num_workers: usize) -> f64 {
        if num_workers == 0 || self.total_memory == 0.0 {
            return 0.0;
        }
        let ideal = self.total_memory / num_workers as f64;
        self.max_worker_load / ideal
    }
}

/// Result of a simulated distributed forward pass.
#[derive(Debug)]
pub struct ForwardPassResult {
    /// Simulated output value per shard (used in tests for verification).
    #[allow(dead_code)]
    pub outputs: Vec<f64>,
    // Total latency in simulated microseconds.
    pub total_latency_us: f64,
    // Worker ID that was the pipeline bottleneck.
    pub bottleneck_worker: usize,
}

/// Deterministic hash-based pseudo-random value in [0, 1).
pub fn det_rand(seed: u64, index: u64) -> f64 {
    let mut h = DefaultHasher::new();
    (seed, index).hash(&mut h);
    h.finish() as f64 / u64::MAX as f64
}

/// Build a sample model architecture with the given number of layers.
pub fn build_model(num_layers: usize) -> Vec<ModelLayer> {
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
pub fn shard_layer_wise(layers: &[ModelLayer], num_workers: usize) -> ShardingPlan {
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
pub fn shard_tensor_parallel(layers: &[ModelLayer], num_workers: usize) -> ShardingPlan {
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
pub fn shard_pipeline_parallel(layers: &[ModelLayer], num_workers: usize) -> ShardingPlan {
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
pub fn simulate_forward_pass(
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
pub fn reassign_shards(
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
