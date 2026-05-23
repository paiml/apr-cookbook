#![allow(unused_imports)]
//! Distributed Model Sharding Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml
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

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
