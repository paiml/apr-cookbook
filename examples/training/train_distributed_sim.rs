//! # Recipe: Distributed Training Simulation
//!
//! **Category**: training
//! **CLI Equivalent**: `apr train --distributed --world-size 4`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example train_distributed_sim` exits 0
//! 2. [x] `cargo test --example train_distributed_sim` passes
//! 3. [x] Deterministic output (fixed sharding)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr train --distributed` in-process (no sockets)
//! 10. [x] Unit tests cover all-reduce, sharding, loss aggregation
//!
//! ## Learning Objective
//! Demonstrates data-parallel distributed training with a synchronous
//! all-reduce step. We shard a mini-batch across 4 simulated workers, each
//! computes a local gradient, then an all-reduce averages gradients before
//! the optimizer step. Mirrors the parameter-server / all-reduce pattern
//! `apr train --distributed` orchestrates.
//!
//! ## Run Command
//! ```bash
//! cargo run --example train_distributed_sim
//! ```
//!
//! ## References
//! - Li, M. et al. (2014). *Scaling Distributed Machine Learning with the Parameter Server*. OSDI. URL: https://www.usenix.org/conference/osdi14/technical-sessions/presentation/li_mu

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone)]
pub struct WorkerShard {
    pub worker_id: u32,
    pub samples: Vec<(f64, f64)>,
}

pub fn shard_dataset(all: &[(f64, f64)], world_size: u32) -> Vec<WorkerShard> {
    (0..world_size)
        .map(|wid| WorkerShard {
            worker_id: wid,
            samples: all
                .iter()
                .enumerate()
                .filter(|(i, _)| (*i as u32) % world_size == wid)
                .map(|(_, s)| *s)
                .collect(),
        })
        .collect()
}

/// Simulate one worker's local gradient computation.
pub fn local_grad(weight: f64, shard: &WorkerShard) -> f64 {
    if shard.samples.is_empty() {
        return 0.0;
    }
    let n = shard.samples.len() as f64;
    shard
        .samples
        .iter()
        .map(|(x, y)| (weight * x - y) * x)
        .sum::<f64>()
        / n
}

/// Synchronous all-reduce: average across all workers.
pub fn all_reduce_average(local_grads: &[f64]) -> f64 {
    if local_grads.is_empty() {
        return 0.0;
    }
    local_grads.iter().sum::<f64>() / local_grads.len() as f64
}

pub fn global_loss(weight: f64, all: &[(f64, f64)]) -> f64 {
    if all.is_empty() {
        return 0.0;
    }
    all.iter()
        .map(|(x, y)| 0.5 * (weight * x - y).powi(2))
        .sum::<f64>()
        / all.len() as f64
}

fn demo_dataset() -> Vec<(f64, f64)> {
    // y = 0.5 * x for x = 1..=16
    (1..=16u32)
        .map(|i| (f64::from(i), 0.5 * f64::from(i)))
        .collect()
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("train_distributed_sim")?;
    println!("=== Recipe: {} ===", ctx.name());

    let world_size = 4u32;
    let data = demo_dataset();
    let shards = shard_dataset(&data, world_size);

    let mut weight = 2.0f64;
    let lr = 0.01f64;
    let steps = 50u32;

    println!("World size: {}", world_size);
    println!("Dataset:    {} samples", data.len());
    println!(
        "Shards:     {:?}",
        shards.iter().map(|s| s.samples.len()).collect::<Vec<_>>()
    );
    println!("Initial w = {:.6}", weight);

    let mut history = Vec::new();
    for step in 0..steps {
        let local_grads: Vec<f64> = shards.iter().map(|s| local_grad(weight, s)).collect();
        let global_grad = all_reduce_average(&local_grads);
        weight -= lr * global_grad;
        if step % 10 == 0 || step == steps - 1 {
            let loss = global_loss(weight, &data);
            println!(
                "  step {:>3}  global_grad={:+.6}  w={:.6}  loss={:.6}",
                step, global_grad, weight, loss
            );
            history.push(json!({
                "step": step,
                "weight": weight,
                "loss": loss,
                "global_grad": global_grad,
            }));
        }
    }

    let report = json!({
        "recipe": ctx.name(),
        "world_size": world_size,
        "n_samples": data.len(),
        "n_steps": steps,
        "lr": lr,
        "final_weight": weight,
        "final_loss": global_loss(weight, &data),
        "history": history,
    });
    let out = ctx.path("train-distributed.json");
    std::fs::write(
        &out,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharding_covers_all_samples() {
        let data = demo_dataset();
        let shards = shard_dataset(&data, 4);
        let total: usize = shards.iter().map(|s| s.samples.len()).sum();
        assert_eq!(total, data.len());
    }

    #[test]
    fn sharding_is_balanced_when_divisible() {
        let data: Vec<(f64, f64)> = (0..12).map(|i| (i as f64, i as f64 * 0.5)).collect();
        let shards = shard_dataset(&data, 4);
        for s in &shards {
            assert_eq!(s.samples.len(), 3);
        }
    }

    #[test]
    fn all_reduce_is_mean() {
        assert_eq!(all_reduce_average(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(all_reduce_average(&[]), 0.0);
    }

    #[test]
    fn loss_decreases_over_steps() {
        let data = demo_dataset();
        let shards = shard_dataset(&data, 4);
        let mut w = 2.0;
        let initial_loss = global_loss(w, &data);
        for _ in 0..100 {
            let grads: Vec<f64> = shards.iter().map(|s| local_grad(w, s)).collect();
            w -= 0.005 * all_reduce_average(&grads);
        }
        let final_loss = global_loss(w, &data);
        assert!(final_loss < initial_loss);
    }

    #[test]
    fn empty_shard_contributes_zero_gradient() {
        let shard = WorkerShard {
            worker_id: 0,
            samples: vec![],
        };
        assert_eq!(local_grad(1.0, &shard), 0.0);
    }
}
