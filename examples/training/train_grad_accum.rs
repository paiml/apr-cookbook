//! # Recipe: Mini Training Loop with Gradient Accumulation
//!
//! **Category**: training
//! **CLI Equivalent**: `apr train --grad-accum-steps 4`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example train_grad_accum` exits 0
//! 2. [x] `cargo test --example train_grad_accum` passes
//! 3. [x] Deterministic output (seeded updates)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr train --grad-accum` in-process
//! 10. [x] Unit tests cover grad averaging, step boundary, loss decrease
//!
//! ## Learning Objective
//! Demonstrates gradient accumulation for training large effective batches
//! on tight memory: accumulate k micro-batch gradients, divide by k, then
//! apply a single optimizer step. Mirrors the schedule `apr train` uses
//! when `--grad-accum-steps > 1`.
//!
//! ## Run Command
//! ```bash
//! cargo run --example train_grad_accum
//! ```
//!
//! ## References
//! - Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models*. arXiv:2203.15556

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone, Copy)]
pub struct TrainState {
    pub weight: f64,
    pub accum_grad: f64,
    pub accum_count: u32,
    pub last_loss: f64,
}

impl Default for TrainState {
    fn default() -> Self {
        Self {
            weight: 2.0,
            accum_grad: 0.0,
            accum_count: 0,
            last_loss: 0.0,
        }
    }
}

/// Compute loss and gradient for target w* = 0.5 via squared error.
pub fn loss_and_grad(state: &TrainState, x: f64, y: f64) -> (f64, f64) {
    let pred = state.weight * x;
    let diff = pred - y;
    let loss = 0.5 * diff * diff;
    let grad = diff * x;
    (loss, grad)
}

/// Apply one micro-batch: accumulate; if reached grad_accum_steps, step.
pub fn micro_step(state: &mut TrainState, x: f64, y: f64, lr: f64, grad_accum_steps: u32) -> bool {
    let (loss, grad) = loss_and_grad(state, x, y);
    state.accum_grad += grad;
    state.accum_count += 1;
    state.last_loss = loss;

    if state.accum_count >= grad_accum_steps {
        let avg_grad = state.accum_grad / f64::from(state.accum_count);
        state.weight -= lr * avg_grad;
        state.accum_grad = 0.0;
        state.accum_count = 0;
        return true;
    }
    false
}

fn demo_batch() -> Vec<(f64, f64)> {
    // Target relationship: y = 0.5 * x
    vec![
        (1.0, 0.5),
        (2.0, 1.0),
        (3.0, 1.5),
        (4.0, 2.0),
        (5.0, 2.5),
        (6.0, 3.0),
        (7.0, 3.5),
        (8.0, 4.0),
    ]
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("train_grad_accum")?;
    println!("=== Recipe: {} ===", ctx.name());

    let lr = 0.01;
    let grad_accum_steps = 4u32;
    let mut state = TrainState::default();
    let batch = demo_batch();

    println!("Initial w = {:.6}", state.weight);
    let mut step_log = Vec::new();
    for (i, (x, y)) in batch.iter().enumerate() {
        let applied = micro_step(&mut state, *x, *y, lr, grad_accum_steps);
        if applied {
            println!(
                "  step@micro_{}: loss={:.6} w={:.6}",
                i + 1,
                state.last_loss,
                state.weight
            );
            step_log.push(json!({
                "micro_idx": i + 1,
                "loss": state.last_loss,
                "weight": state.weight,
            }));
        }
    }
    println!("Final w = {:.6} (target 0.5)", state.weight);

    let report = json!({
        "recipe": ctx.name(),
        "lr": lr,
        "grad_accum_steps": grad_accum_steps,
        "n_micro_batches": batch.len(),
        "n_optimizer_steps": step_log.len(),
        "final_weight": state.weight,
        "target_weight": 0.5,
        "steps": step_log,
    });
    let out = ctx.path("train-grad-accum.json");
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
    fn micro_step_only_applies_at_boundary() {
        let mut s = TrainState::default();
        let applied = micro_step(&mut s, 1.0, 0.5, 0.1, 4);
        assert!(!applied);
        assert_eq!(s.accum_count, 1);
    }

    #[test]
    fn micro_step_applies_after_k_calls() {
        let mut s = TrainState::default();
        for _ in 0..3 {
            assert!(!micro_step(&mut s, 1.0, 0.5, 0.1, 4));
        }
        assert!(micro_step(&mut s, 1.0, 0.5, 0.1, 4));
        assert_eq!(s.accum_count, 0);
    }

    #[test]
    fn loss_decreases_toward_target() {
        let mut s = TrainState::default();
        let initial_loss = loss_and_grad(&s, 1.0, 0.5).0;
        for _ in 0..50 {
            for (x, y) in demo_batch() {
                micro_step(&mut s, x, y, 0.05, 4);
            }
        }
        let final_loss = loss_and_grad(&s, 1.0, 0.5).0;
        assert!(final_loss < initial_loss);
    }

    #[test]
    fn accum_grad_is_averaged() {
        let mut s1 = TrainState::default();
        let mut s2 = TrainState::default();
        // One step with k=2 should give same weight as averaging manually.
        micro_step(&mut s1, 1.0, 0.5, 0.1, 2);
        micro_step(&mut s1, 2.0, 1.0, 0.1, 2);
        // Equivalent single micro step with averaged gradient:
        let (_, g1) = loss_and_grad(&s2, 1.0, 0.5);
        let (_, g2) = loss_and_grad(&s2, 2.0, 1.0);
        s2.weight -= 0.1 * (g1 + g2) / 2.0;
        assert!((s1.weight - s2.weight).abs() < 1e-9);
    }

    #[test]
    fn deterministic_final_weight() {
        let mut s1 = TrainState::default();
        let mut s2 = TrainState::default();
        for _ in 0..10 {
            for (x, y) in demo_batch() {
                micro_step(&mut s1, x, y, 0.05, 4);
                micro_step(&mut s2, x, y, 0.05, 4);
            }
        }
        assert!((s1.weight - s2.weight).abs() < 1e-12);
    }
}
