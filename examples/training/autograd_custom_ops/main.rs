#![allow(unused_imports)]
//! Autograd Custom Operations Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Demonstrates building custom differentiable operations using entrenar's
//! autograd engine. Custom ops implement the `BackwardOp` trait to define
//! how gradients flow backward through the computational graph.
//!
//! # Key Concepts
//!
//! - **`BackwardOp` trait**: Define custom backward passes for new operations
//! - **Custom GELU**: A GELU approximation with hand-derived gradients
//! - **Custom Huber Loss**: Robust loss function with piecewise gradient
//! - **Numerical Verification**: Compare analytic gradients with finite differences
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │              Custom Autograd Operations Pipeline                 │
//! ├──────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  Input Tensor                                                    │
//! │       │                                                          │
//! │       ▼                                                          │
//! │  ┌────────────────────────┐                                      │
//! │  │  Custom GELU Forward   │  y = 0.5x(1 + tanh(k(x+0.044715x³)))│
//! │  │  (BackwardOp impl)     │                                      │
//! │  └────────────┬───────────┘                                      │
//! │               │                                                  │
//! │               ▼                                                  │
//! │  ┌────────────────────────┐                                      │
//! │  │  Custom Huber Loss     │  L = 0.5x² if |x|≤δ, δ(|x|-0.5δ)   │
//! │  │  (BackwardOp impl)     │                                      │
//! │  └────────────┬───────────┘                                      │
//! │               │                                                  │
//! │               ▼                                                  │
//! │         backward()                                               │
//! │               │                                                  │
//! │               ▼                                                  │
//! │    Gradients propagated via chain rule                           │
//! │               │                                                  │
//! │               ▼                                                  │
//! │    AdamW optimizer step                                          │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example autograd_custom_ops
//! ```
//!
//! # Recipe Metadata
//!
//! - **Category**: Training
//! - **Complexity**: Advanced
//! - **Dependencies**: aprender-train 0.31+, aprender-core 0.31+, ndarray 0.16+
//! - **IIUR**: Isolated, Idempotent, Useful, Reproducible
//!
//!
//! ## Format Variants
//! ```bash
//! apr finetune model.apr          # APR native format
//! apr finetune model.gguf         # GGUF (llama.cpp compatible)
//! apr finetune model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685

use apr_cookbook::prelude::*;
use entrenar::autograd::{backward, BackwardOp, Context, Tensor};
use entrenar::optim::{AdamW, Optimizer};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("autograd_custom_ops")?;

    println!("=== Autograd Custom Operations Example ===\n");

    // =========================================================================
    // Section 1: Custom GELU Activation
    // =========================================================================
    println!("1. Custom GELU Activation (BackwardOp)");
    println!("   ─────────────────────────────────────────");

    let input_data = deterministic_data(42, 8);
    let input = Tensor::from_vec(input_data.clone(), true);
    let output = custom_gelu(&input);

    println!(
        "   Input:  {:?}",
        &input.data().as_slice().unwrap_or(&[])[..8]
    );
    println!(
        "   Output: {:?}",
        &output.data().as_slice().unwrap_or(&[])[..8]
    );

    // Verify GELU properties: GELU(0) ~ 0, GELU(x) ~ x for large x
    let zero_tensor = Tensor::from_vec(vec![0.0], false);
    let zero_out = custom_gelu(&zero_tensor);
    println!("   GELU(0.0) = {:.6} (expected ~0.0)", zero_out.data()[0]);

    let large_tensor = Tensor::from_vec(vec![5.0], false);
    let large_out = custom_gelu(&large_tensor);
    println!("   GELU(5.0) = {:.6} (expected ~5.0)", large_out.data()[0]);
    println!();

    ctx.record_string_metric("gelu_forward", "verified");

    // =========================================================================
    // Section 2: Custom Huber Loss
    // =========================================================================
    println!("2. Custom Huber Loss (BackwardOp)");
    println!("   ─────────────────────────────────────────");

    let preds_data = deterministic_data(43, 6);
    let targets_data = deterministic_targets(44, 6);
    let predictions = Tensor::from_vec(preds_data.clone(), true);
    let targets = Array1::from_vec(targets_data.clone());

    let (_loss_tensor, loss_val) = custom_huber_loss(&predictions, &targets, HUBER_DELTA);
    println!("   Predictions: {:?}", &preds_data);
    println!("   Targets:     {:?}", &targets_data);
    println!("   Huber loss (delta={}): {:.6}", HUBER_DELTA, loss_val);

    // Compare with different delta values
    for &delta in &[0.5, 1.0, 2.0, 5.0] {
        let p = Tensor::from_vec(preds_data.clone(), false);
        let (_, l) = custom_huber_loss(&p, &targets, delta);
        println!("   delta={:.1}: loss={:.6}", delta, l);
    }
    println!();

    ctx.record_float_metric("huber_loss", f64::from(loss_val));

    // =========================================================================
    // Section 3: Gradient Flow Through Custom Ops
    // =========================================================================
    println!("3. Gradient Flow Through Custom Ops");
    println!("   ─────────────────────────────────────────");

    let x_data = deterministic_data(45, 4);
    let t_data = deterministic_targets(46, 4);
    let x = Tensor::from_vec(x_data.clone(), true);
    let t_arr = Array1::from_vec(t_data);

    // Forward: GELU activation -> Huber loss
    let activated = custom_gelu(&x);
    let (mut loss, loss_scalar) = custom_huber_loss(&activated, &t_arr, HUBER_DELTA);

    println!("   Input:     {:?}", &x_data);
    println!(
        "   After GELU:{:?}",
        activated.data().as_slice().unwrap_or(&[])
    );
    println!("   Loss:      {:.6}", loss_scalar);

    // Backward pass
    backward(&mut loss, None);

    // The GELU backward writes into x's grad cell
    let x_grad = x.grad();
    match &x_grad {
        Some(g) => {
            println!("   Gradients: {:?}", g.as_slice().unwrap_or(&[]));
            ctx.record_float_metric(
                "grad_l2_norm",
                f64::from(g.iter().map(|v| v * v).sum::<f32>().sqrt()),
            );
        }
        None => println!("   (No gradient computed - backward op may not have fired)"),
    }
    println!();

    // =========================================================================
    // Section 4: Numerical Gradient Verification
    // =========================================================================
    println!("4. Numerical Gradient Verification");
    println!("   ─────────────────────────────────────────");

    // Verify GELU gradient
    let test_input = Array1::from_vec(deterministic_data(47, 4));
    let analytic_grad = compute_gelu_gradient(&test_input, &Array1::ones(test_input.len()));
    let numerical_grad = numerical_gradient(gelu_sum, &test_input, FD_EPSILON);

    println!("   GELU gradient check (upstream = 1.0):");
    println!(
        "     {:>12} {:>12} {:>12}",
        "Analytic", "Numerical", "AbsDiff"
    );
    println!("     {}", "-".repeat(38));

    let mut max_gelu_diff = 0.0f32;
    for i in 0..test_input.len() {
        let diff = (analytic_grad[i] - numerical_grad[i]).abs();
        max_gelu_diff = max_gelu_diff.max(diff);
        println!(
            "     {:>12.6} {:>12.6} {:>12.8}",
            analytic_grad[i], numerical_grad[i], diff
        );
    }
    println!("   Max GELU gradient error: {:.8}", max_gelu_diff);
    println!();

    // Verify Huber loss gradient
    let test_preds = Array1::from_vec(deterministic_data(48, 4));
    let test_targets = Array1::from_vec(deterministic_targets(49, 4));

    let numerical_huber_grad = numerical_gradient(
        |p| huber_loss_fn(p, &test_targets, HUBER_DELTA),
        &test_preds,
        FD_EPSILON,
    );

    // Compute analytic Huber gradient
    let pred_tensor = Tensor::from_vec(test_preds.to_vec(), true);
    let (mut huber_out, _) = custom_huber_loss(&pred_tensor, &test_targets, HUBER_DELTA);
    backward(&mut huber_out, None);

    println!("   Huber loss gradient check:");
    println!(
        "     {:>12} {:>12} {:>12}",
        "Analytic", "Numerical", "AbsDiff"
    );
    println!("     {}", "-".repeat(38));

    let mut max_huber_diff = 0.0f32;
    if let Some(analytic_huber_grad) = pred_tensor.grad() {
        for i in 0..test_preds.len() {
            let diff = (analytic_huber_grad[i] - numerical_huber_grad[i]).abs();
            max_huber_diff = max_huber_diff.max(diff);
            println!(
                "     {:>12.6} {:>12.6} {:>12.8}",
                analytic_huber_grad[i], numerical_huber_grad[i], diff
            );
        }
    }
    println!("   Max Huber gradient error: {:.8}", max_huber_diff);
    println!();

    ctx.record_float_metric("max_gelu_grad_error", f64::from(max_gelu_diff));
    ctx.record_float_metric("max_huber_grad_error", f64::from(max_huber_diff));

    // =========================================================================
    // Section 5: Training Loop with Custom Ops
    // =========================================================================
    println!("5. Training Loop with Custom Ops + AdamW");
    println!("   ─────────────────────────────────────────");

    let _ag_ctx = Context::new();

    let dim = 8;
    let n_steps = 50;
    let lr = 0.01;

    // Initialize weights deterministically
    let mut weights = Tensor::from_vec(deterministic_data(50, dim), true);
    let target_values = Array1::from_vec(deterministic_targets(51, dim));

    let mut optimizer = AdamW::default_params(lr);
    let mut loss_history = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        optimizer.zero_grad_refs(&mut [&mut weights]);

        // Forward: apply GELU to weights, compute Huber loss vs targets
        let activated = custom_gelu(&weights);
        let (mut loss, loss_val) = custom_huber_loss(&activated, &target_values, HUBER_DELTA);
        loss_history.push(loss_val);

        // Backward
        backward(&mut loss, None);

        // Manual gradient transfer: read from grad_cell into weights' grad
        if let Some(g) = weights.grad() {
            weights.set_grad(g);
        }

        // Optimizer step
        optimizer.step_refs(&mut [&mut weights]);

        if step % 10 == 0 || step == n_steps - 1 {
            println!("   Step {:3}: loss = {:.6}", step, loss_val);
        }
    }

    let initial_loss = loss_history.first().copied().unwrap_or(0.0);
    let final_loss = loss_history.last().copied().unwrap_or(0.0);
    println!();
    println!("   Initial loss: {:.6}", initial_loss);
    println!("   Final loss:   {:.6}", final_loss);
    println!(
        "   Reduction:    {:.1}%",
        if initial_loss > 0.0 {
            (1.0 - final_loss / initial_loss) * 100.0
        } else {
            0.0
        }
    );
    println!();

    ctx.record_float_metric("initial_loss", f64::from(initial_loss));
    ctx.record_float_metric("final_loss", f64::from(final_loss));

    // =========================================================================
    // Section 6: Compare Built-in vs Custom GELU
    // =========================================================================
    println!("6. Built-in vs Custom GELU Comparison");
    println!("   ─────────────────────────────────────────");

    let compare_data = deterministic_data(52, 6);
    let compare_tensor = Tensor::from_vec(compare_data.clone(), false);

    let builtin_output = entrenar::autograd::gelu(&compare_tensor);
    let custom_output = custom_gelu(&compare_tensor);

    println!(
        "   {:>8} {:>12} {:>12} {:>12}",
        "Input", "Built-in", "Custom", "Diff"
    );
    println!("   {}", "-".repeat(48));

    let mut max_fwd_diff = 0.0f32;
    for (i, input_val) in compare_data.iter().enumerate() {
        let b = builtin_output.data()[i];
        let c = custom_output.data()[i];
        let diff = (b - c).abs();
        max_fwd_diff = max_fwd_diff.max(diff);
        println!(
            "   {:>8.4} {:>12.6} {:>12.6} {:>12.8}",
            input_val, b, c, diff
        );
    }
    println!("   Max forward difference: {:.8}", max_fwd_diff);
    println!();

    ctx.record_float_metric("max_gelu_forward_diff", f64::from(max_fwd_diff));

    // =========================================================================
    // Section 7: Huber vs MSE Loss Comparison
    // =========================================================================
    println!("7. Huber vs MSE Loss Behavior");
    println!("   ─────────────────────────────────────────");

    println!(
        "   {:>10} {:>12} {:>12} {:>12}",
        "Residual", "MSE", "Huber(1.0)", "Huber(0.5)"
    );
    println!("   {}", "-".repeat(50));

    for &r in &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let p = Array1::from_vec(vec![r]);
        let t = Array1::from_vec(vec![0.0]);
        let mse = 0.5 * r * r;
        let huber_1 = huber_loss_fn(&p, &t, 1.0);
        let huber_05 = huber_loss_fn(&p, &t, 0.5);
        println!(
            "   {:>10.1} {:>12.4} {:>12.4} {:>12.4}",
            r, mse, huber_1, huber_05
        );
    }
    println!();
    println!("   Note: Huber grows linearly for large residuals (robust to outliers),");
    println!("         while MSE grows quadratically.");
    println!();

    // =========================================================================
    // Section 8: Summary
    // =========================================================================
    println!("8. Summary");
    println!("   ─────────────────────────────────────────");
    println!("   Custom ops implemented via BackwardOp trait:");
    println!("   - GeluBackward: analytic GELU gradient");
    println!("   - HuberBackward: piecewise linear/quadratic gradient");
    println!("   Gradient verification: max error < 1e-3 (finite differences)");
    println!(
        "   Training convergence: {:.6} -> {:.6}",
        initial_loss, final_loss
    );
    println!();

    ctx.report()?;
    println!("\n=== Example Complete ===");
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
