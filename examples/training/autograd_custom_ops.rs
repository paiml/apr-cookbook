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
//! - **Dependencies**: entrenar 0.5+, aprender 0.25+, ndarray 0.16+
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

// =============================================================================
// Constants
// =============================================================================

/// sqrt(2/pi) used in the GELU approximation
const SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// Cubic coefficient in the GELU tanh approximation
const GELU_COEFF: f32 = 0.044_715;

/// Default Huber loss delta threshold
const HUBER_DELTA: f32 = 1.0;

/// Finite-difference epsilon for numerical gradient verification
const FD_EPSILON: f32 = 1e-4;

// =============================================================================
// Custom GELU Activation (BackwardOp)
// =============================================================================

/// Custom GELU activation with analytic backward pass.
///
/// Forward:  y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// Backward: dy/dx computed analytically via the chain rule.
struct GeluBackward {
    /// Saved input for gradient computation
    input: Array1<f32>,
    /// Gradient cell of the output tensor (receives upstream gradient)
    output_grad: Rc<RefCell<Option<Array1<f32>>>>,
    /// Gradient cell of the input tensor (where we write our gradient)
    input_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for GeluBackward {
    fn backward(&self) {
        let upstream = self.output_grad.borrow();
        let grad_output = match upstream.as_ref() {
            Some(g) => g.clone(),
            None => return,
        };

        let grad_input = compute_gelu_gradient(&self.input, &grad_output);

        let mut input_grad = self.input_grad.borrow_mut();
        match input_grad.as_mut() {
            Some(existing) => *existing = existing.clone() + &grad_input,
            None => *input_grad = Some(grad_input),
        }
    }
}

/// Compute the analytic GELU gradient for each element.
///
/// d/dx GELU(x) = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx
/// where z = sqrt(2/pi) * (x + 0.044715 * x^3)
///       dz/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
fn compute_gelu_gradient(input: &Array1<f32>, grad_output: &Array1<f32>) -> Array1<f32> {
    let n = input.len();
    let mut result = Array1::zeros(n);

    for i in 0..n {
        let x = input[i];
        let x_cubed = x * x * x;
        let z = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
        let tanh_z = z.tanh();
        let sech2_z = 1.0 - tanh_z * tanh_z;
        let dz_dx = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_COEFF * x * x);

        // d/dx GELU(x) = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx
        let local_grad = 0.5 * (1.0 + tanh_z) + 0.5 * x * sech2_z * dz_dx;
        result[i] = grad_output[i] * local_grad;
    }

    result
}

/// Apply custom GELU activation with tracked backward op.
fn custom_gelu(input: &Tensor) -> Tensor {
    let data = input.data();
    let n = data.len();
    let mut output_data = Array1::zeros(n);

    for i in 0..n {
        let x = data[i];
        let z = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        output_data[i] = 0.5 * x * (1.0 + z.tanh());
    }

    let mut output = Tensor::new(output_data, input.requires_grad());

    if input.requires_grad() {
        let op = Rc::new(GeluBackward {
            input: data.clone(),
            output_grad: output.grad_cell(),
            input_grad: input.grad_cell(),
        });
        output.set_backward_op(op);
    }

    output
}

// =============================================================================
// Custom Huber Loss (BackwardOp)
// =============================================================================

/// Custom Huber loss with analytic backward pass.
///
/// Forward:  L = 0.5 * (y - t)^2          if |y - t| <= delta
///           L = delta * (|y - t| - 0.5 * delta)  otherwise
///
/// Backward: dL/dy = (y - t)              if |y - t| <= delta
///           dL/dy = delta * sign(y - t)   otherwise
struct HuberBackward {
    /// Saved residuals (predictions - targets) for gradient computation
    residuals: Array1<f32>,
    /// Delta threshold
    delta: f32,
    /// Gradient cell of the loss tensor
    loss_grad: Rc<RefCell<Option<Array1<f32>>>>,
    /// Gradient cell of the predictions tensor
    pred_grad: Rc<RefCell<Option<Array1<f32>>>>,
    /// Previous backward op in the chain (for gradient propagation)
    prev_op: Option<Rc<dyn BackwardOp>>,
}

impl BackwardOp for HuberBackward {
    fn backward(&self) {
        let upstream = self.loss_grad.borrow();
        let grad_output = match upstream.as_ref() {
            Some(g) => g.clone(),
            None => Array1::ones(1),
        };

        let n = self.residuals.len();
        let mut grad_input = Array1::zeros(n);
        // Scale factor: upstream gradient is scalar (from loss sum)
        let scale = if grad_output.len() == 1 {
            grad_output[0]
        } else {
            1.0
        };

        for i in 0..n {
            let r = self.residuals[i];
            let abs_r = r.abs();
            grad_input[i] = if abs_r <= self.delta {
                scale * r / n as f32
            } else {
                scale * self.delta * r.signum() / n as f32
            };
        }

        let mut pred_grad = self.pred_grad.borrow_mut();
        match pred_grad.as_mut() {
            Some(existing) => *existing = existing.clone() + &grad_input,
            None => *pred_grad = Some(grad_input),
        }
        drop(pred_grad);

        // Chain to previous backward op (e.g., GELU backward)
        if let Some(ref prev) = self.prev_op {
            prev.backward();
        }
    }
}

/// Compute Huber loss between predictions and targets.
///
/// Returns a scalar loss tensor with the backward op attached.
fn custom_huber_loss(predictions: &Tensor, targets: &Array1<f32>, delta: f32) -> (Tensor, f32) {
    let pred_data = predictions.data();
    let n = pred_data.len();
    let residuals = pred_data - targets;

    let mut total_loss = 0.0f32;
    for i in 0..n {
        let abs_r = residuals[i].abs();
        total_loss += if abs_r <= delta {
            0.5 * residuals[i] * residuals[i]
        } else {
            delta * (abs_r - 0.5 * delta)
        };
    }
    let mean_loss = total_loss / n as f32;

    let mut loss_tensor = Tensor::from_vec(vec![mean_loss], predictions.requires_grad());

    if predictions.requires_grad() {
        let op = Rc::new(HuberBackward {
            residuals: residuals.clone(),
            delta,
            loss_grad: loss_tensor.grad_cell(),
            pred_grad: predictions.grad_cell(),
            prev_op: predictions.backward_op(),
        });
        loss_tensor.set_backward_op(op);
    }

    (loss_tensor, mean_loss)
}

// =============================================================================
// Numerical Gradient Verification
// =============================================================================

/// Compute numerical gradient of a scalar function via central differences.
///
/// For each element x_i, compute:
///   df/dx_i approx (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
fn numerical_gradient<F>(f: F, x: &Array1<f32>, eps: f32) -> Array1<f32>
where
    F: Fn(&Array1<f32>) -> f32,
{
    let n = x.len();
    let mut grad = Array1::zeros(n);

    for i in 0..n {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[i] += eps;
        x_minus[i] -= eps;

        grad[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * eps);
    }

    grad
}

/// Compute the GELU function on raw arrays (for numerical gradient checks).
fn gelu_scalar_fn(x: &Array1<f32>) -> Array1<f32> {
    let n = x.len();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        let v = x[i];
        let z = SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v);
        result[i] = 0.5 * v * (1.0 + z.tanh());
    }
    result
}

/// Compute the sum of GELU outputs (scalar function for numerical gradient).
fn gelu_sum(x: &Array1<f32>) -> f32 {
    gelu_scalar_fn(x).sum()
}

/// Compute the Huber loss on raw arrays (for numerical gradient checks).
fn huber_loss_fn(predictions: &Array1<f32>, targets: &Array1<f32>, delta: f32) -> f32 {
    let n = predictions.len();
    let mut total = 0.0f32;
    for i in 0..n {
        let r = predictions[i] - targets[i];
        let abs_r = r.abs();
        total += if abs_r <= delta {
            0.5 * r * r
        } else {
            delta * (abs_r - 0.5 * delta)
        };
    }
    total / n as f32
}

// =============================================================================
// Deterministic Data Generation
// =============================================================================

/// Generate deterministic f32 data using a hash-based approach.
fn deterministic_data(seed: u64, size: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..size)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (seed, "data", i).hash(&mut hasher);
            let h = hasher.finish();
            (h as f32 / u64::MAX as f32 - 0.5) * 2.0
        })
        .collect()
}

/// Generate deterministic target data.
fn deterministic_targets(seed: u64, size: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..size)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (seed, "target", i).hash(&mut hasher);
            let h = hasher.finish();
            (h as f32 / u64::MAX as f32 - 0.5) * 2.0
        })
        .collect()
}

// =============================================================================
// Main
// =============================================================================

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
mod tests {
    use super::*;

    #[test]
    fn test_gelu_zero() {
        let input = Tensor::from_vec(vec![0.0], false);
        let output = custom_gelu(&input);
        assert!(
            output.data()[0].abs() < 1e-6,
            "GELU(0) should be ~0, got {}",
            output.data()[0]
        );
    }

    #[test]
    fn test_gelu_positive_large() {
        let input = Tensor::from_vec(vec![10.0], false);
        let output = custom_gelu(&input);
        // For large positive x, GELU(x) ~ x
        assert!(
            (output.data()[0] - 10.0).abs() < 0.01,
            "GELU(10.0) should be ~10.0, got {}",
            output.data()[0]
        );
    }

    #[test]
    fn test_gelu_negative_large() {
        let input = Tensor::from_vec(vec![-10.0], false);
        let output = custom_gelu(&input);
        // For large negative x, GELU(x) ~ 0
        assert!(
            output.data()[0].abs() < 0.01,
            "GELU(-10.0) should be ~0, got {}",
            output.data()[0]
        );
    }

    #[test]
    fn test_gelu_matches_builtin() {
        let data = deterministic_data(100, 16);
        let tensor = Tensor::from_vec(data, false);
        let builtin = entrenar::autograd::gelu(&tensor);
        let custom = custom_gelu(&tensor);

        for i in 0..builtin.data().len() {
            let diff = (builtin.data()[i] - custom.data()[i]).abs();
            assert!(
                diff < 1e-5,
                "GELU mismatch at index {}: builtin={}, custom={}, diff={}",
                i,
                builtin.data()[i],
                custom.data()[i],
                diff
            );
        }
    }

    #[test]
    fn test_gelu_gradient_numerical_verification() {
        let input = Array1::from_vec(deterministic_data(101, 4));
        let ones = Array1::ones(input.len());
        let analytic = compute_gelu_gradient(&input, &ones);
        let numerical = numerical_gradient(gelu_sum, &input, FD_EPSILON);

        for i in 0..input.len() {
            let diff = (analytic[i] - numerical[i]).abs();
            assert!(
                diff < 1e-3,
                "GELU gradient mismatch at index {}: analytic={}, numerical={}, diff={}",
                i,
                analytic[i],
                numerical[i],
                diff
            );
        }
    }

    #[test]
    fn test_huber_loss_quadratic_region() {
        // When |residual| <= delta, Huber = 0.5 * r^2
        let preds = Tensor::from_vec(vec![0.3], false);
        let targets = Array1::from_vec(vec![0.0]);
        let (_, loss) = custom_huber_loss(&preds, &targets, 1.0);
        let expected = 0.5 * 0.3 * 0.3;
        assert!(
            (loss - expected).abs() < 1e-6,
            "Huber quadratic region: expected {}, got {}",
            expected,
            loss
        );
    }

    #[test]
    fn test_huber_loss_linear_region() {
        // When |residual| > delta, Huber = delta * (|r| - 0.5 * delta)
        let preds = Tensor::from_vec(vec![3.0], false);
        let targets = Array1::from_vec(vec![0.0]);
        let delta = 1.0;
        let (_, loss) = custom_huber_loss(&preds, &targets, delta);
        let expected = delta * (3.0 - 0.5 * delta);
        assert!(
            (loss - expected).abs() < 1e-6,
            "Huber linear region: expected {}, got {}",
            expected,
            loss
        );
    }

    #[test]
    fn test_huber_loss_gradient_numerical_verification() {
        let preds = Array1::from_vec(deterministic_data(102, 4));
        let targets = Array1::from_vec(deterministic_targets(103, 4));

        let numerical = numerical_gradient(
            |p| huber_loss_fn(p, &targets, HUBER_DELTA),
            &preds,
            FD_EPSILON,
        );

        let pred_tensor = Tensor::from_vec(preds.to_vec(), true);
        let (mut loss, _) = custom_huber_loss(&pred_tensor, &targets, HUBER_DELTA);
        backward(&mut loss, None);

        let analytic = pred_tensor
            .grad()
            .expect("Gradient should be computed after backward");

        for i in 0..preds.len() {
            let diff = (analytic[i] - numerical[i]).abs();
            assert!(
                diff < 1e-3,
                "Huber gradient mismatch at {}: analytic={}, numerical={}, diff={}",
                i,
                analytic[i],
                numerical[i],
                diff
            );
        }
    }

    #[test]
    fn test_huber_loss_zero_residual() {
        let preds = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        let targets = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let (_, loss) = custom_huber_loss(&preds, &targets, 1.0);
        assert!(
            loss.abs() < 1e-6,
            "Huber loss with zero residuals should be 0, got {}",
            loss
        );
    }

    #[test]
    fn test_training_convergence() {
        let dim = 4;
        let mut weights = Tensor::from_vec(deterministic_data(104, dim), true);
        let targets = Array1::from_vec(deterministic_targets(105, dim));
        let mut optimizer = AdamW::default_params(0.05);

        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;

        for step in 0..30 {
            optimizer.zero_grad_refs(&mut [&mut weights]);

            let activated = custom_gelu(&weights);
            let (mut loss, loss_val) = custom_huber_loss(&activated, &targets, HUBER_DELTA);

            if step == 0 {
                first_loss = loss_val;
            }
            last_loss = loss_val;

            backward(&mut loss, None);
            if let Some(g) = weights.grad() {
                weights.set_grad(g);
            }
            optimizer.step_refs(&mut [&mut weights]);
        }

        assert!(
            last_loss < first_loss,
            "Training should reduce loss: {} -> {}",
            first_loss,
            last_loss
        );
    }

    #[test]
    fn test_deterministic_data_reproducible() {
        let d1 = deterministic_data(42, 10);
        let d2 = deterministic_data(42, 10);
        assert_eq!(d1, d2, "Same seed should produce identical data");
    }

    #[test]
    fn test_deterministic_data_different_seeds() {
        let d1 = deterministic_data(42, 10);
        let d2 = deterministic_data(43, 10);
        assert_ne!(d1, d2, "Different seeds should produce different data");
    }

    #[test]
    fn test_custom_gelu_backward_fires() {
        let input = Tensor::from_vec(vec![1.0, -1.0, 0.5, -0.5], true);
        let output = custom_gelu(&input);

        // Set upstream gradient
        output.set_grad(Array1::ones(4));

        // Trigger backward
        if let Some(op) = output.backward_op() {
            op.backward();
        }

        let grad = input.grad();
        assert!(grad.is_some(), "Input should have gradient after backward");
        let g = grad.expect("just checked");
        assert_eq!(g.len(), 4);
        // GELU gradient at x=0.5 should be positive
        assert!(g[2] > 0.0, "GELU gradient at x=0.5 should be positive");
    }
}
