#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
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
pub const SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// Cubic coefficient in the GELU tanh approximation
pub const GELU_COEFF: f32 = 0.044_715;

/// Default Huber loss delta threshold
pub const HUBER_DELTA: f32 = 1.0;

/// Finite-difference epsilon for numerical gradient verification
pub const FD_EPSILON: f32 = 1e-4;

// =============================================================================
// Custom GELU Activation (BackwardOp)
// =============================================================================

// Custom GELU activation with analytic backward pass.
//
// Forward:  y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// Backward: dy/dx computed analytically via the chain rule.
pub struct GeluBackward {
    // Saved input for gradient computation
    pub input: Array1<f32>,
    // Gradient cell of the output tensor (receives upstream gradient)
    pub output_grad: Rc<RefCell<Option<Array1<f32>>>>,
    // Gradient cell of the input tensor (where we write our gradient)
    pub input_grad: Rc<RefCell<Option<Array1<f32>>>>,
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

// Compute the analytic GELU gradient for each element.
//
// d/dx GELU(x) = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx
// where z = sqrt(2/pi) * (x + 0.044715 * x^3)
///       dz/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
pub fn compute_gelu_gradient(input: &Array1<f32>, grad_output: &Array1<f32>) -> Array1<f32> {
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
pub fn custom_gelu(input: &Tensor) -> Tensor {
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

// Custom Huber loss with analytic backward pass.
//
// Forward:  L = 0.5 * (y - t)^2          if |y - t| <= delta
//           L = delta * (|y - t| - 0.5 * delta)  otherwise
//
// Backward: dL/dy = (y - t)              if |y - t| <= delta
///           dL/dy = delta * sign(y - t)   otherwise
pub struct HuberBackward {
    // Saved residuals (predictions - targets) for gradient computation
    pub residuals: Array1<f32>,
    // Delta threshold
    pub delta: f32,
    // Gradient cell of the loss tensor
    pub loss_grad: Rc<RefCell<Option<Array1<f32>>>>,
    // Gradient cell of the predictions tensor
    pub pred_grad: Rc<RefCell<Option<Array1<f32>>>>,
    // Previous backward op in the chain (for gradient propagation)
    pub prev_op: Option<Rc<dyn BackwardOp>>,
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

// Compute Huber loss between predictions and targets.
//
/// Returns a scalar loss tensor with the backward op attached.
pub fn custom_huber_loss(predictions: &Tensor, targets: &Array1<f32>, delta: f32) -> (Tensor, f32) {
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

// Compute numerical gradient of a scalar function via central differences.
//
// For each element x_i, compute:
///   df/dx_i approx (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
pub fn numerical_gradient<F>(f: F, x: &Array1<f32>, eps: f32) -> Array1<f32>
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
pub fn gelu_scalar_fn(x: &Array1<f32>) -> Array1<f32> {
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
pub fn gelu_sum(x: &Array1<f32>) -> f32 {
    gelu_scalar_fn(x).sum()
}

/// Compute the Huber loss on raw arrays (for numerical gradient checks).
pub fn huber_loss_fn(predictions: &Array1<f32>, targets: &Array1<f32>, delta: f32) -> f32 {
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
pub fn deterministic_data(seed: u64, size: usize) -> Vec<f32> {
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
pub fn deterministic_targets(seed: u64, size: usize) -> Vec<f32> {
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
