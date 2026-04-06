#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
//! Autograd Gradient Clipping Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/cli-parity-v1.yaml
//! Demonstrates gradient clipping techniques for training stability using
//! entrenar's autograd API. Gradient clipping prevents exploding gradients
//! by bounding gradient magnitudes before the optimizer step.
//!
//! # Clipping Strategies
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                  Gradient Clipping Strategies                       │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  1. Global Norm Clipping                                            │
//! │     ‖g‖ = sqrt(Σ gᵢ²)                                             │
//! │     if ‖g‖ > max_norm: g ← g × (max_norm / ‖g‖)                  │
//! │                                                                     │
//! │  2. Per-Parameter Clipping                                          │
//! │     for each param p: clip(grad_p, -max_norm, max_norm)            │
//! │                                                                     │
//! │  3. Value Clipping                                                  │
//! │     gᵢ ← clamp(gᵢ, -max_val, max_val)                            │
//! │                                                                     │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  Forward ─► Loss ─► Backward ─► Clip Gradients ─► Optimizer Step   │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example autograd_gradient_clipping
//! ```
//!
//! # Recipe Metadata
//!
//! - **Category**: Training
//! - **Complexity**: Intermediate
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
use entrenar::autograd::Tensor;
use entrenar::optim::{AdamW, Optimizer};
use ndarray::Array1;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ── Model dimensions ──

pub const INPUT_DIM: usize = 8;
pub const HIDDEN_DIM: usize = 16;
pub const OUTPUT_DIM: usize = 4;
/// Gradient clipping strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClipStrategy {
    /// No clipping applied
    None,
    /// Global L2 norm clipping: scale all gradients if total norm exceeds threshold
    GlobalNorm(f32),
    /// Per-parameter norm clipping: clip each parameter independently
    PerParam(f32),
    /// Value clipping: clamp each gradient element to [-max_val, max_val]
    Value(f32),
}

impl ClipStrategy {
    pub fn label(self) -> String {
        match self {
            Self::None => "None".to_string(),
            Self::GlobalNorm(t) => format!("GlobalNorm({t})"),
            Self::PerParam(t) => format!("PerParam({t})"),
            Self::Value(t) => format!("Value({t})"),
        }
    }
}

/// Deterministic hash-based value
pub fn hash_f32(seed: u64, index: usize, label: &str) -> f32 {
    let mut h = DefaultHasher::new();
    (seed, label, index).hash(&mut h);
    h.finish() as f32 / u64::MAX as f32 - 0.5
}

/// Two-layer MLP backed by entrenar Tensors
pub struct ClipModel {
    /// pub w1: [HIDDEN_DIM x INPUT_DIM], pub b1: [HIDDEN_DIM],
    /// w2: [OUTPUT_DIM x HIDDEN_DIM], pub b2: [OUTPUT_DIM]
    pub params: Vec<Tensor>,
}

impl ClipModel {
    pub const W1: usize = 0;
    pub const B1: usize = 1;
    pub const W2: usize = 2;
    pub const B2: usize = 3;
    pub fn new(seed: u64) -> Self {
        let w1_scale = (2.0 / (INPUT_DIM + HIDDEN_DIM) as f32).sqrt();
        let w2_scale = (2.0 / (HIDDEN_DIM + OUTPUT_DIM) as f32).sqrt();

        let w1: Vec<f32> = (0..HIDDEN_DIM * INPUT_DIM)
            .map(|i| hash_f32(seed, i, "w1") * w1_scale)
            .collect();
        let w2: Vec<f32> = (0..OUTPUT_DIM * HIDDEN_DIM)
            .map(|i| hash_f32(seed, i, "w2") * w2_scale)
            .collect();

        let params = vec![
            Tensor::from_vec(w1, true),
            Tensor::zeros(HIDDEN_DIM, true),
            Tensor::from_vec(w2, true),
            Tensor::zeros(OUTPUT_DIM, true),
        ];

        Self { params }
    }

    /// Forward pass: input -> hidden (ReLU) -> logits
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let w1 = &self.params[Self::W1];
        let b1 = &self.params[Self::B1];
        let w2 = &self.params[Self::W2];
        let b2 = &self.params[Self::B2];

        // Hidden = ReLU(x @ W1^T + b1)
        let mut hidden = [0.0f32; HIDDEN_DIM];
        #[allow(clippy::needless_range_loop)]
        for j in 0..HIDDEN_DIM {
            let mut sum = b1.data()[j];
            for i in 0..INPUT_DIM {
                sum += x[i] * w1.data()[j * INPUT_DIM + i];
            }
            hidden[j] = sum.max(0.0);
        }

        // Output = hidden @ W2^T + b2
        let mut output = [0.0f32; OUTPUT_DIM];
        #[allow(clippy::needless_range_loop)]
        for k in 0..OUTPUT_DIM {
            let mut sum = b2.data()[k];
            for j in 0..HIDDEN_DIM {
                sum += hidden[j] * w2.data()[k * HIDDEN_DIM + j];
            }
            output[k] = sum;
        }

        output.to_vec()
    }

    /// Softmax cross-entropy loss
    pub fn loss(&self, logits: &[f32], target: usize) -> f32 {
        softmax_cross_entropy(logits, target)
    }

    pub fn param_count(&self) -> usize {
        self.params.iter().map(Tensor::len).sum()
    }

    pub fn params_mut(&mut self) -> &mut [Tensor] {
        &mut self.params
    }
}

/// Numerically stable softmax cross-entropy
pub fn softmax_cross_entropy(logits: &[f32], target: usize) -> f32 {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    -(exps[target] / sum).max(1e-10).ln()
}

/// Compute finite-difference gradients and set them on the model parameters.
///
/// This simulates backward() for demonstration purposes: it computes
/// the gradient of the loss w.r.t. each parameter using central differences.
pub fn compute_and_set_grads(model: &mut ClipModel, x: &[f32], target: usize) {
    let eps = 1e-4_f32;

    for param_idx in 0..model.params.len() {
        let n = model.params[param_idx].len();
        let mut grad = Array1::<f32>::zeros(n);

        for elem in 0..n {
            // +eps
            let orig = model.params[param_idx].data()[elem];
            // We need to modify the parameter in-place. Since Tensor doesn't expose
            // direct mutation of individual elements, we reconstruct. But for efficiency,
            // we use the same pattern as gradient_accumulation: modify the model weights
            // directly through a clone-modify-replace cycle.
            let mut data_plus = model.params[param_idx].data().to_vec();
            data_plus[elem] = orig + eps;
            let tmp = Tensor::from_vec(data_plus, true);
            let old = std::mem::replace(&mut model.params[param_idx], tmp);
            let logits_plus = model.forward(x);
            let loss_plus = model.loss(&logits_plus, target);

            // -eps
            let mut data_minus = old.data().to_vec();
            data_minus[elem] = orig - eps;
            let tmp2 = Tensor::from_vec(data_minus, true);
            let _ = std::mem::replace(&mut model.params[param_idx], tmp2);
            let logits_minus = model.forward(x);
            let loss_minus = model.loss(&logits_minus, target);

            // Restore original
            let mut data_orig = model.params[param_idx].data().to_vec();
            data_orig[elem] = orig;
            model.params[param_idx] = Tensor::from_vec(data_orig, true);

            grad[elem] = (loss_plus - loss_minus) / (2.0 * eps);
        }

        model.params[param_idx].set_grad(grad);
    }
}

/// Compute the global L2 norm of all parameter gradients
pub fn global_gradient_norm(params: &[Tensor]) -> f32 {
    let mut sum_sq = 0.0f32;
    for p in params {
        if let Some(g) = p.grad() {
            sum_sq += g.iter().map(|&v| v * v).sum::<f32>();
        }
    }
    sum_sq.sqrt()
}

/// Apply gradient clipping in place (modifies gradients stored in Tensors)
pub fn clip_gradients(params: &[Tensor], strategy: ClipStrategy) -> f32 {
    let pre_norm = global_gradient_norm(params);

    match strategy {
        ClipStrategy::None => {}
        ClipStrategy::GlobalNorm(max_norm) => {
            if pre_norm > max_norm {
                let scale = max_norm / pre_norm;
                for p in params {
                    if let Some(g) = p.grad() {
                        let clipped = g.mapv(|v| v * scale);
                        p.set_grad(clipped);
                    }
                }
            }
        }
        ClipStrategy::PerParam(max_norm) => {
            for p in params {
                if let Some(g) = p.grad() {
                    let pnorm = g.iter().map(|&v| v * v).sum::<f32>().sqrt();
                    if pnorm > max_norm {
                        let scale = max_norm / pnorm;
                        let clipped = g.mapv(|v| v * scale);
                        p.set_grad(clipped);
                    }
                }
            }
        }
        ClipStrategy::Value(max_val) => {
            for p in params {
                if let Some(g) = p.grad() {
                    let clipped = g.mapv(|v| v.clamp(-max_val, max_val));
                    p.set_grad(clipped);
                }
            }
        }
    }

    pre_norm
}
