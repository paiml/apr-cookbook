//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use entrenar::autograd::Tensor;
use entrenar::optim::{Optimizer, SGD};
use ndarray::Array1;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---- Configuration ----------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NetworkConfig {
    pub layer_dims: Vec<usize>,
    pub lr: f32,
    pub seed: u64,
}

impl NetworkConfig {
    pub fn deep_network() -> Self {
        Self {
            layer_dims: vec![8, 12, 12, 12, 12, 12, 4],
            lr: 0.01,
            seed: 42,
        }
    }
    pub fn num_layers(&self) -> usize {
        self.layer_dims.len() - 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    ReLU,
    Sigmoid,
}

impl Activation {
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Self::ReLU => x.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        }
    }
    #[cfg(test)]
    pub fn derivative(self, x: f32) -> f32 {
        match self {
            Self::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            }
        }
    }
}

// ---- Gradient Statistics ----------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GradientStats {
    pub layer_name: String,
    pub mean: f32,
    pub std_dev: f32,
    pub max_abs: f32,
    pub min_abs: f32,
    pub norm: f32,
    pub pct_near_zero: f32,
    pub count: usize,
}

impl GradientStats {
    pub fn from_gradients(name: &str, grads: &[f32]) -> Self {
        let count = grads.len();
        if count == 0 {
            return Self {
                layer_name: name.to_string(),
                mean: 0.0,
                std_dev: 0.0,
                max_abs: 0.0,
                min_abs: 0.0,
                norm: 0.0,
                pct_near_zero: 100.0,
                count: 0,
            };
        }
        let n = count as f32;
        let mean = grads.iter().sum::<f32>() / n;
        let std_dev = (grads.iter().map(|g| (g - mean) * (g - mean)).sum::<f32>() / n).sqrt();
        let abs_vals: Vec<f32> = grads.iter().map(|g| g.abs()).collect();
        let max_abs = abs_vals.iter().copied().fold(0.0_f32, f32::max);
        let min_abs = abs_vals.iter().copied().fold(f32::MAX, f32::min);
        let norm = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        let pct_near_zero = 100.0 * abs_vals.iter().filter(|&&v| v < 1e-6).count() as f32 / n;
        Self {
            layer_name: name.to_string(),
            mean,
            std_dev,
            max_abs,
            min_abs,
            norm,
            pct_near_zero,
            count,
        }
    }
}

// ---- Gradient Health --------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientHealth {
    Healthy,
    Vanishing,
    Exploding,
}

impl GradientHealth {
    pub fn label(self) -> &'static str {
        match self {
            Self::Healthy => "HEALTHY",
            Self::Vanishing => "VANISHING",
            Self::Exploding => "EXPLODING",
        }
    }
}

pub const VANISHING_THRESHOLD: f32 = 1e-4;
pub const EXPLODING_THRESHOLD: f32 = 1e3;

pub fn diagnose_gradient_health(stats: &[GradientStats]) -> Vec<(String, GradientHealth)> {
    stats
        .iter()
        .map(|s| {
            let h = if s.norm < VANISHING_THRESHOLD {
                GradientHealth::Vanishing
            } else if s.norm > EXPLODING_THRESHOLD {
                GradientHealth::Exploding
            } else {
                GradientHealth::Healthy
            };
            (s.layer_name.clone(), h)
        })
        .collect()
}

// ---- Deep Network -----------------------------------------------------------

pub struct DeepNetwork {
    pub weights: Vec<Tensor>,
    pub biases: Vec<Tensor>,
    pub dims: Vec<usize>,
    pub activation: Activation,
}

impl DeepNetwork {
    pub fn new(config: &NetworkConfig, activation: Activation) -> Self {
        let mut weights = Vec::with_capacity(config.num_layers());
        let mut biases = Vec::with_capacity(config.num_layers());
        for li in 0..config.num_layers() {
            let (d_in, d_out) = (config.layer_dims[li], config.layer_dims[li + 1]);
            let scale = (2.0 / d_in as f32).sqrt();
            let w: Vec<f32> = (0..d_in * d_out)
                .map(|i| {
                    let mut h = DefaultHasher::new();
                    (config.seed, "w", li, i).hash(&mut h);
                    (h.finish() as f32 / u64::MAX as f32 - 0.5) * scale
                })
                .collect();
            weights.push(Tensor::from_vec(w, true));
            biases.push(Tensor::zeros(d_out, true));
        }
        Self {
            weights,
            biases,
            dims: config.layer_dims.clone(),
            activation,
        }
    }

    pub fn linear_forward(&self, current: &[f32], layer_idx: usize) -> Vec<f32> {
        let (d_in, d_out) = (self.dims[layer_idx], self.dims[layer_idx + 1]);
        let (w, b) = (
            self.weights[layer_idx].data(),
            self.biases[layer_idx].data(),
        );
        let mut out = vec![0.0_f32; d_out];
        #[allow(clippy::needless_range_loop)]
        for j in 0..d_out {
            let mut sum = b[j];
            for i in 0..d_in {
                sum += current[i] * w[i * d_out + j];
            }
            out[j] = sum;
        }
        out
    }

    pub fn forward_with_preactivations(&self, input: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut current = input.to_vec();
        let mut pre_acts = Vec::with_capacity(self.weights.len());
        for li in 0..self.weights.len() {
            let output = self.linear_forward(&current, li);
            pre_acts.push(output.clone());
            current = if li < self.weights.len() - 1 {
                output
                    .into_iter()
                    .map(|v| self.activation.apply(v))
                    .collect()
            } else {
                output
            };
        }
        (current, pre_acts)
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.forward_with_preactivations(input).0
    }

    pub fn cross_entropy_loss(&self, input: &[f32], target: usize) -> f32 {
        -softmax(&self.forward(input))[target].max(1e-10).ln()
    }

    pub fn compute_layer_gradients(&mut self, input: &[f32], target: usize) -> Vec<Vec<f32>> {
        let eps = 1e-4_f32;
        let base_loss = self.cross_entropy_loss(input, target);
        (0..self.weights.len())
            .map(|li| {
                (0..self.weights[li].len())
                    .map(|pi| {
                        let orig = self.weights[li].data().to_vec();
                        let mut perturbed = orig.clone();
                        perturbed[pi] += eps;
                        self.weights[li] = Tensor::from_vec(perturbed, true);
                        let grad = (self.cross_entropy_loss(input, target) - base_loss) / eps;
                        self.weights[li] = Tensor::from_vec(orig, true);
                        grad
                    })
                    .collect()
            })
            .collect()
    }

    pub fn apply_gradients(&self, layer_grads: &[Vec<f32>]) {
        for (i, grads) in layer_grads.iter().enumerate() {
            self.weights[i].set_grad(Array1::from_vec(grads.clone()));
        }
    }

    pub fn param_count(&self) -> usize {
        self.weights.iter().map(Tensor::len).sum::<usize>()
            + self.biases.iter().map(Tensor::len).sum::<usize>()
    }
}

// ---- Skip Network -----------------------------------------------------------

pub struct SkipNetwork {
    pub base: DeepNetwork,
}

impl SkipNetwork {
    pub fn new(config: &NetworkConfig, activation: Activation) -> Self {
        Self {
            base: DeepNetwork::new(config, activation),
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut current = input.to_vec();
        for li in 0..self.base.weights.len() {
            let mut output = self.base.linear_forward(&current, li);
            if li < self.base.weights.len() - 1 {
                for v in &mut output {
                    *v = self.base.activation.apply(*v);
                }
                if self.base.dims[li] == self.base.dims[li + 1] {
                    for (o, &c) in output.iter_mut().zip(current.iter()) {
                        *o += c;
                    }
                }
            }
            current = output;
        }
        current
    }

    pub fn cross_entropy_loss(&self, input: &[f32], target: usize) -> f32 {
        -softmax(&self.forward(input))[target].max(1e-10).ln()
    }

    pub fn compute_layer_gradients(&mut self, input: &[f32], target: usize) -> Vec<Vec<f32>> {
        let eps = 1e-4_f32;
        let base_loss = self.cross_entropy_loss(input, target);
        (0..self.base.weights.len())
            .map(|li| {
                (0..self.base.weights[li].len())
                    .map(|pi| {
                        let orig = self.base.weights[li].data().to_vec();
                        let mut p = orig.clone();
                        p[pi] += eps;
                        self.base.weights[li] = Tensor::from_vec(p, true);
                        let grad = (self.cross_entropy_loss(input, target) - base_loss) / eps;
                        self.base.weights[li] = Tensor::from_vec(orig, true);
                        grad
                    })
                    .collect()
            })
            .collect()
    }
}

// ---- Utilities --------------------------------------------------------------

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

pub fn generate_input(dim: usize, seed: u64, idx: usize) -> Vec<f32> {
    (0..dim)
        .map(|j| {
            let mut h = DefaultHasher::new();
            (seed, "input", idx, j).hash(&mut h);
            h.finish() as f32 / u64::MAX as f32 - 0.5
        })
        .collect()
}

pub fn batch_norm_proxy(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return vec![];
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let std = (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n + 1e-5).sqrt();
    values.iter().map(|v| (v - mean) / std).collect()
}

pub fn to_stats(grads: &[Vec<f32>]) -> Vec<GradientStats> {
    grads
        .iter()
        .enumerate()
        .map(|(i, g)| GradientStats::from_gradients(&format!("Layer {}", i + 1), g))
        .collect()
}

pub fn print_gradient_flow(stats: &[GradientStats]) {
    if stats.is_empty() {
        return;
    }
    let max_norm = stats
        .iter()
        .map(|s| s.norm)
        .fold(0.0_f32, f32::max)
        .max(1e-10);
    println!("   {:<10} {:>12} Bar", "Layer", "Norm");
    for s in stats {
        let f = (s.norm / max_norm).clamp(0.0, 1.0);
        let filled = (f * 40.0) as usize;
        let h = if s.norm < VANISHING_THRESHOLD {
            " [VANISHING]"
        } else if s.norm > EXPLODING_THRESHOLD {
            " [EXPLODING]"
        } else {
            ""
        };
        println!(
            "   {:<10} {:>12.6} |{}{}|{}",
            s.layer_name,
            s.norm,
            "#".repeat(filled),
            " ".repeat(40 - filled),
            h
        );
    }
}

// ---- Main -------------------------------------------------------------------
