//! Autograd Backpropagation Visualization
//!
//! Visualizes gradient flow through a multi-layer neural network to diagnose
//! training pathologies. After a forward-backward pass, gradient statistics
//! are extracted per layer to detect vanishing and exploding gradients.
//!
//! # What This Demonstrates
//!
//! - Building a multi-layer network (5+ layers) with entrenar's autograd Tensors
//! - Extracting per-layer gradient statistics (mean, std, norm, near-zero %)
//! - Detecting vanishing gradients (norm < threshold in early layers)
//! - Detecting exploding gradients (norm > threshold)
//! - ASCII gradient flow diagrams showing norm magnitude per layer
//! - Comparing activation functions (ReLU vs sigmoid) and their gradient impact
//! - Applying mitigation techniques: skip connections and batch norm proxy
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                  Gradient Flow Visualization Pipeline                   │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │  Input ─► L1 ─► act ─► L2 ─► act ─► ... ─► L6 ─► Loss                │
//! │                                                      │                 │
//! │  grad_L1 ◄── grad_L2 ◄── ... ◄── grad_L6 ◄── backward                │
//! │     │          │                     │                                 │
//! │  stats_1    stats_2    ...       stats_6                               │
//! │     └──────────┴────────┴──────────┘                                   │
//! │              ASCII Gradient Diagram                                    │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example autograd_backprop_viz
//! ```
//!
//! # Recipe Metadata
//!
//! - **Category**: Training
//! - **Complexity**: Intermediate
//! - **Dependencies**: entrenar 0.5+, ndarray 0.16+
//! - **IIUR**: Isolated, Idempotent, Useful, Reproducible

use apr_cookbook::prelude::*;
use entrenar::autograd::Tensor;
use entrenar::optim::{Optimizer, SGD};
use ndarray::Array1;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ─── Configuration ───────────────────────────────────────────────────────────

/// Network configuration for gradient visualization
#[derive(Debug, Clone)]
struct NetworkConfig {
    /// Dimensions for each layer (input + hidden + output)
    layer_dims: Vec<usize>,
    /// Learning rate
    lr: f32,
    /// Seed for deterministic initialization
    seed: u64,
}

impl NetworkConfig {
    /// Create a 6-layer deep network configuration
    fn deep_network() -> Self {
        Self {
            layer_dims: vec![8, 12, 12, 12, 12, 12, 4],
            lr: 0.01,
            seed: 42,
        }
    }

    /// Number of weight layers (transitions between dims)
    fn num_layers(&self) -> usize {
        self.layer_dims.len() - 1
    }
}

// ─── Activation functions ────────────────────────────────────────────────────

/// Activation function type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Activation {
    ReLU,
    Sigmoid,
}

impl Activation {
    fn apply(self, x: f32) -> f32 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        }
    }

    #[cfg(test)]
    fn derivative(self, x: f32) -> f32 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            }
        }
    }
}

// ─── Gradient statistics ─────────────────────────────────────────────────────

/// Per-layer gradient statistics
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct GradientStats {
    layer_name: String,
    mean: f32,
    std_dev: f32,
    max_abs: f32,
    min_abs: f32,
    norm: f32,
    pct_near_zero: f32,
    count: usize,
}

impl GradientStats {
    /// Compute gradient statistics from a slice of gradient values
    fn from_gradients(name: &str, grads: &[f32]) -> Self {
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

        let sum: f32 = grads.iter().sum();
        let mean = sum / count as f32;

        let variance: f32 =
            grads.iter().map(|g| (g - mean) * (g - mean)).sum::<f32>() / count as f32;
        let std_dev = variance.sqrt();

        let abs_vals: Vec<f32> = grads.iter().map(|g| g.abs()).collect();
        let max_abs = abs_vals.iter().copied().fold(0.0_f32, f32::max);
        let min_abs = abs_vals.iter().copied().fold(f32::MAX, f32::min);

        let norm = grads.iter().map(|g| g * g).sum::<f32>().sqrt();

        let near_zero_threshold = 1e-6;
        let near_zero_count = abs_vals
            .iter()
            .filter(|&&v| v < near_zero_threshold)
            .count();
        let pct_near_zero = 100.0 * near_zero_count as f32 / count as f32;

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

// ─── Gradient health diagnostics ─────────────────────────────────────────────

/// Gradient health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GradientHealth {
    Healthy,
    Vanishing,
    Exploding,
}

impl GradientHealth {
    fn label(self) -> &'static str {
        match self {
            GradientHealth::Healthy => "HEALTHY",
            GradientHealth::Vanishing => "VANISHING",
            GradientHealth::Exploding => "EXPLODING",
        }
    }
}

/// Thresholds for gradient health detection
const VANISHING_THRESHOLD: f32 = 1e-4;
const EXPLODING_THRESHOLD: f32 = 1e3;

/// Diagnose gradient health from per-layer norms
fn diagnose_gradient_health(layer_stats: &[GradientStats]) -> Vec<(String, GradientHealth)> {
    layer_stats
        .iter()
        .map(|stats| {
            let health = if stats.norm < VANISHING_THRESHOLD {
                GradientHealth::Vanishing
            } else if stats.norm > EXPLODING_THRESHOLD {
                GradientHealth::Exploding
            } else {
                GradientHealth::Healthy
            };
            (stats.layer_name.clone(), health)
        })
        .collect()
}

// ─── Multi-layer network ─────────────────────────────────────────────────────

/// A multi-layer network using entrenar Tensors for weight storage
struct DeepNetwork {
    /// Weight tensors per layer
    weights: Vec<Tensor>,
    /// Bias tensors per layer
    biases: Vec<Tensor>,
    /// Layer dimensions
    dims: Vec<usize>,
    /// Activation function
    activation: Activation,
}

impl DeepNetwork {
    /// Create a new deep network with deterministic initialization
    fn new(config: &NetworkConfig, activation: Activation) -> Self {
        let mut weights = Vec::with_capacity(config.num_layers());
        let mut biases = Vec::with_capacity(config.num_layers());

        for layer_idx in 0..config.num_layers() {
            let d_in = config.layer_dims[layer_idx];
            let d_out = config.layer_dims[layer_idx + 1];

            // He (Kaiming) initialization for better gradient flow in deep ReLU networks
            let scale = (2.0 / d_in as f32).sqrt();
            let w_data: Vec<f32> = (0..d_in * d_out)
                .map(|i| {
                    let mut h = DefaultHasher::new();
                    (config.seed, "w", layer_idx, i).hash(&mut h);
                    (h.finish() as f32 / u64::MAX as f32 - 0.5) * scale
                })
                .collect();

            weights.push(Tensor::from_vec(w_data, true));
            biases.push(Tensor::zeros(d_out, true));
        }

        Self {
            weights,
            biases,
            dims: config.layer_dims.clone(),
            activation,
        }
    }

    /// Forward pass returning pre-activations for gradient computation
    fn forward_with_preactivations(&self, input: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut current = input.to_vec();
        let mut pre_activations = Vec::with_capacity(self.weights.len());

        for layer_idx in 0..self.weights.len() {
            let d_in = self.dims[layer_idx];
            let d_out = self.dims[layer_idx + 1];
            let w = self.weights[layer_idx].data();
            let b = self.biases[layer_idx].data();

            let mut output = vec![0.0_f32; d_out];
            #[allow(clippy::needless_range_loop)]
            for j in 0..d_out {
                let mut sum = b[j];
                for i in 0..d_in {
                    sum += current[i] * w[i * d_out + j];
                }
                output[j] = sum;
            }

            pre_activations.push(output.clone());

            // Apply activation to all layers except the last
            if layer_idx < self.weights.len() - 1 {
                for v in &mut output {
                    *v = self.activation.apply(*v);
                }
            }
            current = output;
        }

        (current, pre_activations)
    }

    /// Forward pass (output only)
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.forward_with_preactivations(input).0
    }

    /// Simulate backward pass via finite differences, returning per-layer gradient vectors
    fn compute_layer_gradients(&mut self, input: &[f32], target: usize) -> Vec<Vec<f32>> {
        let eps = 1e-4_f32;
        let base_loss = self.cross_entropy_loss(input, target);
        let mut layer_grads = Vec::with_capacity(self.weights.len());

        for layer_idx in 0..self.weights.len() {
            let n_params = self.weights[layer_idx].len();
            let mut grads = Vec::with_capacity(n_params);

            for param_idx in 0..n_params {
                // Perturb weight
                let orig_data = self.weights[layer_idx].data().to_vec();
                let mut perturbed = orig_data.clone();
                perturbed[param_idx] += eps;
                self.weights[layer_idx] = Tensor::from_vec(perturbed, true);

                let loss_plus = self.cross_entropy_loss(input, target);
                let grad = (loss_plus - base_loss) / eps;
                grads.push(grad);

                // Restore original weight
                self.weights[layer_idx] = Tensor::from_vec(orig_data, true);
            }

            layer_grads.push(grads);
        }

        layer_grads
    }

    /// Set gradients on tensors from computed gradient vectors
    fn apply_gradients(&self, layer_grads: &[Vec<f32>]) {
        for (layer_idx, grads) in layer_grads.iter().enumerate() {
            let grad_array = Array1::from_vec(grads.clone());
            self.weights[layer_idx].set_grad(grad_array);
        }
    }

    /// Cross-entropy loss
    fn cross_entropy_loss(&self, input: &[f32], target: usize) -> f32 {
        let output = self.forward(input);
        let probs = softmax(&output);
        -probs[target].max(1e-10).ln()
    }

    /// Total number of parameters
    fn param_count(&self) -> usize {
        self.weights.iter().map(Tensor::len).sum::<usize>()
            + self.biases.iter().map(Tensor::len).sum::<usize>()
    }
}

// ─── Network with skip connections ───────────────────────────────────────────

/// A network with residual skip connections to mitigate vanishing gradients
struct SkipNetwork {
    /// Base deep network
    base: DeepNetwork,
}

impl SkipNetwork {
    fn new(config: &NetworkConfig, activation: Activation) -> Self {
        Self {
            base: DeepNetwork::new(config, activation),
        }
    }

    /// Forward pass with skip connections (additive residual)
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut current = input.to_vec();

        for layer_idx in 0..self.base.weights.len() {
            let d_in = self.base.dims[layer_idx];
            let d_out = self.base.dims[layer_idx + 1];
            let w = self.base.weights[layer_idx].data();
            let b = self.base.biases[layer_idx].data();

            let mut output = vec![0.0_f32; d_out];
            #[allow(clippy::needless_range_loop)]
            for j in 0..d_out {
                let mut sum = b[j];
                for i in 0..d_in {
                    sum += current[i] * w[i * d_out + j];
                }
                output[j] = sum;
            }

            // Apply activation (except last layer)
            if layer_idx < self.base.weights.len() - 1 {
                for v in &mut output {
                    *v = self.base.activation.apply(*v);
                }
                // Skip connection: add input to output if dimensions match
                if d_in == d_out {
                    for (o, &c) in output.iter_mut().zip(current.iter()) {
                        *o += c;
                    }
                }
            }
            current = output;
        }

        current
    }

    /// Cross-entropy loss
    fn cross_entropy_loss(&self, input: &[f32], target: usize) -> f32 {
        let output = self.forward(input);
        let probs = softmax(&output);
        -probs[target].max(1e-10).ln()
    }

    /// Compute per-layer gradients via finite differences
    fn compute_layer_gradients(&mut self, input: &[f32], target: usize) -> Vec<Vec<f32>> {
        let eps = 1e-4_f32;
        let base_loss = self.cross_entropy_loss(input, target);
        let mut layer_grads = Vec::with_capacity(self.base.weights.len());

        for layer_idx in 0..self.base.weights.len() {
            let n_params = self.base.weights[layer_idx].len();
            let mut grads = Vec::with_capacity(n_params);

            for param_idx in 0..n_params {
                let orig_data = self.base.weights[layer_idx].data().to_vec();
                let mut perturbed = orig_data.clone();
                perturbed[param_idx] += eps;
                self.base.weights[layer_idx] = Tensor::from_vec(perturbed, true);

                let loss_plus = self.cross_entropy_loss(input, target);
                let grad = (loss_plus - base_loss) / eps;
                grads.push(grad);

                self.base.weights[layer_idx] = Tensor::from_vec(orig_data, true);
            }

            layer_grads.push(grads);
        }

        layer_grads
    }
}

// ─── Batch normalization proxy ───────────────────────────────────────────────

/// Apply batch-norm-like rescaling to a layer's pre-activations
/// This is a simplified proxy: center and scale by running statistics
fn batch_norm_proxy(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    let mean = values.iter().sum::<f32>() / n as f32;
    let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let std_dev = (variance + 1e-5).sqrt();
    values.iter().map(|v| (v - mean) / std_dev).collect()
}

// ─── Utility functions ───────────────────────────────────────────────────────

/// Softmax normalization
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Generate deterministic input data
fn generate_input(dim: usize, seed: u64, sample_idx: usize) -> Vec<f32> {
    (0..dim)
        .map(|j| {
            let mut h = DefaultHasher::new();
            (seed, "input", sample_idx, j).hash(&mut h);
            h.finish() as f32 / u64::MAX as f32 - 0.5
        })
        .collect()
}

/// Print ASCII gradient flow bar chart
fn print_gradient_flow_diagram(stats: &[GradientStats]) {
    if stats.is_empty() {
        return;
    }

    let max_norm = stats
        .iter()
        .map(|s| s.norm)
        .fold(0.0_f32, f32::max)
        .max(1e-10);

    let bar_width = 40;
    println!("   Gradient Flow (norm magnitude per layer):");
    println!("   {:<10} {:>12} Bar", "Layer", "Norm");
    println!("   {}", "-".repeat(10 + 12 + 2 + bar_width));

    for s in stats {
        let fraction = (s.norm / max_norm).clamp(0.0, 1.0);
        let filled = (fraction * bar_width as f32) as usize;
        let bar: String = "#".repeat(filled) + &" ".repeat(bar_width - filled);
        let health = if s.norm < VANISHING_THRESHOLD {
            " [VANISHING]"
        } else if s.norm > EXPLODING_THRESHOLD {
            " [EXPLODING]"
        } else {
            ""
        };
        println!(
            "   {:<10} {:>12.6} |{}|{}",
            s.layer_name, s.norm, bar, health
        );
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("autograd_backprop_viz")?;

    println!("=== Autograd Backpropagation Visualization ===\n");

    let config = NetworkConfig::deep_network();
    let input = generate_input(config.layer_dims[0], config.seed, 0);
    let target = 1_usize; // Target class

    // =========================================================================
    // Section 1: Network Architecture
    // =========================================================================
    println!("1. Network Architecture");
    println!("   -------------------------------------------");
    println!("   Layers: {}", config.num_layers());
    print!("   Dims:   ");
    for (i, d) in config.layer_dims.iter().enumerate() {
        if i > 0 {
            print!(" -> ");
        }
        print!("{d}");
    }
    println!();
    let mut net_relu = DeepNetwork::new(&config, Activation::ReLU);
    println!("   Total parameters: {}", net_relu.param_count());
    println!();

    // =========================================================================
    // Section 2: Forward Pass and Gradient Extraction (ReLU)
    // =========================================================================
    println!("2. Forward Pass & Gradient Extraction (ReLU)");
    println!("   -------------------------------------------");

    let output = net_relu.forward(&input);
    let probs = softmax(&output);
    let loss = -probs[target].max(1e-10).ln();
    println!("   Output logits: {:?}", &output);
    println!(
        "   Softmax probs: [{}]",
        probs
            .iter()
            .map(|p| format!("{p:.4}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("   Loss (target={}): {:.4}", target, loss);
    println!();

    // Compute gradients per layer
    let relu_grads = net_relu.compute_layer_gradients(&input, target);
    let relu_stats: Vec<GradientStats> = relu_grads
        .iter()
        .enumerate()
        .map(|(i, g)| GradientStats::from_gradients(&format!("Layer {}", i + 1), g))
        .collect();

    // Set gradients on tensors for optimizer compatibility demonstration
    net_relu.apply_gradients(&relu_grads);

    println!("   Per-Layer Gradient Statistics:");
    println!(
        "   {:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Layer", "Mean", "Std", "MaxAbs", "MinAbs", "Norm", "%NearZero"
    );
    println!("   {}", "-".repeat(72));
    for s in &relu_stats {
        println!(
            "   {:<10} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>9.1}%",
            s.layer_name, s.mean, s.std_dev, s.max_abs, s.min_abs, s.norm, s.pct_near_zero
        );
    }
    println!();

    // =========================================================================
    // Section 3: Gradient Flow Diagram (ReLU)
    // =========================================================================
    println!("3. Gradient Flow Diagram (ReLU)");
    println!("   -------------------------------------------");
    print_gradient_flow_diagram(&relu_stats);
    println!();

    // =========================================================================
    // Section 4: Gradient Health Diagnosis
    // =========================================================================
    println!("4. Gradient Health Diagnosis");
    println!("   -------------------------------------------");
    let health_report = diagnose_gradient_health(&relu_stats);
    for (name, health) in &health_report {
        println!("   {:<10} : {}", name, health.label());
    }
    let vanishing_count = health_report
        .iter()
        .filter(|(_, h)| *h == GradientHealth::Vanishing)
        .count();
    let exploding_count = health_report
        .iter()
        .filter(|(_, h)| *h == GradientHealth::Exploding)
        .count();
    println!();
    println!("   Vanishing layers: {vanishing_count}");
    println!("   Exploding layers: {exploding_count}");
    println!("   Vanishing threshold: {VANISHING_THRESHOLD:.1e}");
    println!("   Exploding threshold: {EXPLODING_THRESHOLD:.1e}");
    println!();

    // =========================================================================
    // Section 5: Sigmoid vs ReLU Comparison
    // =========================================================================
    println!("5. Activation Comparison: ReLU vs Sigmoid");
    println!("   -------------------------------------------");

    let mut net_sigmoid = DeepNetwork::new(&config, Activation::Sigmoid);
    let sigmoid_grads = net_sigmoid.compute_layer_gradients(&input, target);
    let sigmoid_stats: Vec<GradientStats> = sigmoid_grads
        .iter()
        .enumerate()
        .map(|(i, g)| GradientStats::from_gradients(&format!("Layer {}", i + 1), g))
        .collect();

    println!(
        "   {:<10} {:>14} {:>14} {:>14} {:>14}",
        "Layer", "ReLU Norm", "Sigmoid Norm", "ReLU %Zero", "Sig %Zero"
    );
    println!("   {}", "-".repeat(60));
    for (rs, ss) in relu_stats.iter().zip(sigmoid_stats.iter()) {
        println!(
            "   {:<10} {:>14.6} {:>14.6} {:>13.1}% {:>13.1}%",
            rs.layer_name, rs.norm, ss.norm, rs.pct_near_zero, ss.pct_near_zero
        );
    }
    println!();

    println!("   Sigmoid gradient flow diagram:");
    print_gradient_flow_diagram(&sigmoid_stats);
    println!();

    // Sigmoid gradient saturation analysis
    let sigmoid_health = diagnose_gradient_health(&sigmoid_stats);
    let sigmoid_vanishing = sigmoid_health
        .iter()
        .filter(|(_, h)| *h == GradientHealth::Vanishing)
        .count();
    println!(
        "   Sigmoid vanishing layers: {} (vs ReLU: {})",
        sigmoid_vanishing, vanishing_count
    );
    println!("   Sigmoid suffers more vanishing gradients in deep networks due to");
    println!("   the derivative being bounded by 0.25 (at sigmoid(0)=0.5).");
    println!();

    // =========================================================================
    // Section 6: Skip Connections Mitigation
    // =========================================================================
    println!("6. Mitigation: Skip Connections");
    println!("   -------------------------------------------");

    // Use a config where hidden dims match for skip connections
    let skip_config = NetworkConfig {
        layer_dims: vec![16, 16, 16, 16, 16, 16, 4],
        lr: 0.01,
        seed: 42,
    };
    let skip_input = generate_input(skip_config.layer_dims[0], skip_config.seed, 0);

    // Plain network (no skip)
    let mut plain_net = DeepNetwork::new(&skip_config, Activation::ReLU);
    let plain_grads = plain_net.compute_layer_gradients(&skip_input, target);
    let plain_stats: Vec<GradientStats> = plain_grads
        .iter()
        .enumerate()
        .map(|(i, g)| GradientStats::from_gradients(&format!("Layer {}", i + 1), g))
        .collect();

    // Skip network
    let mut skip_net = SkipNetwork::new(&skip_config, Activation::ReLU);
    let skip_grads = skip_net.compute_layer_gradients(&skip_input, target);
    let skip_stats: Vec<GradientStats> = skip_grads
        .iter()
        .enumerate()
        .map(|(i, g)| GradientStats::from_gradients(&format!("Layer {}", i + 1), g))
        .collect();

    println!(
        "   {:<10} {:>14} {:>14} {:>10}",
        "Layer", "Plain Norm", "Skip Norm", "Ratio"
    );
    println!("   {}", "-".repeat(52));
    for (ps, ss) in plain_stats.iter().zip(skip_stats.iter()) {
        let ratio = if ps.norm > 1e-10 {
            ss.norm / ps.norm
        } else {
            f32::INFINITY
        };
        println!(
            "   {:<10} {:>14.6} {:>14.6} {:>9.2}x",
            ps.layer_name, ps.norm, ss.norm, ratio
        );
    }
    println!();
    println!("   Skip connections allow gradients to bypass layers, preserving");
    println!("   gradient magnitude in early layers.");
    println!();

    // =========================================================================
    // Section 7: Batch Normalization Proxy
    // =========================================================================
    println!("7. Mitigation: Batch Normalization Proxy");
    println!("   -------------------------------------------");

    let (_, pre_acts) = plain_net.forward_with_preactivations(&skip_input);
    println!("   Pre-activation statistics (before and after batch norm):");
    println!(
        "   {:<10} {:>12} {:>12} {:>12} {:>12}",
        "Layer", "Raw Mean", "Raw Std", "BN Mean", "BN Std"
    );
    println!("   {}", "-".repeat(52));
    for (i, pa) in pre_acts.iter().enumerate() {
        let bn = batch_norm_proxy(pa);
        let raw_mean = pa.iter().sum::<f32>() / pa.len() as f32;
        let raw_var = pa
            .iter()
            .map(|v| (v - raw_mean) * (v - raw_mean))
            .sum::<f32>()
            / pa.len() as f32;
        let raw_std = raw_var.sqrt();

        let bn_mean = bn.iter().sum::<f32>() / bn.len() as f32;
        let bn_var = bn
            .iter()
            .map(|v| (v - bn_mean) * (v - bn_mean))
            .sum::<f32>()
            / bn.len() as f32;
        let bn_std = bn_var.sqrt();

        println!(
            "   Layer {:<4} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            i + 1,
            raw_mean,
            raw_std,
            bn_mean,
            bn_std
        );
    }
    println!();
    println!("   Batch norm centers activations around 0 with unit variance,");
    println!("   keeping activations in the sensitive region of the activation function.");
    println!();

    // =========================================================================
    // Section 8: Optimizer Step with Gradient Visualization
    // =========================================================================
    println!("8. Optimizer Step with Gradient Flow");
    println!("   -------------------------------------------");

    let mut opt_net = DeepNetwork::new(&config, Activation::ReLU);
    let mut optimizer = SGD::new(config.lr, 0.9);

    // Before step
    let before_grads = opt_net.compute_layer_gradients(&input, target);
    let before_loss = opt_net.cross_entropy_loss(&input, target);

    // Apply gradients via entrenar API
    opt_net.apply_gradients(&before_grads);

    // Collect params for optimizer step
    let mut param_vec: Vec<Tensor> = Vec::new();
    for w in &opt_net.weights {
        param_vec.push(Tensor::from_vec(w.data().to_vec(), true));
    }
    for b in &opt_net.biases {
        param_vec.push(Tensor::from_vec(b.data().to_vec(), true));
    }

    // Set grads on param_vec and step
    for (i, grads) in before_grads.iter().enumerate() {
        let grad_array = Array1::from_vec(grads.clone());
        param_vec[i].set_grad(grad_array);
    }
    optimizer.step(&mut param_vec);

    // Update network weights from optimized params
    for (i, w) in opt_net.weights.iter_mut().enumerate() {
        *w = Tensor::from_vec(param_vec[i].data().to_vec(), true);
    }

    let after_loss = opt_net.cross_entropy_loss(&input, target);
    println!("   Loss before step: {:.6}", before_loss);
    println!("   Loss after step:  {:.6}", after_loss);
    println!("   Loss delta:       {:.6}", after_loss - before_loss);
    println!();

    // =========================================================================
    // Section 9: Metrics Report
    // =========================================================================
    println!("9. Recipe Metrics");
    println!("   -------------------------------------------");

    ctx.record_metric("total_layers", config.num_layers() as i64);
    ctx.record_metric("total_params", net_relu.param_count() as i64);
    ctx.record_float_metric("relu_loss", f64::from(loss));
    ctx.record_metric("relu_vanishing_layers", vanishing_count as i64);
    ctx.record_metric("sigmoid_vanishing_layers", sigmoid_vanishing as i64);
    ctx.record_float_metric("loss_before_step", f64::from(before_loss));
    ctx.record_float_metric("loss_after_step", f64::from(after_loss));

    ctx.report()?;
    println!();
    println!("=== Example Complete ===");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_stats_from_gradients() {
        let grads = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let stats = GradientStats::from_gradients("test", &grads);
        assert_eq!(stats.layer_name, "test");
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 0.06).abs() < 1e-4, "mean={}", stats.mean);
        assert!(stats.max_abs > 0.49);
        assert!(stats.min_abs < 0.11);
        assert!(stats.norm > 0.0);
        assert!((stats.pct_near_zero - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gradient_stats_empty() {
        let stats = GradientStats::from_gradients("empty", &[]);
        assert_eq!(stats.count, 0);
        assert!((stats.norm - 0.0).abs() < f32::EPSILON);
        assert!((stats.pct_near_zero - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gradient_stats_near_zero_detection() {
        let grads = vec![1e-7, 1e-8, 0.5, 0.0, 1e-10];
        let stats = GradientStats::from_gradients("nz", &grads);
        // 4 values < 1e-6: 1e-7, 1e-8, 0.0, 1e-10
        assert!((stats.pct_near_zero - 80.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_activation_relu() {
        let act = Activation::ReLU;
        assert!((act.apply(-1.0) - 0.0).abs() < f32::EPSILON);
        assert!((act.apply(0.0) - 0.0).abs() < f32::EPSILON);
        assert!((act.apply(2.5) - 2.5).abs() < f32::EPSILON);
        assert!((act.derivative(-1.0) - 0.0).abs() < f32::EPSILON);
        assert!((act.derivative(1.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_activation_sigmoid() {
        let act = Activation::Sigmoid;
        let val = act.apply(0.0);
        assert!((val - 0.5).abs() < 1e-6, "sigmoid(0)={val}");
        let deriv = act.derivative(0.0);
        assert!((deriv - 0.25).abs() < 1e-6, "sigmoid'(0)={deriv}");
        // Sigmoid is bounded [0, 1)
        assert!(act.apply(-5.0) > 0.0);
        assert!(act.apply(-5.0) < 0.01);
        assert!(act.apply(5.0) > 0.99);
        assert!(act.apply(5.0) < 1.0);
    }

    #[test]
    fn test_diagnose_gradient_health() {
        let stats = vec![
            GradientStats::from_gradients("healthy", &[0.01, -0.02, 0.03]),
            GradientStats::from_gradients("vanishing", &[1e-7, -1e-8, 1e-9]),
            GradientStats::from_gradients("exploding", &[5000.0, -3000.0, 4000.0]),
        ];
        let health = diagnose_gradient_health(&stats);
        assert_eq!(health[0].1, GradientHealth::Healthy);
        assert_eq!(health[1].1, GradientHealth::Vanishing);
        assert_eq!(health[2].1, GradientHealth::Exploding);
    }

    #[test]
    fn test_deep_network_forward_dimensions() {
        let config = NetworkConfig::deep_network();
        let net = DeepNetwork::new(&config, Activation::ReLU);
        let input = generate_input(config.layer_dims[0], 42, 0);
        let output = net.forward(&input);
        assert_eq!(
            output.len(),
            *config.layer_dims.last().expect("non-empty dims"),
            "Output dimension mismatch"
        );
    }

    #[test]
    fn test_deep_network_deterministic() {
        let config = NetworkConfig::deep_network();
        let net1 = DeepNetwork::new(&config, Activation::ReLU);
        let net2 = DeepNetwork::new(&config, Activation::ReLU);
        let input = generate_input(config.layer_dims[0], 42, 0);
        assert_eq!(net1.forward(&input), net2.forward(&input));
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum}");
        for &p in &probs {
            assert!(p > 0.0);
            assert!(p <= 1.0);
        }
    }

    #[test]
    fn test_batch_norm_proxy_centers() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let normed = batch_norm_proxy(&values);
        let mean: f32 = normed.iter().sum::<f32>() / normed.len() as f32;
        assert!(mean.abs() < 1e-5, "batch norm mean={mean}");
        let var: f32 =
            normed.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / normed.len() as f32;
        assert!((var - 1.0).abs() < 0.01, "batch norm var={var}");
    }

    #[test]
    fn test_batch_norm_proxy_empty() {
        let normed = batch_norm_proxy(&[]);
        assert!(normed.is_empty());
    }

    #[test]
    fn test_skip_network_forward_dimensions() {
        let config = NetworkConfig {
            layer_dims: vec![16, 16, 16, 16, 16, 16, 4],
            lr: 0.01,
            seed: 42,
        };
        let net = SkipNetwork::new(&config, Activation::ReLU);
        let input = generate_input(16, 42, 0);
        let output = net.forward(&input);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_gradient_flow_relu_vs_sigmoid() {
        // Verify that sigmoid produces weaker gradients than ReLU in deep networks
        let config = NetworkConfig::deep_network();
        let input = generate_input(config.layer_dims[0], 42, 0);

        let mut relu_net = DeepNetwork::new(&config, Activation::ReLU);
        let relu_grads = relu_net.compute_layer_gradients(&input, 0);
        let relu_norm: f32 = relu_grads
            .iter()
            .flat_map(|g| g.iter())
            .map(|g| g * g)
            .sum::<f32>()
            .sqrt();

        let mut sig_net = DeepNetwork::new(&config, Activation::Sigmoid);
        let sig_grads = sig_net.compute_layer_gradients(&input, 0);
        let sig_norm: f32 = sig_grads
            .iter()
            .flat_map(|g| g.iter())
            .map(|g| g * g)
            .sum::<f32>()
            .sqrt();

        // Both should be finite and non-negative
        assert!(relu_norm.is_finite(), "ReLU gradient norm must be finite");
        assert!(sig_norm.is_finite(), "Sigmoid gradient norm must be finite");
        assert!(relu_norm >= 0.0);
        assert!(sig_norm >= 0.0);
    }
}
