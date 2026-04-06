//! Autograd Backpropagation Visualization
//!
//! Visualizes gradient flow through a multi-layer neural network.
//! Detects vanishing/exploding gradients, compares ReLU vs sigmoid,
//! and demonstrates skip connections and batch norm mitigations.
//!
//! ```bash
//! cargo run --example autograd_backprop_viz
//! ```
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
use entrenar::optim::{Optimizer, SGD};
use ndarray::Array1;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---- Configuration ----------------------------------------------------------

#[derive(Debug, Clone)]
struct NetworkConfig {
    layer_dims: Vec<usize>,
    lr: f32,
    seed: u64,
}

impl NetworkConfig {
    fn deep_network() -> Self {
        Self {
            layer_dims: vec![8, 12, 12, 12, 12, 12, 4],
            lr: 0.01,
            seed: 42,
        }
    }
    fn num_layers(&self) -> usize {
        self.layer_dims.len() - 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Activation {
    ReLU,
    Sigmoid,
}

impl Activation {
    fn apply(self, x: f32) -> f32 {
        match self {
            Self::ReLU => x.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        }
    }
    #[cfg(test)]
    fn derivative(self, x: f32) -> f32 {
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
enum GradientHealth {
    Healthy,
    Vanishing,
    Exploding,
}

impl GradientHealth {
    fn label(self) -> &'static str {
        match self {
            Self::Healthy => "HEALTHY",
            Self::Vanishing => "VANISHING",
            Self::Exploding => "EXPLODING",
        }
    }
}

const VANISHING_THRESHOLD: f32 = 1e-4;
const EXPLODING_THRESHOLD: f32 = 1e3;

fn diagnose_gradient_health(stats: &[GradientStats]) -> Vec<(String, GradientHealth)> {
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

struct DeepNetwork {
    weights: Vec<Tensor>,
    biases: Vec<Tensor>,
    dims: Vec<usize>,
    activation: Activation,
}

impl DeepNetwork {
    fn new(config: &NetworkConfig, activation: Activation) -> Self {
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

    fn linear_forward(&self, current: &[f32], layer_idx: usize) -> Vec<f32> {
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

    fn forward_with_preactivations(&self, input: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
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

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.forward_with_preactivations(input).0
    }

    fn cross_entropy_loss(&self, input: &[f32], target: usize) -> f32 {
        -softmax(&self.forward(input))[target].max(1e-10).ln()
    }

    fn compute_layer_gradients(&mut self, input: &[f32], target: usize) -> Vec<Vec<f32>> {
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

    fn apply_gradients(&self, layer_grads: &[Vec<f32>]) {
        for (i, grads) in layer_grads.iter().enumerate() {
            self.weights[i].set_grad(Array1::from_vec(grads.clone()));
        }
    }

    fn param_count(&self) -> usize {
        self.weights.iter().map(Tensor::len).sum::<usize>()
            + self.biases.iter().map(Tensor::len).sum::<usize>()
    }
}

// ---- Skip Network -----------------------------------------------------------

struct SkipNetwork {
    base: DeepNetwork,
}

impl SkipNetwork {
    fn new(config: &NetworkConfig, activation: Activation) -> Self {
        Self {
            base: DeepNetwork::new(config, activation),
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
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

    fn cross_entropy_loss(&self, input: &[f32], target: usize) -> f32 {
        -softmax(&self.forward(input))[target].max(1e-10).ln()
    }

    fn compute_layer_gradients(&mut self, input: &[f32], target: usize) -> Vec<Vec<f32>> {
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

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn generate_input(dim: usize, seed: u64, idx: usize) -> Vec<f32> {
    (0..dim)
        .map(|j| {
            let mut h = DefaultHasher::new();
            (seed, "input", idx, j).hash(&mut h);
            h.finish() as f32 / u64::MAX as f32 - 0.5
        })
        .collect()
}

fn batch_norm_proxy(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return vec![];
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let std = (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n + 1e-5).sqrt();
    values.iter().map(|v| (v - mean) / std).collect()
}

fn to_stats(grads: &[Vec<f32>]) -> Vec<GradientStats> {
    grads
        .iter()
        .enumerate()
        .map(|(i, g)| GradientStats::from_gradients(&format!("Layer {}", i + 1), g))
        .collect()
}

fn print_gradient_flow(stats: &[GradientStats]) {
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

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("autograd_backprop_viz")?;
    println!("=== Autograd Backpropagation Visualization ===\n");
    let config = NetworkConfig::deep_network();
    let input = generate_input(config.layer_dims[0], config.seed, 0);
    let target = 1_usize;

    println!(
        "1. Network: {} layers, dims {:?}, params {}",
        config.num_layers(),
        config.layer_dims,
        {
            let net = DeepNetwork::new(&config, Activation::ReLU);
            net.param_count()
        }
    );

    let mut net_relu = DeepNetwork::new(&config, Activation::ReLU);
    let output = net_relu.forward(&input);
    let probs = softmax(&output);
    let loss = -probs[target].max(1e-10).ln();
    println!("\n2. Forward (ReLU): loss={:.4}", loss);
    let relu_grads = net_relu.compute_layer_gradients(&input, target);
    let relu_stats = to_stats(&relu_grads);
    net_relu.apply_gradients(&relu_grads);
    println!(
        "   Per-layer norms: {}",
        relu_stats
            .iter()
            .map(|s| format!("{:.6}", s.norm))
            .collect::<Vec<_>>()
            .join(", ")
    );

    println!("\n3. Gradient Flow (ReLU)");
    print_gradient_flow(&relu_stats);

    println!("\n4. Health Diagnosis");
    let health = diagnose_gradient_health(&relu_stats);
    for (name, h) in &health {
        println!("   {:<10} : {}", name, h.label());
    }
    let vanishing_count = health
        .iter()
        .filter(|(_, h)| *h == GradientHealth::Vanishing)
        .count();
    let _exploding_count = health
        .iter()
        .filter(|(_, h)| *h == GradientHealth::Exploding)
        .count();

    println!("\n5. ReLU vs Sigmoid");
    let mut net_sig = DeepNetwork::new(&config, Activation::Sigmoid);
    let sig_stats = to_stats(&net_sig.compute_layer_gradients(&input, target));
    for (rs, ss) in relu_stats.iter().zip(sig_stats.iter()) {
        println!(
            "   {:<10} relu={:.6} sig={:.6}",
            rs.layer_name, rs.norm, ss.norm
        );
    }
    let sig_vanishing = diagnose_gradient_health(&sig_stats)
        .iter()
        .filter(|(_, h)| *h == GradientHealth::Vanishing)
        .count();

    println!("\n6. Skip Connections");
    let skip_config = NetworkConfig {
        layer_dims: vec![16, 16, 16, 16, 16, 16, 4],
        lr: 0.01,
        seed: 42,
    };
    let skip_input = generate_input(16, 42, 0);
    let plain_stats = to_stats(
        &DeepNetwork::new(&skip_config, Activation::ReLU)
            .compute_layer_gradients(&skip_input, target),
    );
    let skip_stats = to_stats(
        &SkipNetwork::new(&skip_config, Activation::ReLU)
            .compute_layer_gradients(&skip_input, target),
    );
    for (ps, ss) in plain_stats.iter().zip(skip_stats.iter()) {
        let ratio = if ps.norm > 1e-10 {
            ss.norm / ps.norm
        } else {
            f32::INFINITY
        };
        println!(
            "   {:<10} plain={:.6} skip={:.6} ratio={:.2}x",
            ps.layer_name, ps.norm, ss.norm, ratio
        );
    }

    println!("\n7. Batch Norm Proxy");
    let plain_net = DeepNetwork::new(&skip_config, Activation::ReLU);
    let (_, pre_acts) = plain_net.forward_with_preactivations(&skip_input);
    for (i, pa) in pre_acts.iter().enumerate() {
        let bn = batch_norm_proxy(pa);
        let bn_mean = bn.iter().sum::<f32>() / bn.len() as f32;
        let bn_var = bn
            .iter()
            .map(|v| (v - bn_mean) * (v - bn_mean))
            .sum::<f32>()
            / bn.len() as f32;
        println!(
            "   Layer {} BN mean={:.6} std={:.6}",
            i + 1,
            bn_mean,
            bn_var.sqrt()
        );
    }

    println!("\n8. Optimizer Step");
    let mut opt_net = DeepNetwork::new(&config, Activation::ReLU);
    let mut optimizer = SGD::new(config.lr, 0.9);
    let before_grads = opt_net.compute_layer_gradients(&input, target);
    let before_loss = opt_net.cross_entropy_loss(&input, target);
    opt_net.apply_gradients(&before_grads);
    let mut param_vec: Vec<Tensor> = opt_net
        .weights
        .iter()
        .chain(opt_net.biases.iter())
        .map(|t| Tensor::from_vec(t.data().to_vec(), true))
        .collect();
    for (i, grads) in before_grads.iter().enumerate() {
        param_vec[i].set_grad(Array1::from_vec(grads.clone()));
    }
    optimizer.step(&mut param_vec);
    for (i, w) in opt_net.weights.iter_mut().enumerate() {
        *w = Tensor::from_vec(param_vec[i].data().to_vec(), true);
    }
    let after_loss = opt_net.cross_entropy_loss(&input, target);
    println!(
        "   Before: {:.6}  After: {:.6}  Delta: {:.6}",
        before_loss,
        after_loss,
        after_loss - before_loss
    );

    ctx.record_metric("total_layers", config.num_layers() as i64);
    ctx.record_metric("total_params", net_relu.param_count() as i64);
    ctx.record_float_metric("relu_loss", f64::from(loss));
    ctx.record_metric("relu_vanishing_layers", vanishing_count as i64);
    ctx.record_metric("sigmoid_vanishing_layers", sig_vanishing as i64);
    ctx.record_float_metric("loss_before_step", f64::from(before_loss));
    ctx.record_float_metric("loss_after_step", f64::from(after_loss));
    println!("\n9. Metrics");
    ctx.report()?;
    println!("\n=== Example Complete ===");
    Ok(())
}

// ---- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_stats() {
        let stats = GradientStats::from_gradients("test", &[0.1, -0.2, 0.3, -0.4, 0.5]);
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 0.06).abs() < 1e-4);
        assert!(stats.max_abs > 0.49);
        assert!(stats.norm > 0.0);
        let empty = GradientStats::from_gradients("e", &[]);
        assert_eq!(empty.count, 0);
        assert!((empty.pct_near_zero - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_near_zero_detection() {
        let stats = GradientStats::from_gradients("nz", &[1e-7, 1e-8, 0.5, 0.0, 1e-10]);
        assert!((stats.pct_near_zero - 80.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_activations() {
        let relu = Activation::ReLU;
        assert!((relu.apply(-1.0)).abs() < f32::EPSILON);
        assert!((relu.apply(2.5) - 2.5).abs() < f32::EPSILON);
        assert!((relu.derivative(-1.0)).abs() < f32::EPSILON);
        assert!((relu.derivative(1.0) - 1.0).abs() < f32::EPSILON);
        let sig = Activation::Sigmoid;
        assert!((sig.apply(0.0) - 0.5).abs() < 1e-6);
        assert!((sig.derivative(0.0) - 0.25).abs() < 1e-6);
        assert!(sig.apply(-5.0) < 0.01);
        assert!(sig.apply(5.0) > 0.99);
    }

    #[test]
    fn test_gradient_health_diagnosis() {
        let stats = vec![
            GradientStats::from_gradients("ok", &[0.01, -0.02, 0.03]),
            GradientStats::from_gradients("van", &[1e-7, -1e-8, 1e-9]),
            GradientStats::from_gradients("exp", &[5000.0, -3000.0, 4000.0]),
        ];
        let h = diagnose_gradient_health(&stats);
        assert_eq!(h[0].1, GradientHealth::Healthy);
        assert_eq!(h[1].1, GradientHealth::Vanishing);
        assert_eq!(h[2].1, GradientHealth::Exploding);
    }

    #[test]
    fn test_network_forward_and_determinism() {
        let config = NetworkConfig::deep_network();
        let net1 = DeepNetwork::new(&config, Activation::ReLU);
        let net2 = DeepNetwork::new(&config, Activation::ReLU);
        let input = generate_input(config.layer_dims[0], 42, 0);
        let out1 = net1.forward(&input);
        assert_eq!(out1.len(), *config.layer_dims.last().unwrap());
        assert_eq!(out1, net2.forward(&input));
    }

    #[test]
    fn test_softmax() {
        let probs = softmax(&[1.0, 2.0, 3.0, 4.0]);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        for &p in &probs {
            assert!(p > 0.0 && p <= 1.0);
        }
    }

    #[test]
    fn test_batch_norm_proxy() {
        let normed = batch_norm_proxy(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let mean: f32 = normed.iter().sum::<f32>() / normed.len() as f32;
        assert!(mean.abs() < 1e-5);
        let var: f32 =
            normed.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / normed.len() as f32;
        assert!((var - 1.0).abs() < 0.01);
        assert!(batch_norm_proxy(&[]).is_empty());
    }

    #[test]
    fn test_skip_network_dims() {
        let config = NetworkConfig {
            layer_dims: vec![16, 16, 16, 16, 16, 16, 4],
            lr: 0.01,
            seed: 42,
        };
        let net = SkipNetwork::new(&config, Activation::ReLU);
        assert_eq!(net.forward(&generate_input(16, 42, 0)).len(), 4);
    }

    #[test]
    fn test_relu_vs_sigmoid_finite() {
        let config = NetworkConfig::deep_network();
        let input = generate_input(config.layer_dims[0], 42, 0);
        let relu_norm: f32 = DeepNetwork::new(&config, Activation::ReLU)
            .compute_layer_gradients(&input, 0)
            .iter()
            .flat_map(|g| g.iter())
            .map(|g| g * g)
            .sum::<f32>()
            .sqrt();
        let sig_norm: f32 = DeepNetwork::new(&config, Activation::Sigmoid)
            .compute_layer_gradients(&input, 0)
            .iter()
            .flat_map(|g| g.iter())
            .map(|g| g * g)
            .sum::<f32>()
            .sqrt();
        assert!(relu_norm.is_finite() && relu_norm >= 0.0);
        assert!(sig_norm.is_finite() && sig_norm >= 0.0);
    }
}
