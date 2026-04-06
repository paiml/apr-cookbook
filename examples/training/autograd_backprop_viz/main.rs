#![allow(unused_imports)]
//! Autograd Backpropagation Visualization
//!
//! Contract: contracts/recipe-iiur-v1.yaml
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

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
