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
#[allow(unused_imports, clippy::wildcard_imports)]
use super::types::*;
use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Deterministic pseudo-random f64 in [-1, 1] from a seed and index.
pub fn pseudo_random(seed: u64, index: u64) -> f64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    index.hash(&mut hasher);
    let hash = hasher.finish();
    // Map u64 to [-1.0, 1.0]
    (hash as f64 / u64::MAX as f64) * 2.0 - 1.0
}

/// Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
pub fn xavier_scale(fan_in: usize, fan_out: usize) -> f64 {
    (2.0 / (fan_in + fan_out) as f64).sqrt()
}

/// Initialize a neural network with Xavier weight initialization.
pub fn initialize_network(
    architecture: &[usize],
    activations: &[Activation],
    seed: u64,
    name: &str,
) -> NeuralNetwork {
    let mut layers = Vec::with_capacity(architecture.len() - 1);
    let mut rng_index: u64 = 0;

    for i in 0..architecture.len() - 1 {
        let fan_in = architecture[i];
        let fan_out = architecture[i + 1];
        let scale = xavier_scale(fan_in, fan_out);

        let mut weights = Vec::with_capacity(fan_out);
        for _ in 0..fan_out {
            let mut row = Vec::with_capacity(fan_in);
            for _ in 0..fan_in {
                let val = pseudo_random(seed, rng_index) * scale;
                rng_index += 1;
                row.push(val);
            }
            weights.push(row);
        }

        let bias = vec![0.0; fan_out]; // Biases initialized to zero (standard practice)

        layers.push(Layer {
            weights,
            bias,
            activation: activations[i],
        });
    }

    NeuralNetwork {
        layers,
        name: name.to_string(),
    }
}

/// Apply activation function to a value.
pub fn apply_activation(value: f64, activation: Activation) -> f64 {
    match activation {
        Activation::ReLU => value.max(0.0),
        Activation::Sigmoid => 1.0 / (1.0 + (-value).exp()),
        Activation::None => value,
    }
}

/// Run a forward pass through the network.
pub fn forward(network: &NeuralNetwork, input: &[f64]) -> ForwardResult {
    let mut current = input.to_vec();
    let mut layer_outputs = Vec::with_capacity(network.layers.len());

    for layer in &network.layers {
        let output_dim = layer.weights.len();
        let mut output = Vec::with_capacity(output_dim);

        for j in 0..output_dim {
            let mut sum = layer.bias[j];
            for (k, &input_val) in current.iter().enumerate() {
                sum += layer.weights[j][k] * input_val;
            }
            output.push(apply_activation(sum, layer.activation));
        }

        layer_outputs.push(output.clone());
        current = output;
    }

    ForwardResult {
        output: current,
        layer_outputs,
    }
}

/// Compute architecture summary with parameter counts per layer.
pub fn architecture_summary(network: &NeuralNetwork) -> Vec<ArchitectureSummary> {
    network
        .layers
        .iter()
        .enumerate()
        .map(|(i, layer)| {
            let input_dim = layer.weights[0].len();
            let output_dim = layer.weights.len();
            let weight_params = input_dim * output_dim;
            let bias_params = output_dim;
            let activation_str = match layer.activation {
                Activation::ReLU => "ReLU",
                Activation::Sigmoid => "Sigmoid",
                Activation::None => "None",
            };

            ArchitectureSummary {
                layer_name: format!("layer{}", i),
                input_dim,
                output_dim,
                params: weight_params + bias_params,
                activation: activation_str.to_string(),
            }
        })
        .collect()
}

/// Flatten a 2D weight matrix into a 1D vector (row-major).
pub fn flatten_weights(weights: &[Vec<f64>]) -> Vec<f64> {
    weights.iter().flat_map(|row| row.iter().copied()).collect()
}

/// Convert f64 values to f32 little-endian bytes for APR storage.
pub fn f64_to_f32_bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| (v as f32).to_le_bytes())
        .collect()
}

/// Count total parameters in a network.
pub fn total_parameters(network: &NeuralNetwork) -> usize {
    network
        .layers
        .iter()
        .map(|layer| {
            let input_dim = layer.weights[0].len();
            let output_dim = layer.weights.len();
            input_dim * output_dim + output_dim
        })
        .sum()
}
