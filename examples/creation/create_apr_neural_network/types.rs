#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use proptest::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Activation function for a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    // Rectified Linear Unit: max(0, x)
    ReLU,
    // Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    // No activation (identity)
    None,
}

/// A single layer in the neural network.
#[derive(Debug, Clone)]
pub struct Layer {
    // Weight matrix: rows = output_dim, cols = input_dim
    pub weights: Vec<Vec<f64>>,
    // Bias vector: length = output_dim
    pub bias: Vec<f64>,
    // Activation function
    pub activation: Activation,
}

/// A feedforward neural network (multi-layer perceptron).
#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    // Ordered list of layers
    pub layers: Vec<Layer>,
    // Model name
    pub name: String,
}

/// Result of a forward pass through the network.
#[derive(Debug, Clone)]
pub struct ForwardResult {
    // Final output of the network
    pub output: Vec<f64>,
    // Intermediate outputs for each layer (pre-activation excluded for simplicity)
    pub layer_outputs: Vec<Vec<f64>>,
}

/// Summary of a single layer's architecture.
#[derive(Debug, Clone)]
pub struct ArchitectureSummary {
    // Human-readable layer name
    pub layer_name: String,
    // Input dimensionality
    pub input_dim: usize,
    // Output dimensionality
    pub output_dim: usize,
    // Total parameters (weights + biases)
    pub params: usize,
    // Activation function name
    pub activation: String,
}

// Recipe entry point - isolated and idempotent
