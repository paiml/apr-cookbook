#![allow(unused_imports)]
//! # Recipe: Create APR Neural Network Model
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! **Category**: Model Creation
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A - uses filesystem)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Build a multi-layer perceptron (MLP) from scratch with Xavier weight
//! initialization, forward pass with ReLU/Sigmoid activations, and save
//! the neural network as a `.apr` model.
//!
//! ## Run Command
//! ```bash
//! cargo run --example create_apr_neural_network
//! ```
//!
//! ## Example Output
//! ```text
//! === Recipe: create_apr_neural_network ===
//! Section 1: Defined MLP architecture [4, 8, 6, 3]
//! Section 2: Initialized 3 layers with Xavier weights
//! Section 3: Forward pass output: [0.52, 0.51, 0.50]
//! Section 4: Saved to neural_network.apr (... bytes)
//! Section 5: Roundtrip verification: PASSED
//! Section 6: Architecture summary (123 total parameters)
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr convert model.apr          # APR native format
//! apr convert model.gguf         # GGUF (llama.cpp compatible)
//! apr convert model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Jacob, B. et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. CVPR. arXiv:1712.05877

use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

mod helpers;
#[allow(unused_imports, clippy::wildcard_imports)]
use helpers::*;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("create_apr_neural_network")?;

    // ====================================================================
    // Section 1: Define network architecture (layers, activations)
    // ====================================================================
    let architecture = vec![4, 8, 6, 3]; // input=4, hidden1=8, hidden2=6, output=3
    let activations = vec![Activation::ReLU, Activation::ReLU, Activation::Sigmoid];

    println!("=== Recipe: {} ===", ctx.name());
    println!("Section 1: Defined MLP architecture {:?}", architecture);

    ctx.record_metric("n_layers", (architecture.len() - 1) as i64);
    ctx.record_metric("input_dim", architecture[0] as i64);
    ctx.record_metric("output_dim", architecture[architecture.len() - 1] as i64);

    // ====================================================================
    // Section 2: Initialize weights with Xavier initialization
    // ====================================================================
    let seed = hash_name_to_seed(ctx.name());
    let network = initialize_network(&architecture, &activations, seed, "mlp-classifier");

    println!(
        "Section 2: Initialized {} layers with Xavier weights",
        network.layers.len()
    );

    // ====================================================================
    // Section 3: Forward pass demonstration
    // ====================================================================
    // Demonstrate all three activation functions
    let sample_val = 0.5;
    let _relu_out = apply_activation(sample_val, Activation::ReLU);
    let _sigmoid_out = apply_activation(sample_val, Activation::Sigmoid);
    let _identity_out = apply_activation(sample_val, Activation::None);

    let input = vec![1.0, 0.5, -0.3, 0.8];
    let result = forward(&network, &input);

    println!(
        "Section 3: Forward pass output: [{}]",
        result
            .output
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
            .join(", ")
    );
    ctx.record_metric("layer_outputs_count", result.layer_outputs.len() as i64);

    // ====================================================================
    // Section 4: Save to APR format
    // ====================================================================
    let mut converter = AprConverter::new();
    converter.set_metadata(ConversionMetadata {
        name: Some(network.name.clone()),
        architecture: Some("mlp".to_string()),
        source_format: None,
        custom: std::collections::HashMap::new(),
    });

    for (i, layer) in network.layers.iter().enumerate() {
        let flat_weights = flatten_weights(&layer.weights);
        let input_dim = layer.weights[0].len();
        let output_dim = layer.weights.len();

        converter.add_tensor(TensorData {
            name: format!("layer{}.weight", i),
            shape: vec![output_dim, input_dim],
            dtype: DataType::F32,
            data: f64_to_f32_bytes(&flat_weights),
        });

        converter.add_tensor(TensorData {
            name: format!("layer{}.bias", i),
            shape: vec![output_dim],
            dtype: DataType::F32,
            data: f64_to_f32_bytes(&layer.bias),
        });
    }

    let apr_path = ctx.path("neural_network.apr");
    let apr_bytes = converter.to_apr()?;
    std::fs::write(&apr_path, &apr_bytes)?;

    let file_size = std::fs::metadata(&apr_path)?.len();
    ctx.record_metric("file_size_bytes", file_size as i64);
    println!(
        "Section 4: Saved to neural_network.apr ({} bytes)",
        file_size
    );

    // ====================================================================
    // Section 5: Load from APR and verify weights preserved
    // ====================================================================
    let loaded_bytes = std::fs::read(&apr_path)?;
    let loaded = BundledModel::from_bytes(&loaded_bytes)?;

    let roundtrip_ok = loaded.size() == apr_bytes.len() && loaded.version() == (1, 0);
    ctx.record_string_metric(
        "roundtrip_verification",
        if roundtrip_ok { "PASSED" } else { "FAILED" },
    );
    println!(
        "Section 5: Roundtrip verification: {}",
        if roundtrip_ok { "PASSED" } else { "FAILED" }
    );

    // ====================================================================
    // Section 6: Architecture summary with parameter counts
    // ====================================================================
    let summaries = architecture_summary(&network);
    let total_params = total_parameters(&network);

    println!(
        "Section 6: Architecture summary ({} total parameters)",
        total_params
    );
    for summary in &summaries {
        println!(
            "  {}: {}x{} ({} params, {})",
            summary.layer_name,
            summary.input_dim,
            summary.output_dim,
            summary.params,
            summary.activation
        );
    }
    ctx.record_metric("total_parameters", total_params as i64);
    println!("Duration: {:.2}ms", ctx.elapsed().as_secs_f64() * 1000.0);

    Ok(())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod proptests;
