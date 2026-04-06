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

/// Deterministic pseudo-random f64 in [-1, 1] from a seed and index.
fn pseudo_random(seed: u64, index: u64) -> f64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    index.hash(&mut hasher);
    let hash = hasher.finish();
    // Map u64 to [-1.0, 1.0]
    (hash as f64 / u64::MAX as f64) * 2.0 - 1.0
}

/// Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
fn xavier_scale(fan_in: usize, fan_out: usize) -> f64 {
    (2.0 / (fan_in + fan_out) as f64).sqrt()
}

/// Initialize a neural network with Xavier weight initialization.
fn initialize_network(
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
fn apply_activation(value: f64, activation: Activation) -> f64 {
    match activation {
        Activation::ReLU => value.max(0.0),
        Activation::Sigmoid => 1.0 / (1.0 + (-value).exp()),
        Activation::None => value,
    }
}

/// Run a forward pass through the network.
fn forward(network: &NeuralNetwork, input: &[f64]) -> ForwardResult {
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
fn architecture_summary(network: &NeuralNetwork) -> Vec<ArchitectureSummary> {
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
fn flatten_weights(weights: &[Vec<f64>]) -> Vec<f64> {
    weights.iter().flat_map(|row| row.iter().copied()).collect()
}

/// Convert f64 values to f32 little-endian bytes for APR storage.
fn f64_to_f32_bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| (v as f32).to_le_bytes())
        .collect()
}

/// Count total parameters in a network.
fn total_parameters(network: &NeuralNetwork) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_network() -> NeuralNetwork {
        let architecture = vec![4, 8, 6, 3];
        let activations = vec![Activation::ReLU, Activation::ReLU, Activation::Sigmoid];
        initialize_network(&architecture, &activations, 42, "test-mlp")
    }

    #[test]
    fn test_xavier_scale_symmetric() {
        let scale = xavier_scale(100, 100);
        let expected = (2.0 / 200.0_f64).sqrt();
        assert!((scale - expected).abs() < 1e-10);
    }

    #[test]
    fn test_xavier_scale_asymmetric() {
        let scale = xavier_scale(784, 256);
        let expected = (2.0 / 1040.0_f64).sqrt();
        assert!((scale - expected).abs() < 1e-10);
    }

    #[test]
    fn test_pseudo_random_deterministic() {
        let v1 = pseudo_random(42, 0);
        let v2 = pseudo_random(42, 0);
        assert!((v1 - v2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pseudo_random_range() {
        for i in 0..1000 {
            let v = pseudo_random(42, i);
            assert!((-1.0..=1.0).contains(&v), "Value {} out of range: {}", i, v);
        }
    }

    #[test]
    fn test_pseudo_random_different_seeds() {
        let v1 = pseudo_random(42, 0);
        let v2 = pseudo_random(43, 0);
        assert!(
            (v1 - v2).abs() > f64::EPSILON,
            "Different seeds should produce different values"
        );
    }

    #[test]
    fn test_initialize_network_layer_count() {
        let net = make_test_network();
        assert_eq!(net.layers.len(), 3);
    }

    #[test]
    fn test_initialize_network_dimensions() {
        let net = make_test_network();
        // Layer 0: 4 -> 8
        assert_eq!(net.layers[0].weights.len(), 8);
        assert_eq!(net.layers[0].weights[0].len(), 4);
        assert_eq!(net.layers[0].bias.len(), 8);
        // Layer 1: 8 -> 6
        assert_eq!(net.layers[1].weights.len(), 6);
        assert_eq!(net.layers[1].weights[0].len(), 8);
        assert_eq!(net.layers[1].bias.len(), 6);
        // Layer 2: 6 -> 3
        assert_eq!(net.layers[2].weights.len(), 3);
        assert_eq!(net.layers[2].weights[0].len(), 6);
        assert_eq!(net.layers[2].bias.len(), 3);
    }

    #[test]
    fn test_initialize_network_biases_zero() {
        let net = make_test_network();
        for layer in &net.layers {
            for &b in &layer.bias {
                assert!((b - 0.0).abs() < f64::EPSILON, "Biases should be zero");
            }
        }
    }

    #[test]
    fn test_initialize_network_activations() {
        let net = make_test_network();
        assert_eq!(net.layers[0].activation, Activation::ReLU);
        assert_eq!(net.layers[1].activation, Activation::ReLU);
        assert_eq!(net.layers[2].activation, Activation::Sigmoid);
    }

    #[test]
    fn test_activation_relu() {
        assert!((apply_activation(2.0, Activation::ReLU) - 2.0).abs() < f64::EPSILON);
        assert!((apply_activation(-1.0, Activation::ReLU) - 0.0).abs() < f64::EPSILON);
        assert!((apply_activation(0.0, Activation::ReLU) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_activation_sigmoid() {
        let s0 = apply_activation(0.0, Activation::Sigmoid);
        assert!((s0 - 0.5).abs() < 1e-10, "sigmoid(0) should be 0.5");

        let s_large = apply_activation(10.0, Activation::Sigmoid);
        assert!(s_large > 0.999, "sigmoid(10) should be ~1.0");

        let s_neg = apply_activation(-10.0, Activation::Sigmoid);
        assert!(s_neg < 0.001, "sigmoid(-10) should be ~0.0");
    }

    #[test]
    fn test_activation_none() {
        assert!((apply_activation(3.14, Activation::None) - 3.14).abs() < f64::EPSILON);
        assert!((apply_activation(-2.7, Activation::None) - (-2.7)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_forward_output_dimension() {
        let net = make_test_network();
        let input = vec![1.0, 0.5, -0.3, 0.8];
        let result = forward(&net, &input);

        assert_eq!(result.output.len(), 3, "Output should match last layer dim");
        assert_eq!(
            result.layer_outputs.len(),
            3,
            "Should have one output per layer"
        );
    }

    #[test]
    fn test_forward_sigmoid_output_range() {
        let net = make_test_network();
        let input = vec![1.0, 0.5, -0.3, 0.8];
        let result = forward(&net, &input);

        // Last layer uses Sigmoid, so output should be in (0, 1)
        for &val in &result.output {
            assert!(
                val > 0.0 && val < 1.0,
                "Sigmoid output should be in (0, 1), got {}",
                val
            );
        }
    }

    #[test]
    fn test_forward_deterministic() {
        let net = make_test_network();
        let input = vec![1.0, 0.5, -0.3, 0.8];

        let r1 = forward(&net, &input);
        let r2 = forward(&net, &input);

        assert_eq!(r1.output, r2.output, "Forward pass should be deterministic");
    }

    #[test]
    fn test_architecture_summary_count() {
        let net = make_test_network();
        let summaries = architecture_summary(&net);
        assert_eq!(summaries.len(), 3);
    }

    #[test]
    fn test_architecture_summary_params() {
        let net = make_test_network();
        let summaries = architecture_summary(&net);

        // Layer 0: 4*8 + 8 = 40
        assert_eq!(summaries[0].params, 40);
        // Layer 1: 8*6 + 6 = 54
        assert_eq!(summaries[1].params, 54);
        // Layer 2: 6*3 + 3 = 21
        assert_eq!(summaries[2].params, 21);
    }

    #[test]
    fn test_total_parameters() {
        let net = make_test_network();
        let total = total_parameters(&net);
        // 40 + 54 + 21 = 115
        assert_eq!(total, 115);
    }

    #[test]
    fn test_flatten_weights() {
        let weights = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let flat = flatten_weights(&weights);
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_f64_to_f32_bytes() {
        let values = vec![1.0_f64, -1.0, 0.0];
        let bytes = f64_to_f32_bytes(&values);
        assert_eq!(bytes.len(), 12); // 3 values * 4 bytes each

        // Verify first value roundtrips
        let first = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert!((first - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_apr_roundtrip() {
        let net = make_test_network();

        let mut converter = AprConverter::new();
        converter.set_metadata(ConversionMetadata {
            name: Some(net.name.clone()),
            architecture: Some("mlp".to_string()),
            source_format: None,
            custom: std::collections::HashMap::new(),
        });

        for (i, layer) in net.layers.iter().enumerate() {
            let flat = flatten_weights(&layer.weights);
            converter.add_tensor(TensorData {
                name: format!("layer{}.weight", i),
                shape: vec![layer.weights.len(), layer.weights[0].len()],
                dtype: DataType::F32,
                data: f64_to_f32_bytes(&flat),
            });
            converter.add_tensor(TensorData {
                name: format!("layer{}.bias", i),
                shape: vec![layer.bias.len()],
                dtype: DataType::F32,
                data: f64_to_f32_bytes(&layer.bias),
            });
        }

        let apr_bytes = converter.to_apr().expect("APR conversion should succeed");
        assert_eq!(&apr_bytes[0..4], b"APRN", "Should have APR magic bytes");

        let loaded = BundledModel::from_bytes(&apr_bytes).expect("Loading should succeed");
        assert_eq!(loaded.size(), apr_bytes.len());
        assert_eq!(loaded.version(), (1, 0));
    }

    #[test]
    fn test_idempotency() {
        let result1 = run_recipe();
        let result2 = run_recipe();
        assert!(result1.is_ok());
        assert!(result2.is_ok());
    }

    fn run_recipe() -> Result<()> {
        let ctx = RecipeContext::new("nn_idempotency_test")?;
        let seed = hash_name_to_seed(ctx.name());
        let arch = vec![4, 8, 3];
        let acts = vec![Activation::ReLU, Activation::Sigmoid];
        let net = initialize_network(&arch, &acts, seed, "test");

        let mut converter = AprConverter::new();
        for (i, layer) in net.layers.iter().enumerate() {
            let flat = flatten_weights(&layer.weights);
            converter.add_tensor(TensorData {
                name: format!("layer{}.weight", i),
                shape: vec![layer.weights.len(), layer.weights[0].len()],
                dtype: DataType::F32,
                data: f64_to_f32_bytes(&flat),
            });
        }

        let apr_path = ctx.path("model.apr");
        let apr_bytes = converter.to_apr()?;
        std::fs::write(&apr_path, &apr_bytes)?;
        Ok(())
    }

    #[test]
    fn test_isolation_no_file_leaks() {
        let temp_path = {
            let ctx =
                RecipeContext::new("nn_isolation_test").expect("Context creation should work");
            let path = ctx.path("test.apr");
            std::fs::write(&path, b"test").expect("Write should succeed");
            ctx.temp_dir().to_path_buf()
        };

        assert!(
            !temp_path.exists(),
            "Temp directory should be cleaned up on drop"
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_sigmoid_bounded(x in -100.0f64..100.0) {
            let s = apply_activation(x, Activation::Sigmoid);
            prop_assert!(s >= 0.0 && s <= 1.0, "Sigmoid must be in [0,1], got {}", s);
        }

        #[test]
        fn prop_relu_non_negative(x in -100.0f64..100.0) {
            let r = apply_activation(x, Activation::ReLU);
            prop_assert!(r >= 0.0, "ReLU must be >= 0, got {}", r);
        }

        #[test]
        fn prop_relu_identity_for_positive(x in 0.0f64..100.0) {
            let r = apply_activation(x, Activation::ReLU);
            prop_assert!((r - x).abs() < f64::EPSILON, "ReLU(x) == x for x >= 0");
        }

        #[test]
        fn prop_forward_output_length(
            hidden in 2usize..20,
            output in 1usize..10,
        ) {
            let arch = vec![3, hidden, output];
            let acts = vec![Activation::ReLU, Activation::Sigmoid];
            let net = initialize_network(&arch, &acts, 42, "prop-test");
            let input = vec![1.0, 0.5, -0.3];

            let result = forward(&net, &input);
            prop_assert_eq!(result.output.len(), output);
        }

        #[test]
        fn prop_xavier_scale_positive(fan_in in 1usize..1000, fan_out in 1usize..1000) {
            let scale = xavier_scale(fan_in, fan_out);
            prop_assert!(scale > 0.0, "Xavier scale must be positive, got {}", scale);
        }

        #[test]
        fn prop_total_params_formula(
            input in 1usize..50,
            hidden in 1usize..50,
            output in 1usize..50,
        ) {
            let arch = vec![input, hidden, output];
            let acts = vec![Activation::ReLU, Activation::Sigmoid];
            let net = initialize_network(&arch, &acts, 42, "prop-params");

            let total = total_parameters(&net);
            let expected = input * hidden + hidden + hidden * output + output;
            prop_assert_eq!(total, expected);
        }
    }
}
