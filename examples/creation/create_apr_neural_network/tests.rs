#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
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
        let ctx = RecipeContext::new("nn_isolation_test").expect("Context creation should work");
        let path = ctx.path("test.apr");
        std::fs::write(&path, b"test").expect("Write should succeed");
        ctx.temp_dir().to_path_buf()
    };

    assert!(
        !temp_path.exists(),
        "Temp directory should be cleaned up on drop"
    );
}
