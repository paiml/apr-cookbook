#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use super::*;

/// Test basic inspection functionality
#[test]
fn test_basic_inspection() {
    let ctx = RecipeContext::new("test_basic").expect("context");
    let weights = generate_test_weights(&ctx, 100, false, false).expect("weights");
    let mut model = SimpleModel::new(10, 10);
    model.weights = weights.clone();
    let result = inspect_model(&model, &weights, "test.apr").expect("inspect");

    assert_eq!(result.header.magic, "APRN");
    assert!(result.quality_score.total > 0);
}

/// Test NaN detection
#[test]
fn test_nan_detection() {
    let ctx = RecipeContext::new("test_nan").expect("context");
    let weights = generate_test_weights(&ctx, 100, true, false).expect("weights");
    let stats = compute_weight_stats(&weights).expect("stats");

    assert_eq!(stats.nan_count, 1, "Should detect 1 NaN");
}

/// Test Inf detection
#[test]
fn test_inf_detection() {
    let ctx = RecipeContext::new("test_inf").expect("context");
    let weights = generate_test_weights(&ctx, 100, false, true).expect("weights");
    let stats = compute_weight_stats(&weights).expect("stats");

    assert_eq!(stats.inf_count, 1, "Should detect 1 Inf");
}

/// Test quality score calculation
#[test]
fn test_quality_score() {
    let header = HeaderInfo {
        magic: "APRN".to_string(),
        version: (1, 0),
        flags: FeatureFlags::default(),
        compression_ratio: 1.0,
        checksum: 12345,
    };
    let stats = LayerStats {
        name: "test".to_string(),
        shape: vec![100],
        dtype: "f32".to_string(),
        min: -1.0,
        max: 1.0,
        mean: 0.0,
        std: 0.5,
        nan_count: 0,
        inf_count: 0,
        zero_count: 10,
        sparsity: 0.1,
    };

    let score = calculate_quality_score(&header, &stats).expect("score");
    assert!(score.total >= 60, "Healthy model should score >= 60");
    assert_eq!(
        score.structural, 25,
        "Valid header should get full structural score"
    );
}

/// Test health status determination
#[test]
fn test_health_status() {
    assert_eq!(determine_health_status(100), HealthStatus::Healthy);
    assert_eq!(determine_health_status(85), HealthStatus::Healthy);
    assert_eq!(determine_health_status(84), HealthStatus::Warning);
    assert_eq!(determine_health_status(60), HealthStatus::Warning);
    assert_eq!(determine_health_status(59), HealthStatus::Critical);
    assert_eq!(determine_health_status(0), HealthStatus::Critical);
}

/// Test model diff
#[test]
fn test_model_diff_identical() {
    let weights = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let diff = compute_model_diff(&weights, &weights, "a", "b").expect("diff");

    assert!(
        (diff.total_l2_distance - 0.0).abs() < 1e-6,
        "Identical models should have L2=0"
    );
    assert!(
        (diff.cosine_similarity - 1.0).abs() < 1e-6,
        "Identical models should have cos=1"
    );
    assert!(
        !diff.drift_detected,
        "Identical models should not detect drift"
    );
}

/// Test model diff with changes
#[test]
fn test_model_diff_changed() {
    let weights_a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let weights_b = vec![1.0_f32, 2.0, 3.0, 4.0, 100.0]; // Large change
    let diff = compute_model_diff(&weights_a, &weights_b, "a", "b").expect("diff");

    assert!(
        diff.total_l2_distance > 0.0,
        "Changed models should have L2 > 0"
    );
    assert!(
        diff.drift_detected,
        "Large change should trigger drift detection"
    );
}

/// Test CRC32 checksum
#[test]
fn test_crc32_deterministic() {
    let weights = vec![1.0_f32, 2.0, 3.0];
    let crc1 = compute_crc32(&weights);
    let crc2 = compute_crc32(&weights);
    assert_eq!(crc1, crc2, "CRC32 should be deterministic");
}

/// Test empty weights handling
#[test]
fn test_empty_weights() {
    let weights: Vec<f32> = vec![];
    let result = compute_weight_stats(&weights);
    assert!(result.is_err(), "Empty weights should return error");
}

/// Test sparsity calculation
#[test]
fn test_sparsity() {
    let weights = vec![0.0_f32, 0.0, 1.0, 0.0, 2.0]; // 3/5 = 60% zeros
    let stats = compute_weight_stats(&weights).expect("stats");
    assert!(
        (stats.sparsity - 0.6).abs() < 0.01,
        "Sparsity should be 60%"
    );
}
