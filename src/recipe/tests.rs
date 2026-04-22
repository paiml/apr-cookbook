//! Tests for recipe infrastructure.

use crate::recipe::*;
use std::time::Duration;

#[test]
fn test_recipe_context_creation() {
    let ctx = RecipeContext::new("test_recipe").unwrap();
    assert_eq!(ctx.name(), "test_recipe");
    assert!(ctx.temp_dir().exists());
}

#[test]
fn test_recipe_context_path() {
    let ctx = RecipeContext::new("test_recipe").unwrap();
    let path = ctx.path("model.apr");
    assert!(path.starts_with(ctx.temp_dir()));
    assert!(path.ends_with("model.apr"));
}

#[test]
fn test_deterministic_rng() {
    // Same recipe name should produce same RNG sequence
    let mut ctx1 = RecipeContext::new("deterministic_test").unwrap();
    let mut ctx2 = RecipeContext::new("deterministic_test").unwrap();

    use rand::Rng;
    let seq1: Vec<u64> = (0..10).map(|_| ctx1.rng().gen()).collect();
    let seq2: Vec<u64> = (0..10).map(|_| ctx2.rng().gen()).collect();

    assert_eq!(
        seq1, seq2,
        "Same recipe name should produce same RNG sequence"
    );
}

#[test]
fn test_different_recipes_different_rng() {
    let mut ctx1 = RecipeContext::new("recipe_a").unwrap();
    let mut ctx2 = RecipeContext::new("recipe_b").unwrap();

    use rand::Rng;
    let val1: u64 = ctx1.rng().gen();
    let val2: u64 = ctx2.rng().gen();

    assert_ne!(
        val1, val2,
        "Different recipe names should produce different RNG"
    );
}

#[test]
fn test_temp_dir_isolation() {
    let ctx1 = RecipeContext::new("isolation_test_1").unwrap();
    let ctx2 = RecipeContext::new("isolation_test_2").unwrap();

    assert_ne!(
        ctx1.temp_dir(),
        ctx2.temp_dir(),
        "Each context should have its own temp directory"
    );
}

#[test]
fn test_metrics_recording() {
    let mut ctx = RecipeContext::new("metrics_test").unwrap();

    ctx.record_metric("byte_count", 1024);
    ctx.record_float_metric("throughput", 123.456);
    ctx.record_duration("inference_time", Duration::from_millis(42));
    ctx.record_string_metric("model_name", "test-model");

    match ctx.get_metric("byte_count") {
        Some(MetricValue::Int(v)) => assert_eq!(*v, 1024),
        _ => panic!("Expected Int metric"),
    }

    match ctx.get_metric("throughput") {
        Some(MetricValue::Float(v)) => assert!((v - 123.456).abs() < 0.001),
        _ => panic!("Expected Float metric"),
    }
}

#[test]
fn test_hash_name_to_seed_deterministic() {
    let seed1 = hash_name_to_seed("my_recipe");
    let seed2 = hash_name_to_seed("my_recipe");
    assert_eq!(seed1, seed2);
}

#[test]
fn test_hash_name_to_seed_different_names() {
    let seed1 = hash_name_to_seed("recipe_a");
    let seed2 = hash_name_to_seed("recipe_b");
    assert_ne!(seed1, seed2);
}

#[test]
fn test_generate_test_data_deterministic() {
    let data1 = generate_test_data(42, 100);
    let data2 = generate_test_data(42, 100);
    assert_eq!(data1, data2);
}

#[test]
fn test_generate_test_data_different_seeds() {
    let data1 = generate_test_data(42, 100);
    let data2 = generate_test_data(43, 100);
    assert_ne!(data1, data2);
}

#[test]
fn test_generate_model_payload_deterministic() {
    let payload1 = generate_model_payload(42, 256);
    let payload2 = generate_model_payload(42, 256);
    assert_eq!(payload1, payload2);
}

#[test]
fn test_generate_model_payload_size() {
    let payload = generate_model_payload(42, 256);
    // 256 f32 values * 4 bytes each = 1024 bytes
    assert_eq!(payload.len(), 256 * 4);
}

#[test]
fn test_recipe_metadata_builder() {
    let metadata = RecipeMetadata::from_name("test")
        .with_category("bundling")
        .with_objective("Learn model embedding")
        .with_feature("encryption");

    assert_eq!(metadata.name, "test");
    assert_eq!(metadata.category, Some("bundling".to_string()));
    assert_eq!(
        metadata.objective,
        Some("Learn model embedding".to_string())
    );
    assert_eq!(metadata.features, vec!["encryption"]);
}

#[test]
fn test_verify_idempotency() {
    let mut ctx = RecipeContext::new("idempotency_test").unwrap();

    let is_idempotent = ctx.verify_idempotency(|ctx| {
        use rand::Rng;
        ctx.rng().gen::<u64>()
    });

    assert!(is_idempotent, "Same RNG operations should be idempotent");
}

#[test]
fn test_temp_dir_cleanup() {
    let path = {
        let ctx = RecipeContext::new("cleanup_test").unwrap();
        ctx.temp_dir().to_path_buf()
    };
    // After ctx is dropped, temp dir should be cleaned up
    assert!(
        !path.exists(),
        "Temp directory should be cleaned up on drop"
    );
}

#[test]
fn test_elapsed_time() {
    let ctx = RecipeContext::new("elapsed_test").unwrap();
    std::thread::sleep(Duration::from_millis(10));
    let elapsed = ctx.elapsed();
    assert!(elapsed >= Duration::from_millis(10));
}

#[test]
fn test_with_metadata() {
    let metadata = RecipeMetadata::from_name("meta_test")
        .with_category("testing")
        .with_objective("Test metadata");

    let ctx = RecipeContext::with_metadata("meta_test", metadata).unwrap();
    assert_eq!(ctx.metadata().name, "meta_test");
    assert_eq!(ctx.metadata().category, Some("testing".to_string()));
    assert_eq!(ctx.metadata().objective, Some("Test metadata".to_string()));
}

#[test]
fn test_report() {
    let mut ctx = RecipeContext::new("report_test").unwrap();
    ctx.record_metric("count", 42);
    ctx.record_float_metric("ratio", 3.125);
    ctx.record_duration("elapsed", Duration::from_millis(100));
    ctx.record_string_metric("status", "ok");

    // Report should not error
    let result = ctx.report();
    assert!(result.is_ok());
}

#[test]
fn test_get_metric_missing() {
    let ctx = RecipeContext::new("missing_metric_test").unwrap();
    assert!(ctx.get_metric("nonexistent").is_none());
}

#[test]
fn test_duration_metric() {
    let mut ctx = RecipeContext::new("duration_test").unwrap();
    let duration = Duration::from_secs(1);
    ctx.record_duration("time", duration);

    match ctx.get_metric("time") {
        Some(MetricValue::Duration(d)) => assert_eq!(*d, duration),
        _ => panic!("Expected Duration metric"),
    }
}

#[test]
fn test_string_metric() {
    let mut ctx = RecipeContext::new("string_test").unwrap();
    ctx.record_string_metric("name", "test-value");

    match ctx.get_metric("name") {
        Some(MetricValue::String(s)) => assert_eq!(s, "test-value"),
        _ => panic!("Expected String metric"),
    }
}
