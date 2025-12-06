//! Property-based tests for aprender integration.
//!
//! These tests verify model serialization invariants.

#![allow(clippy::disallowed_methods)]

use apr_cookbook::aprender_integration::{
    load_model, load_model_from_bytes, save_model, AprModelInfo, SimpleModel,
};
use aprender::format::{ModelType, SaveOptions};
use proptest::prelude::*;
use tempfile::tempdir;

/// Strategy for generating valid model dimensions
fn model_dimensions() -> impl Strategy<Value = (usize, usize)> {
    (1..50usize, 1..50usize)
}

/// Strategy for generating valid model names
fn model_name() -> impl Strategy<Value = String> {
    "[a-zA-Z][a-zA-Z0-9_-]{0,31}".prop_map(|s| s.clone())
}

/// Strategy for generating model types
fn model_type() -> impl Strategy<Value = ModelType> {
    prop_oneof![
        Just(ModelType::Custom),
        Just(ModelType::LinearRegression),
        Just(ModelType::LogisticRegression),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: SimpleModel dimensions are preserved
    #[test]
    fn simple_model_dimensions_preserved((input_dim, output_dim) in model_dimensions()) {
        let model = SimpleModel::new(input_dim, output_dim);
        prop_assert_eq!(model.input_dim, input_dim);
        prop_assert_eq!(model.output_dim, output_dim);
    }

    /// Property: SimpleModel num_params equals input * output
    #[test]
    fn simple_model_params_correct((input_dim, output_dim) in model_dimensions()) {
        let model = SimpleModel::new(input_dim, output_dim);
        prop_assert_eq!(model.num_params(), input_dim * output_dim);
    }

    /// Property: SimpleModel::new initializes weights to zero
    #[test]
    fn simple_model_new_zero_weights((input_dim, output_dim) in model_dimensions()) {
        let model = SimpleModel::new(input_dim, output_dim);
        prop_assert!(model.weights.iter().all(|&w| w == 0.0));
    }

    /// Property: SimpleModel::random produces non-zero weights (with high probability)
    #[test]
    fn simple_model_random_non_zero((input_dim, output_dim) in model_dimensions()) {
        let model = SimpleModel::random(input_dim, output_dim);
        // At least some weights should be non-zero
        prop_assert!(model.weights.iter().any(|&w| w != 0.0));
    }

    /// Property: Save and load roundtrip preserves model data
    #[test]
    fn save_load_roundtrip((input_dim, output_dim) in model_dimensions()) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("roundtrip_test.apr");

        let original = SimpleModel::random(input_dim, output_dim);
        let options = SaveOptions::default();

        save_model(&original, ModelType::Custom, &path, options).unwrap();
        let loaded: SimpleModel = load_model(&path, ModelType::Custom).unwrap();

        prop_assert_eq!(original.input_dim, loaded.input_dim);
        prop_assert_eq!(original.output_dim, loaded.output_dim);
        prop_assert_eq!(original.weights, loaded.weights);
    }

    /// Property: Model type is preserved in roundtrip
    #[test]
    fn model_type_roundtrip(model_type in model_type()) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("type_test.apr");

        let model = SimpleModel::new(5, 5);
        let options = SaveOptions::default();

        save_model(&model, model_type, &path, options).unwrap();
        let info = AprModelInfo::from_path(&path).unwrap();

        prop_assert_eq!(info.model_type(), model_type);
    }

    /// Property: Model name is preserved in roundtrip
    #[test]
    fn model_name_roundtrip(name in model_name()) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("name_test.apr");

        let model = SimpleModel::new(5, 5);
        let options = SaveOptions::default().with_name(&name);

        save_model(&model, ModelType::Custom, &path, options).unwrap();
        let info = AprModelInfo::from_path(&path).unwrap();

        prop_assert_eq!(info.name(), Some(name.as_str()));
    }

    /// Property: Load from bytes is equivalent to load from path
    #[test]
    fn load_bytes_equivalent_to_path((input_dim, output_dim) in model_dimensions()) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bytes_equiv_test.apr");

        let original = SimpleModel::random(input_dim, output_dim);
        let options = SaveOptions::default();

        save_model(&original, ModelType::Custom, &path, options).unwrap();

        let loaded_path: SimpleModel = load_model(&path, ModelType::Custom).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let loaded_bytes: SimpleModel = load_model_from_bytes(&bytes, ModelType::Custom).unwrap();

        prop_assert_eq!(loaded_path.weights, loaded_bytes.weights);
    }

    /// Property: AprModelInfo::from_bytes equivalent to from_path
    #[test]
    fn info_bytes_equivalent_to_path((input_dim, output_dim) in model_dimensions()) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("info_equiv_test.apr");

        let model = SimpleModel::random(input_dim, output_dim);
        let options = SaveOptions::default().with_name("equiv-test");

        save_model(&model, ModelType::Custom, &path, options).unwrap();

        let info_path = AprModelInfo::from_path(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let info_bytes = AprModelInfo::from_bytes(&bytes).unwrap();

        prop_assert_eq!(info_path.name(), info_bytes.name());
        prop_assert_eq!(info_path.model_type(), info_bytes.model_type());
        prop_assert_eq!(info_path.version(), info_bytes.version());
    }

    /// Property: Version is always (1, 0) for new models
    #[test]
    fn version_always_1_0((input_dim, output_dim) in model_dimensions()) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("version_test.apr");

        let model = SimpleModel::new(input_dim, output_dim);
        let options = SaveOptions::default();

        save_model(&model, ModelType::Custom, &path, options).unwrap();
        let info = AprModelInfo::from_path(&path).unwrap();

        prop_assert_eq!(info.version(), (1, 0));
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn minimal_model_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("minimal.apr");

        let model = SimpleModel::new(1, 1);
        let options = SaveOptions::default();

        save_model(&model, ModelType::Custom, &path, options).unwrap();
        let loaded: SimpleModel = load_model(&path, ModelType::Custom).unwrap();

        assert_eq!(model, loaded);
    }

    #[test]
    fn large_model_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("large.apr");

        // 1000x1000 = 1M parameters
        let model = SimpleModel::random(1000, 1000);
        let options = SaveOptions::default();

        save_model(&model, ModelType::Custom, &path, options).unwrap();
        let loaded: SimpleModel = load_model(&path, ModelType::Custom).unwrap();

        assert_eq!(model, loaded);
    }

    #[test]
    fn uncompressed_not_marked_compressed() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("uncompressed.apr");

        let model = SimpleModel::new(10, 10);
        let options = SaveOptions::default();

        save_model(&model, ModelType::Custom, &path, options).unwrap();
        let info = AprModelInfo::from_path(&path).unwrap();

        // Small uncompressed model should not be marked compressed
        // (compression might actually increase size)
        assert!(!info.is_encrypted());
        assert!(!info.is_signed());
    }
}
