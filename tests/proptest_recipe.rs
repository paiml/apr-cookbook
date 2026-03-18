//! Property-based tests for recipe infrastructure.

#![allow(clippy::disallowed_methods)]

use apr_cookbook::recipe::{
    generate_model_payload, generate_test_data, hash_name_to_seed, MetricValue, RecipeContext,
    RecipeMetadata,
};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // 1. hash_name_to_seed is deterministic
    #[test]
    fn prop_hash_deterministic(name in "[a-z_]{1,50}") {
        let s1 = hash_name_to_seed(&name);
        let s2 = hash_name_to_seed(&name);
        prop_assert_eq!(s1, s2);
    }

    // 2. Different names produce different seeds (probabilistic, but practically true)
    #[test]
    fn prop_hash_different_names(
        name1 in "[a-z]{3,20}",
        name2 in "[A-Z]{3,20}"
    ) {
        // Different character sets ensures different names
        let s1 = hash_name_to_seed(&name1);
        let s2 = hash_name_to_seed(&name2);
        prop_assert_ne!(s1, s2);
    }

    // 3. RecipeContext creation succeeds for valid names
    #[test]
    fn prop_context_creation(name in "[a-z_]{1,30}") {
        let ctx = RecipeContext::new(&name);
        prop_assert!(ctx.is_ok());
        let ctx = ctx.unwrap();
        prop_assert_eq!(ctx.name(), name.as_str());
    }

    // 4. Temp dirs are isolated
    #[test]
    fn prop_temp_dirs_isolated(
        name1 in "[a-z]{3,10}",
        name2 in "[A-Z]{3,10}"
    ) {
        let ctx1 = RecipeContext::new(&name1).unwrap();
        let ctx2 = RecipeContext::new(&name2).unwrap();
        prop_assert_ne!(ctx1.temp_dir(), ctx2.temp_dir());
    }

    // 5. generate_test_data is deterministic
    #[test]
    fn prop_test_data_deterministic(seed in 0u64..10000, size in 1usize..500) {
        let d1 = generate_test_data(seed, size);
        let d2 = generate_test_data(seed, size);
        prop_assert_eq!(d1, d2);
    }

    // 6. generate_test_data has correct length
    #[test]
    fn prop_test_data_length(seed in 0u64..10000, size in 0usize..1000) {
        let data = generate_test_data(seed, size);
        prop_assert_eq!(data.len(), size);
    }

    // 7. generate_test_data values in range [-1, 1]
    #[test]
    fn prop_test_data_bounded(seed in 0u64..10000, size in 1usize..500) {
        let data = generate_test_data(seed, size);
        for &v in &data {
            prop_assert!((-1.0..1.0).contains(&v));
        }
    }

    // 8. generate_model_payload size is n_params * 4
    #[test]
    fn prop_model_payload_size(seed in 0u64..10000, n_params in 1usize..500) {
        let payload = generate_model_payload(seed, n_params);
        prop_assert_eq!(payload.len(), n_params * 4);
    }

    // 9. generate_model_payload is deterministic
    #[test]
    fn prop_model_payload_deterministic(seed in 0u64..10000, n_params in 1usize..200) {
        let p1 = generate_model_payload(seed, n_params);
        let p2 = generate_model_payload(seed, n_params);
        prop_assert_eq!(p1, p2);
    }

    // 10. Metrics can be recorded and retrieved
    #[test]
    fn prop_metrics_roundtrip(name in "[a-z]{3,10}", value in -1000000i64..1000000) {
        let mut ctx = RecipeContext::new("prop_metrics").unwrap();
        ctx.record_metric(&name, value);
        match ctx.get_metric(&name) {
            Some(MetricValue::Int(v)) => prop_assert_eq!(*v, value),
            _ => prop_assert!(false, "Expected Int metric"),
        }
    }

    // 11. RecipeMetadata builder works
    #[test]
    fn prop_metadata_builder(
        name in "[a-z]{3,20}",
        category in "[a-z]{3,15}",
        feature in "[a-z]{3,10}"
    ) {
        let meta = RecipeMetadata::from_name(&name)
            .with_category(&category)
            .with_feature(&feature);
        prop_assert_eq!(meta.name, name);
        prop_assert_eq!(meta.category, Some(category));
        prop_assert_eq!(meta.features, vec![feature]);
    }

    // 12. verify_idempotency returns true for pure RNG operations
    #[test]
    fn prop_idempotency_holds(name in "[a-z]{3,20}") {
        let mut ctx = RecipeContext::new(&name).unwrap();
        use rand::Rng;
        let ok = ctx.verify_idempotency(|ctx| -> Vec<u64> {
            (0..5).map(|_| ctx.rng().gen()).collect()
        });
        prop_assert!(ok);
    }
}
