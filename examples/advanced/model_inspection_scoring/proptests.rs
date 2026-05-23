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
use super::*;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: Weight stats are always valid for non-empty input
    #[test]
    fn prop_weight_stats_valid(weights in proptest::collection::vec(-1000.0f32..1000.0, 1..1000)) {
        let stats = compute_weight_stats(&weights).expect("stats");
        prop_assert!(stats.min <= stats.max);
        prop_assert!(stats.sparsity >= 0.0 && stats.sparsity <= 1.0);
        prop_assert!(stats.nan_count == 0);
        prop_assert!(stats.inf_count == 0);
    }

    /// Property: Quality score is always in range [0, 100]
    #[test]
    fn prop_quality_score_range(
        compressed in any::<bool>(),
        signed in any::<bool>(),
        nan_count in 0usize..10,
    ) {
        let header = HeaderInfo {
            magic: "APRN".to_string(),
            version: (1, 0),
            flags: FeatureFlags { compressed, signed, ..Default::default() },
            compression_ratio: if compressed { 2.0 } else { 1.0 },
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
            nan_count,
            inf_count: 0,
            zero_count: 0,
            sparsity: 0.0,
        };

        let score = calculate_quality_score(&header, &stats).expect("score");
        prop_assert!(score.total <= 100);
    }

    /// Property: Model diff is symmetric in L2 distance
    #[test]
    fn prop_diff_l2_symmetric(
        weights_a in proptest::collection::vec(-10.0f32..10.0, 10..100),
    ) {
        let weights_b: Vec<f32> = weights_a.iter().map(|w| w + 0.1).collect();
        let diff_ab = compute_model_diff(&weights_a, &weights_b, "a", "b").expect("diff");
        let diff_ba = compute_model_diff(&weights_b, &weights_a, "b", "a").expect("diff");

        prop_assert!((diff_ab.total_l2_distance - diff_ba.total_l2_distance).abs() < 1e-6);
    }

    /// Property: CRC32 changes when weights change
    #[test]
    fn prop_crc32_changes(weights in proptest::collection::vec(-10.0f32..10.0, 10..100)) {
        let crc1 = compute_crc32(&weights);
        let mut modified = weights.clone();
        if !modified.is_empty() {
            modified[0] += 1.0;
        }
        let crc2 = compute_crc32(&modified);
        prop_assert_ne!(crc1, crc2, "CRC should change when weights change");
    }

    /// Property: Health status covers all score ranges
    #[test]
    fn prop_health_status_coverage(score in 0u8..=100) {
        let status = determine_health_status(score);
        match score {
            85..=100 => prop_assert_eq!(status, HealthStatus::Healthy),
            60..=84 => prop_assert_eq!(status, HealthStatus::Warning),
            0..=59 => prop_assert_eq!(status, HealthStatus::Critical),
            _ => unreachable!(),
        }
    }
}
