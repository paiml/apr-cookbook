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

    #[test]
    fn prop_accuracy_bounded(epoch in 1u32..100) {
        let config = SelfDistillConfig {
            temperature: 3.0,
            alpha_kl: 0.6,
            alpha_aux: 0.2,
            alpha_task: 0.2,
            epochs: 100,
        };

        let result = simulate_self_distill_epoch(epoch, &config).unwrap();

        prop_assert!(result.accuracy >= 0.0);
        prop_assert!(result.accuracy <= 1.0);
    }

    #[test]
    fn prop_loss_positive(epoch in 1u32..50) {
        let config = SelfDistillConfig {
            temperature: 3.0,
            alpha_kl: 0.6,
            alpha_aux: 0.2,
            alpha_task: 0.2,
            epochs: 50,
        };

        let result = simulate_self_distill_epoch(epoch, &config).unwrap();

        prop_assert!(result.kl_loss > 0.0);
        prop_assert!(result.aux_loss > 0.0);
        prop_assert!(result.task_loss > 0.0);
        prop_assert!(result.total_loss > 0.0);
    }

    #[test]
    fn prop_born_again_monotonic(num_gens in 2u32..10) {
        let config = SelfDistillConfig {
            temperature: 3.0,
            alpha_kl: 0.6,
            alpha_aux: 0.2,
            alpha_task: 0.2,
            epochs: 10,
        };

        let generations = simulate_born_again_generations(num_gens, &config).unwrap();

        for window in generations.windows(2) {
            prop_assert!(window[1].accuracy >= window[0].accuracy);
        }
    }

    #[test]
    fn prop_layer_accuracy_after_ge_before(num_layers in 2u32..20) {
        let model = SelfDistillModel {
            name: "prop-test".to_string(),
            num_layers,
            hidden_size: 256,
            num_classes: 10,
            params_millions: 10.0,
        };
        let config = SelfDistillConfig {
            temperature: 3.0,
            alpha_kl: 0.6,
            alpha_aux: 0.2,
            alpha_task: 0.2,
            epochs: 10,
        };

        let accuracies = compute_layer_accuracies(&model, &config).unwrap();

        for la in &accuracies {
            prop_assert!(la.after_sd >= la.before_sd);
            prop_assert!(la.after_sd <= 1.0);
            prop_assert!(la.before_sd >= 0.0);
        }
    }
}
