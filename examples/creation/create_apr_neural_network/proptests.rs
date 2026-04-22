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
