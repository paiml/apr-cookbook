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
    fn prop_lazy_loads_less_bytes(fraction in 0.01f64..0.99) {
        let ctx = RecipeContext::new("prop_lazy_less").expect("ctx");
        let path = ctx.path("model.bin");
        let mut rng = rand::SeedableRng::seed_from_u64(77);
        let tensors = create_test_model(&path, 2, &mut rng).expect("create");

        let n_to_load = ((tensors.len() as f64 * fraction) as usize).max(1).min(tensors.len());
        let indices: Vec<usize> = (0..n_to_load).collect();

        let (_, eager_bytes) = benchmark_eager_load(&path).expect("eager");
        let (_, lazy_bytes) = benchmark_lazy_load(&path, &tensors, &indices).expect("lazy");

        // Lazy should read at most the tensor data we asked for, which is
        // always less than the full file as long as we skip at least one tensor.
        if n_to_load < tensors.len() {
            prop_assert!(
                lazy_bytes < eager_bytes,
                "lazy={} should < eager={} at fraction={:.2}",
                lazy_bytes, eager_bytes, fraction
            );
        }
    }

    #[test]
    fn prop_tensor_roundtrip(seed in 0u64..10_000) {
        let ctx = RecipeContext::new("prop_roundtrip").expect("ctx");
        let path = ctx.path("model.bin");
        let mut rng = rand::SeedableRng::seed_from_u64(seed);
        let tensors = create_test_model(&path, 1, &mut rng).expect("create");

        // Load first tensor twice -- must produce identical f32 data
        let mut loader = simulate_mmap_load(&path, &tensors).expect("mmap");
        let data_a = lazy_load_tensor(&mut loader, &tensors[0]).expect("a");
        let data_b = lazy_load_tensor(&mut loader, &tensors[0]).expect("b");

        prop_assert_eq!(data_a.len(), data_b.len());
        for (i, (a, b)) in data_a.iter().zip(data_b.iter()).enumerate() {
            prop_assert!(
                (a - b).abs() < f32::EPSILON,
                "mismatch at index {}: {} vs {}", i, a, b
            );
        }
    }

    #[test]
    fn prop_deterministic_loads(seed in 0u64..10_000) {
        // Two independent create + load cycles with the same seed
        // must yield identical tensor content.
        let ctx1 = RecipeContext::new("prop_det_1").expect("ctx1");
        let path1 = ctx1.path("model.bin");
        let mut rng1 = rand::SeedableRng::seed_from_u64(seed);
        let tensors1 = create_test_model(&path1, 1, &mut rng1).expect("c1");

        let ctx2 = RecipeContext::new("prop_det_2").expect("ctx2");
        let path2 = ctx2.path("model.bin");
        let mut rng2 = rand::SeedableRng::seed_from_u64(seed);
        let tensors2 = create_test_model(&path2, 1, &mut rng2).expect("c2");

        let mut loader1 = simulate_mmap_load(&path1, &tensors1).expect("m1");
        let mut loader2 = simulate_mmap_load(&path2, &tensors2).expect("m2");

        let d1 = lazy_load_tensor(&mut loader1, &tensors1[0]).expect("l1");
        let d2 = lazy_load_tensor(&mut loader2, &tensors2[0]).expect("l2");

        prop_assert_eq!(d1.len(), d2.len());
        for (i, (a, b)) in d1.iter().zip(d2.iter()).enumerate() {
            prop_assert!(
                (a - b).abs() < f32::EPSILON,
                "determinism broken at index {}: {} vs {}", i, a, b
            );
        }
    }
}
