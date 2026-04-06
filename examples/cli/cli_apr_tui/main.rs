#![allow(unused_imports)]
//! # Recipe: APR Model TUI (Headless Simulation)
//! **CLI Equivalent**: `apr tui`
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/cli-parity-v1.yaml
//!
//! Simulate a terminal UI for interactive model exploration, rendered in
//! headless mode. Mirrors `apr tui` with 4 tabs: Overview, Tensors, Stats, Help.
//!
//! ```bash
//! cargo run --example cli_apr_tui
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr inspect model.apr          # APR native format
//! apr inspect model.gguf         # GGUF (llama.cpp compatible)
//! apr inspect model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Amershi, S. et al. (2019). *Software Engineering for Machine Learning: A Case Study*. ICSE. DOI: 10.1109/ICSE-SEIP.2019.00042

use apr_cookbook::prelude::*;
use rand::Rng;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("cli_apr_tui")?;
    println!("=== apr tui (Headless Mode) ===\n");
    let tensors = generate_tensors(ctx.rng(), NUM_TENSORS);
    let mut state = AppState {
        current_tab: Tab::Overview,
        model_name: "llama-3.2-1b.apr".to_string(),
        model_version: "1.0.0".to_string(),
        format_version: "2.0".to_string(),
        creation_date: "2026-02-25T12:00:00Z".to_string(),
        tensors,
        page: 0,
        page_size: PAGE_SIZE,
    };
    println!(
        "Model: {}  Tensors: {}  Params: {}  Size: {}",
        state.model_name,
        state.tensors.len(),
        format_params(state.total_params()),
        format_bytes(state.total_size_bytes())
    );
    let tensor_stats = generate_tensor_stats(ctx.rng(), &state.tensors);
    println!("\n--- Simulating TUI Navigation ---");
    simulate_navigation(&mut state, &tensor_stats);
    ctx.record_metric("tensor_count", state.tensors.len() as i64);
    ctx.record_metric("total_params", state.total_params() as i64);
    ctx.record_metric("total_size_bytes", state.total_size_bytes() as i64);
    ctx.record_float_metric("compression_ratio", state.compression_ratio());
    ctx.record_metric("total_pages", state.total_pages() as i64);
    println!();
    ctx.report()?;
    Ok(())
}

// ---- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn test_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn sample_state() -> AppState {
        AppState {
            current_tab: Tab::Overview,
            model_name: "test-model.apr".to_string(),
            model_version: "1.0.0".to_string(),
            format_version: "2.0".to_string(),
            creation_date: "2026-01-01T00:00:00Z".to_string(),
            tensors: vec![
                TensorInfo {
                    name: "layer.0.weight".to_string(),
                    shape: vec![768, 768],
                    dtype: "fp32".to_string(),
                    size_bytes: 768 * 768 * 4,
                },
                TensorInfo {
                    name: "layer.1.weight".to_string(),
                    shape: vec![768, 3072],
                    dtype: "fp16".to_string(),
                    size_bytes: 768 * 3072 * 2,
                },
            ],
            page: 0,
            page_size: PAGE_SIZE,
        }
    }

    #[test]
    fn test_tab_navigation_and_labels() {
        for (i, tab) in Tab::ALL.iter().enumerate() {
            assert_eq!(tab.index(), i + 1);
            assert!(!tab.label().is_empty());
        }
        let mut t = Tab::Overview;
        for expected in [Tab::Tensors, Tab::Stats, Tab::Help, Tab::Overview] {
            t = t.next();
            assert_eq!(t, expected);
        }
    }

    #[test]
    fn test_app_state_computations() {
        let state = sample_state();
        assert_eq!(state.total_params(), 768 * 768 + 768 * 3072);
        assert_eq!(state.total_size_bytes(), 768 * 768 * 4 + 768 * 3072 * 2);
        let expected_ratio = (state.total_params() * 4) as f64 / state.total_size_bytes() as f64;
        assert!((state.compression_ratio() - expected_ratio).abs() < 1e-10);
        assert_eq!(state.total_pages(), 1);
    }

    #[test]
    fn test_pagination() {
        let mut rng = test_rng();
        let mut state = AppState {
            tensors: generate_tensors(&mut rng, 24),
            page: 0,
            page_size: 10,
            current_tab: Tab::Tensors,
            model_name: "t".into(),
            model_version: "1.0".into(),
            format_version: "2.0".into(),
            creation_date: "2026-01-01".into(),
        };
        assert_eq!(state.total_pages(), 3);
        assert_eq!(state.current_page_tensors().len(), 10);
        state.page = 2;
        assert_eq!(state.current_page_tensors().len(), 4);
        state.page = 3;
        assert_eq!(state.current_page_tensors().len(), 0);
    }

    #[test]
    fn test_format_bytes_and_params() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1_048_576), "1.00 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.00 GB");
        assert_eq!(format_params(500), "500");
        assert_eq!(format_params(1_500), "1.50K");
        assert_eq!(format_params(2_500_000), "2.50M");
        assert_eq!(format_params(7_000_000_000), "7.00B");
    }

    #[test]
    fn test_render_tab_bar_and_bar() {
        let bar = render_tab_bar(Tab::Tensors);
        assert!(bar.contains("*Tensors*"));
        assert!(!bar.contains("*Overview*"));
        assert_eq!(render_bar(0.0, 20).matches('#').count(), 0);
        assert_eq!(render_bar(1.0, 20).matches('#').count(), 20);
        assert_eq!(render_bar(2.0, 20).matches('#').count(), 20); // clamp
    }

    #[test]
    fn test_generate_tensors_and_stats() {
        let mut rng = test_rng();
        let tensors = generate_tensors(&mut rng, 12);
        assert_eq!(tensors.len(), 12);
        for t in &tensors {
            assert!(!t.name.is_empty());
            assert!(t.size_bytes > 0);
        }
        let mut rng2 = test_rng();
        let t2 = generate_tensors(&mut rng2, 5);
        let s1 = generate_tensor_stats(&mut rng, &tensors);
        let s2 = generate_tensor_stats(&mut rng2, &t2);
        assert_eq!(s1.len(), 12);
        assert_eq!(s2.len(), 5);
    }

    #[test]
    fn test_render_frame_structure() {
        let frame = render_frame("Test", &["Hello".to_string()], 40);
        assert!(frame[0].starts_with(BOX_TL));
        assert!(frame[0].contains("Test"));
        assert!(frame.last().unwrap().starts_with(BOX_BL));
    }

    #[test]
    fn test_render_overview_and_help() {
        let state = sample_state();
        let ov = render_overview(&state).join("\n");
        assert!(ov.contains("test-model.apr"));
        assert!(ov.contains("Compression"));
        let help = render_help().join("\n");
        assert!(help.contains("Tab / Right Arrow"));
        assert!(help.contains("Quit"));
    }

    #[test]
    fn test_compression_ratio_empty() {
        let state = AppState {
            current_tab: Tab::Overview,
            model_name: "empty".into(),
            model_version: "0.0".into(),
            format_version: "1.0".into(),
            creation_date: "2026-01-01".into(),
            tensors: vec![],
            page: 0,
            page_size: PAGE_SIZE,
        };
        assert!((state.compression_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_buckets_empty() {
        assert!(compute_histogram_buckets(&[], |s| s.mean, 5, -1.0, 1.0).is_empty());
    }

    #[test]
    fn test_dtype_distribution_counts() {
        let tensors = vec![
            TensorInfo {
                name: "a".into(),
                shape: vec![10],
                dtype: "fp32".into(),
                size_bytes: 40,
            },
            TensorInfo {
                name: "b".into(),
                shape: vec![10],
                dtype: "fp32".into(),
                size_bytes: 40,
            },
            TensorInfo {
                name: "c".into(),
                shape: vec![10],
                dtype: "fp16".into(),
                size_bytes: 20,
            },
        ];
        let dist = dtype_distribution(&tensors);
        assert!(dist.contains("fp32: 2"));
        assert!(dist.contains("fp16: 1"));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use rand::SeedableRng;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_format_bytes_never_empty(bytes in 0usize..10_000_000_000) { prop_assert!(!format_bytes(bytes).is_empty()); }

        #[test]
        fn prop_render_bar_correct_length(fraction in -2.0f64..2.0, width in 1usize..50) { prop_assert_eq!(render_bar(fraction, width).len(), width + 2); }

        #[test]
        fn prop_total_pages_covers_all(n in 0usize..100, ps in 1usize..20) {
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            let state = AppState { current_tab: Tab::Overview, model_name: "t".into(), model_version: "1.0".into(),
                format_version: "1.0".into(), creation_date: "2026-01-01".into(), tensors: generate_tensors(&mut rng, n), page: 0, page_size: ps };
            prop_assert!(state.total_pages() * ps >= n);
        }
    }
}
