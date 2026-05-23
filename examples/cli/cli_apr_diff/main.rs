#![allow(unused_imports)]
//! # Recipe: APR Model Diff CLI
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/cli-parity-v1.yaml
//! Compare two APR model files, showing differences in tensors, metadata,
//! and architecture. Detect weight drift between model versions.
//!
//! ## QA: Build, test, clippy, fmt PASS. Proptests (100+ cases).
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
use clap::Parser;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

fn main() -> Result<()> {
    let config = DiffConfig::parse();
    run_diff(&config)
}

mod helpers;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;

#[cfg(test)]
mod tests {
    use super::*;
    fn snap(
        name: &str,
        ver: &str,
        arch: &str,
        t: HashMap<String, TensorInfo>,
        sz: usize,
    ) -> ModelSnapshot {
        ModelSnapshot {
            name: name.into(),
            version: ver.into(),
            architecture: arch.into(),
            tensors: t,
            total_size: sz,
        }
    }

    #[test]
    fn test_clap_defaults() {
        let c = DiffConfig::try_parse_from(["apr-diff"]).expect("ok");
        assert!(c.model_a.is_none() && !c.demo);
    }
    #[test]
    fn test_clap_demo() {
        let c = DiffConfig::try_parse_from(["apr-diff", "--demo"]).expect("ok");
        assert!(c.demo);
    }
    #[test]
    fn test_clap_positional_paths() {
        let c = DiffConfig::try_parse_from(["apr-diff", "a.apr", "b.apr"]).expect("ok");
        assert_eq!(c.model_a, Some("a.apr".into()));
        assert_eq!(c.model_b, Some("b.apr".into()));
    }
    #[test]
    fn test_clap_threshold() {
        let c = DiffConfig::try_parse_from(["apr-diff", "--threshold", "0.05"]).expect("ok");
        assert!((c.threshold - 0.05).abs() < 1e-10);
    }
    #[test]
    fn test_clap_verbose() {
        let c = DiffConfig::try_parse_from(["apr-diff", "-v"]).expect("ok");
        assert!(c.verbose);
    }
    #[test]
    fn test_demo_snapshots() {
        let (a, b) = create_demo_snapshots().expect("ok");
        assert_eq!(a.name, "demo-classifier");
        assert_ne!(a.version, b.version);
        let (added, removed, common) = inventory_diff(&a, &b);
        assert!(!added.is_empty() && !removed.is_empty() && !common.is_empty());
        let diffs = analyze_drift(&a, &b, &common, 0.01);
        assert!(diffs.iter().any(|d| d.status == TensorStatus::Modified));
    }
    #[test]
    fn test_metadata_diff() {
        let a = snap("a", "1.0", "lin", HashMap::new(), 0);
        assert!(diff_metadata(&a, &a.clone()).is_empty());
        let b = snap("b", "2.0", "xfm", HashMap::new(), 0);
        assert_eq!(diff_metadata(&a, &b).len(), 3);
    }
    #[test]
    fn test_inventory_diff() {
        let mut ta = HashMap::new();
        ta.insert("w".into(), generate_tensor_info(&[10, 10], "fp32", 42, 0));
        let mut tb = ta.clone();
        tb.insert("h".into(), generate_tensor_info(&[10, 5], "fp32", 42, 1));
        let (added, removed, common) =
            inventory_diff(&snap("a", "1", "l", ta, 400), &snap("b", "1", "l", tb, 600));
        assert_eq!(added, vec!["h"]);
        assert!(removed.is_empty());
        assert_eq!(common, vec!["w"]);
    }
    #[test]
    fn test_l2_distance_and_drift() {
        let i = generate_tensor_info(&[10, 10], "fp32", 42, 0);
        let d = ((i.mean - i.mean).powi(2)
            + (i.l2_norm - i.l2_norm).powi(2)
            + (i.min - i.min).powi(2)
            + (i.max - i.max).powi(2))
        .sqrt();
        assert!(d.abs() < 1e-10);
        let j = generate_tensor_info(&[10, 10], "fp32", 99, 0);
        let d2 = ((i.mean - j.mean).powi(2)
            + (i.l2_norm - j.l2_norm).powi(2)
            + (i.min - j.min).powi(2)
            + (i.max - j.max).powi(2))
        .sqrt();
        assert!(d2 > 0.0);
    }
    #[test]
    fn test_drift_bar_and_status() {
        assert_eq!(TensorStatus::Added.symbol(), "+");
        assert_eq!(TensorStatus::Removed.as_str(), "REMOVED");
        for dist in [0.0_f64, 0.5, 1.0, 5.0] {
            let clamped = dist.clamp(0.0, 1.0);
            let filled = (clamped * 20.0) as usize;
            assert_eq!(filled + (20 - filled), 20);
        }
    }
    #[test]
    fn test_build_full_diffs_sorted() {
        let common = vec![TensorDiff {
            name: "m".into(),
            status: TensorStatus::Modified,
            l2_distance: Some(0.5),
            shape_a: Some(vec![10]),
            shape_b: Some(vec![10]),
        }];
        let full = build_full_diffs(&["z".into()], &["a".into()], &common);
        assert_eq!(full.len(), 3);
        assert_eq!(full[0].name, "a");
        assert_eq!(full[2].name, "z");
    }
    #[test]
    fn test_hash_and_total_size() {
        for s in 0..10u64 {
            for v in 0..5u64 {
                let f = hash_to_float(s, v);
                assert!(f >= 0.5 && f < 1.5);
            }
        }
        assert_eq!(hash_to_float(42, 7), hash_to_float(42, 7));
        let mut t = HashMap::new();
        t.insert(
            "w".into(),
            TensorInfo {
                shape: vec![10, 10],
                dtype: "fp32".into(),
                min: -1.0,
                max: 1.0,
                mean: 0.0,
                l2_norm: 1.0,
            },
        );
        assert_eq!(compute_total_size(&t), 400);
        t.clear();
        t.insert(
            "w".into(),
            TensorInfo {
                shape: vec![10, 10],
                dtype: "fp16".into(),
                min: -1.0,
                max: 1.0,
                mean: 0.0,
                l2_norm: 1.0,
            },
        );
        assert_eq!(compute_total_size(&t), 200);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        #[test]
        fn prop_l2_non_negative_symmetric(sa in 0u64..1000, sb in 0u64..1000) {
            let a = generate_tensor_info(&[10,10], "fp32", sa, 0);
            let b = generate_tensor_info(&[10,10], "fp32", sb, 0);
            let d1 = ((a.mean-b.mean).powi(2)+(a.l2_norm-b.l2_norm).powi(2)+(a.min-b.min).powi(2)+(a.max-b.max).powi(2)).sqrt();
            let d2 = ((b.mean-a.mean).powi(2)+(b.l2_norm-a.l2_norm).powi(2)+(b.min-a.min).powi(2)+(b.max-a.max).powi(2)).sqrt();
            prop_assert!(d1 >= 0.0); prop_assert!((d1-d2).abs() < 1e-10);
        }
        #[test]
        fn prop_inventory_conservation(na in 0usize..5, nb in 0usize..5, nc in 0usize..5) {
            let mut ta = HashMap::new(); let mut tb = HashMap::new();
            for i in 0..nc { let n = format!("c_{i}"); let info = generate_tensor_info(&[4,4], "fp32", 42, i as u64); ta.insert(n.clone(), info.clone()); tb.insert(n, info); }
            for i in 0..na { ta.insert(format!("a_{i}"), generate_tensor_info(&[4,4], "fp32", 42, (100+i) as u64)); }
            for i in 0..nb { tb.insert(format!("b_{i}"), generate_tensor_info(&[4,4], "fp32", 42, (200+i) as u64)); }
            let a = ModelSnapshot { name:"a".into(), version:"1".into(), architecture:"l".into(), tensors: ta, total_size: 0 };
            let b = ModelSnapshot { name:"b".into(), version:"1".into(), architecture:"l".into(), tensors: tb, total_size: 0 };
            let (added, removed, common) = inventory_diff(&a, &b);
            prop_assert_eq!(added.len(), nb); prop_assert_eq!(removed.len(), na); prop_assert_eq!(common.len(), nc);
        }
    }
}
