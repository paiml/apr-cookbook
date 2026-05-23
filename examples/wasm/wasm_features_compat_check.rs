//! # WASM Features Compatibility Check
//!
//! Validate that the module's required features are all enabled in
//! the engine. Returns sorted missing features and required-feature
//! count.
//!
//! Demonstrates the **WASM.X** recipe for PMAT-224 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly proposals registry; V8 wasm-engine feature
//!  detection.
//!
//! Run with: cargo run --example wasm_features_compat_check
//!
//! Added by PMAT-224 (catalog 1639→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum FeatureVerdict {
    Ok {
        missing_features: Vec<String>,
        required_count: u32,
    },
    InvalidConfig,
}

pub fn check(required: &[&str], engine: &[&str]) -> FeatureVerdict {
    if required.is_empty() {
        return FeatureVerdict::InvalidConfig;
    }
    let engine_set: BTreeSet<&str> = engine.iter().copied().collect();
    let missing: BTreeSet<String> = required
        .iter()
        .filter(|r| !engine_set.contains(*r))
        .map(|r| (*r).to_string())
        .collect();
    FeatureVerdict::Ok {
        missing_features: missing.into_iter().collect(),
        required_count: required.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_features_compat_check")?;

    let engine = ["simd128", "threads", "tail_call"];
    println!("ok: {:?}", check(&["simd128", "threads"], &engine));
    println!("missing: {:?}", check(&["simd128", "gc"], &engine));
    println!("invalid: {:?}", check(&[], &engine));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn all_supported_no_missing() {
        let v = check(&["a", "b"], &["a", "b", "c"]);
        if let FeatureVerdict::Ok {
            missing_features, ..
        } = v
        {
            assert!(missing_features.is_empty());
        }
    }

    #[test]
    fn missing_feature_flagged() {
        let v = check(&["a", "b"], &["a"]);
        if let FeatureVerdict::Ok {
            missing_features, ..
        } = v
        {
            assert_eq!(missing_features, vec!["b".to_string()]);
        }
    }

    #[test]
    fn empty_required_rejected() {
        assert_eq!(check(&[], &["a"]), FeatureVerdict::InvalidConfig);
    }

    #[test]
    fn empty_engine_all_missing() {
        let v = check(&["a", "b"], &[]);
        if let FeatureVerdict::Ok {
            missing_features, ..
        } = v
        {
            assert_eq!(missing_features.len(), 2);
        }
    }

    #[test]
    fn required_count_correct() {
        let v = check(&["a", "b", "c"], &["a"]);
        if let FeatureVerdict::Ok { required_count, .. } = v {
            assert_eq!(required_count, 3);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(&["a"], &["a"]);
        let r2 = check(&["a"], &["a"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn missing_sorted() {
        let v = check(&["zeta", "alpha"], &[]);
        if let FeatureVerdict::Ok {
            missing_features, ..
        } = v
        {
            assert_eq!(
                missing_features,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn extra_engine_features_ok() {
        let v = check(&["a"], &["a", "b", "c", "d"]);
        if let FeatureVerdict::Ok {
            missing_features, ..
        } = v
        {
            assert!(missing_features.is_empty());
        }
    }

    #[test]
    fn case_sensitive_feature() {
        let v = check(&["SIMD128"], &["simd128"]);
        if let FeatureVerdict::Ok {
            missing_features, ..
        } = v
        {
            assert_eq!(missing_features, vec!["SIMD128".to_string()]);
        }
    }

    #[test]
    fn many_features_handled() {
        let required: Vec<&str> = (0..20).map(|_| "feat").collect();
        let v = check(&required, &[]);
        if let FeatureVerdict::Ok {
            missing_features, ..
        } = v
        {
            // BTreeSet dedupes — single missing entry.
            assert_eq!(missing_features.len(), 1);
        }
    }

    #[test]
    fn unicode_feature_supported() {
        let v = check(&["café"], &["café"]);
        if let FeatureVerdict::Ok {
            missing_features, ..
        } = v
        {
            assert!(missing_features.is_empty());
        }
    }
}
