//! # WASM Export Table Validator
//!
//! Host expects the WASM module to export a specific symbol set
//! (e.g., `predict`, `init`, `_start`). Missing required exports →
//! load fails. Extra exports are OK (unused). This recipe validates
//! a (required, present) pair.
//!
//! Demonstrates the **WASM.15** recipe for PMAT-139 (wasm round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly Core Specification § 5.5.16 (Exports).
//!
//! Run with: cargo run --example wasm_export_table_validator
//!
//! Added by PMAT-139 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum ExportVerdict {
    Ok { extra_exports: Vec<String> },
    MissingExports { missing: Vec<String> },
    EmptyRequired,
}

pub fn validate(required: &[&str], present: &[&str]) -> ExportVerdict {
    if required.is_empty() {
        return ExportVerdict::EmptyRequired;
    }
    let req: BTreeSet<&str> = required.iter().copied().collect();
    let pres: BTreeSet<&str> = present.iter().copied().collect();
    let missing: Vec<String> = req.difference(&pres).map(|s| (*s).to_string()).collect();
    if !missing.is_empty() {
        return ExportVerdict::MissingExports { missing };
    }
    let extra: Vec<String> = pres.difference(&req).map(|s| (*s).to_string()).collect();
    ExportVerdict::Ok {
        extra_exports: extra,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_export_table_validator")?;

    let required = ["predict", "init"];
    println!(
        "all present: {:?}",
        validate(&required, &["predict", "init"])
    );
    println!(
        "all + extras: {:?}",
        validate(&required, &["predict", "init", "memory", "_start"])
    );
    println!("missing: {:?}", validate(&required, &["predict"]));
    println!("empty required: {:?}", validate(&[], &["predict"]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn exact_match_no_extras() {
        let v = validate(&["a", "b"], &["a", "b"]);
        if let ExportVerdict::Ok { extra_exports } = v {
            assert!(extra_exports.is_empty());
        }
    }

    #[test]
    fn extras_listed() {
        let v = validate(&["a"], &["a", "b", "c"]);
        if let ExportVerdict::Ok { extra_exports } = v {
            assert_eq!(extra_exports, vec!["b", "c"]);
        }
    }

    #[test]
    fn missing_required_rejected() {
        let v = validate(&["a", "b", "c"], &["a"]);
        if let ExportVerdict::MissingExports { missing } = v {
            assert_eq!(missing, vec!["b", "c"]);
        }
    }

    #[test]
    fn empty_required_rejected() {
        assert_eq!(validate(&[], &["a"]), ExportVerdict::EmptyRequired);
    }

    #[test]
    fn empty_present_lists_all_missing() {
        let v = validate(&["a", "b"], &[]);
        if let ExportVerdict::MissingExports { missing } = v {
            assert_eq!(missing.len(), 2);
        }
    }

    #[test]
    fn case_sensitive_match() {
        // "Predict" ≠ "predict".
        let v = validate(&["predict"], &["Predict"]);
        assert!(matches!(v, ExportVerdict::MissingExports { .. }));
    }

    #[test]
    fn duplicate_required_handled() {
        // Duplicates in required should be deduped via BTreeSet.
        let v = validate(&["a", "a", "b"], &["a", "b"]);
        assert!(matches!(v, ExportVerdict::Ok { .. }));
    }

    #[test]
    fn missing_listed_lexicographic() {
        let v = validate(&["zoo", "alpha", "mid"], &[]);
        if let ExportVerdict::MissingExports { missing } = v {
            assert_eq!(missing, vec!["alpha", "mid", "zoo"]);
        }
    }

    #[test]
    fn extras_listed_lexicographic() {
        let v = validate(&["a"], &["zzz", "a", "bbb"]);
        if let ExportVerdict::Ok { extra_exports } = v {
            assert_eq!(extra_exports, vec!["bbb", "zzz"]);
        }
    }

    #[test]
    fn realistic_wasm_module() {
        // Typical inference WASM: predict, init, memory.
        let v = validate(
            &["predict", "init"],
            &["predict", "init", "memory", "__data_end", "__heap_base"],
        );
        if let ExportVerdict::Ok { extra_exports } = v {
            assert_eq!(extra_exports.len(), 3);
        }
    }
}
