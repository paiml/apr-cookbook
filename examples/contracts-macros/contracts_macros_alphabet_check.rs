//! # Contracts-Macros Free-Variable Alphabet Check
//!
//! Verify a contract's declared free variables match its actual use.
//! Returns Missing (used but not declared) or Unused (declared but
//! not used) or Ok.
//!
//! Demonstrates the **CMM.20** recipe for PMAT-164 (catalog crosses 1100).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Refinement-typed alphabet checks (LiquidHaskell).
//!
//! Run with: cargo run --example contracts_macros_alphabet_check
//!
//! Added by PMAT-164 (catalog 1099→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum AlphabetVerdict {
    Ok,
    Missing {
        vars: Vec<String>,
    },
    Unused {
        vars: Vec<String>,
    },
    BothMismatch {
        missing: Vec<String>,
        unused: Vec<String>,
    },
    EmptyDeclared,
}

pub fn check(declared: &[&str], used: &[&str]) -> AlphabetVerdict {
    if declared.is_empty() && used.is_empty() {
        return AlphabetVerdict::Ok;
    }
    if declared.is_empty() {
        return AlphabetVerdict::EmptyDeclared;
    }
    let dec: BTreeSet<&str> = declared.iter().copied().collect();
    let usd: BTreeSet<&str> = used.iter().copied().collect();
    let missing: Vec<String> = usd.difference(&dec).map(|s| (*s).to_string()).collect();
    let unused: Vec<String> = dec.difference(&usd).map(|s| (*s).to_string()).collect();
    match (missing.is_empty(), unused.is_empty()) {
        (true, true) => AlphabetVerdict::Ok,
        (false, true) => AlphabetVerdict::Missing { vars: missing },
        (true, false) => AlphabetVerdict::Unused { vars: unused },
        (false, false) => AlphabetVerdict::BothMismatch { missing, unused },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_alphabet_check")?;

    println!("ok: {:?}", check(&["x", "y"], &["x", "y"]));
    println!("missing: {:?}", check(&["x"], &["x", "z"]));
    println!("unused: {:?}", check(&["x", "y", "z"], &["x"]));
    println!("both: {:?}", check(&["x", "y"], &["x", "z"]));
    println!("both empty: {:?}", check(&[], &[]));
    println!("empty declared: {:?}", check(&[], &["x"]));
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
    fn matched_alphabets_ok() {
        assert_eq!(check(&["x", "y"], &["x", "y"]), AlphabetVerdict::Ok);
    }

    #[test]
    fn missing_var_reported() {
        let v = check(&["x"], &["x", "z"]);
        if let AlphabetVerdict::Missing { vars } = v {
            assert_eq!(vars, vec!["z".to_string()]);
        }
    }

    #[test]
    fn unused_var_reported() {
        let v = check(&["x", "y", "z"], &["x"]);
        if let AlphabetVerdict::Unused { vars } = v {
            assert!(vars.contains(&"y".to_string()));
            assert!(vars.contains(&"z".to_string()));
        }
    }

    #[test]
    fn both_mismatch_reported() {
        let v = check(&["x", "y"], &["x", "z"]);
        if let AlphabetVerdict::BothMismatch { missing, unused } = v {
            assert_eq!(missing, vec!["z".to_string()]);
            assert_eq!(unused, vec!["y".to_string()]);
        }
    }

    #[test]
    fn both_empty_ok() {
        assert_eq!(check(&[], &[]), AlphabetVerdict::Ok);
    }

    #[test]
    fn empty_declared_with_used_rejected() {
        assert_eq!(check(&[], &["x"]), AlphabetVerdict::EmptyDeclared);
    }

    #[test]
    fn duplicates_dedup() {
        let v = check(&["x", "x", "y"], &["x", "y", "y"]);
        assert_eq!(v, AlphabetVerdict::Ok);
    }

    #[test]
    fn case_sensitive() {
        let v = check(&["x"], &["X"]);
        assert!(matches!(v, AlphabetVerdict::BothMismatch { .. }));
    }

    #[test]
    fn unicode_var_names() {
        let v = check(&["α", "β"], &["α", "β"]);
        assert_eq!(v, AlphabetVerdict::Ok);
    }

    #[test]
    fn deterministic() {
        let a = check(&["x"], &["x", "z"]);
        let b = check(&["x"], &["x", "z"]);
        assert_eq!(a, b);
    }
}
