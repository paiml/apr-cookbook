//! # Contracts-Macros Recipe Pre-Publish Lint
//!
//! Run a battery of pre-publish lints on a recipe: check IIUR header
//! present, citation present, contract reference present. Returns
//! sorted missing-lint-checks.
//!
//! Demonstrates the **CMM.185** recipe for PMAT-219 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cargo publish pre-flight checks; npm pack dry-run lints.
//!
//! Run with: cargo run --example contracts_macros_recipe_pre_publish_lint
//!
//! Added by PMAT-219 (catalog 1594→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PrePublishVerdict {
    Ok {
        missing_lints: Vec<String>,
        passed_lints: u32,
    },
    InvalidConfig,
}

#[allow(clippy::fn_params_excessive_bools)]
pub fn check(
    has_iiur_header: bool,
    has_citation: bool,
    has_contract_ref: bool,
    has_test_module: bool,
) -> PrePublishVerdict {
    let mut missing: Vec<String> = Vec::new();
    let mut passed = 0u32;
    if has_iiur_header {
        passed += 1;
    } else {
        missing.push("iiur_header".to_string());
    }
    if has_citation {
        passed += 1;
    } else {
        missing.push("citation".to_string());
    }
    if has_contract_ref {
        passed += 1;
    } else {
        missing.push("contract_ref".to_string());
    }
    if has_test_module {
        passed += 1;
    } else {
        missing.push("test_module".to_string());
    }
    missing.sort();
    PrePublishVerdict::Ok {
        missing_lints: missing,
        passed_lints: passed,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_pre_publish_lint")?;

    println!("clean: {:?}", check(true, true, true, true));
    println!("missing-citation: {:?}", check(true, false, true, true));
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
    fn all_pass_no_missing() {
        let v = check(true, true, true, true);
        if let PrePublishVerdict::Ok { missing_lints, .. } = v {
            assert!(missing_lints.is_empty());
        }
    }

    #[test]
    fn missing_iiur_flagged() {
        let v = check(false, true, true, true);
        if let PrePublishVerdict::Ok { missing_lints, .. } = v {
            assert!(missing_lints.contains(&"iiur_header".to_string()));
        }
    }

    #[test]
    fn missing_citation_flagged() {
        let v = check(true, false, true, true);
        if let PrePublishVerdict::Ok { missing_lints, .. } = v {
            assert!(missing_lints.contains(&"citation".to_string()));
        }
    }

    #[test]
    fn missing_contract_ref_flagged() {
        let v = check(true, true, false, true);
        if let PrePublishVerdict::Ok { missing_lints, .. } = v {
            assert!(missing_lints.contains(&"contract_ref".to_string()));
        }
    }

    #[test]
    fn missing_test_module_flagged() {
        let v = check(true, true, true, false);
        if let PrePublishVerdict::Ok { missing_lints, .. } = v {
            assert!(missing_lints.contains(&"test_module".to_string()));
        }
    }

    #[test]
    fn passed_count_correct() {
        let v = check(true, false, true, true);
        if let PrePublishVerdict::Ok { passed_lints, .. } = v {
            assert_eq!(passed_lints, 3);
        }
    }

    #[test]
    fn all_fail_max_missing() {
        let v = check(false, false, false, false);
        if let PrePublishVerdict::Ok { missing_lints, .. } = v {
            assert_eq!(missing_lints.len(), 4);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(true, true, true, true);
        let r2 = check(true, true, true, true);
        assert_eq!(r1, r2);
    }

    #[test]
    fn missing_sorted() {
        let v = check(false, false, false, false);
        if let PrePublishVerdict::Ok { missing_lints, .. } = v {
            assert_eq!(
                missing_lints,
                vec![
                    "citation".to_string(),
                    "contract_ref".to_string(),
                    "iiur_header".to_string(),
                    "test_module".to_string(),
                ]
            );
        }
    }

    #[test]
    fn passed_lints_zero_when_all_fail() {
        let v = check(false, false, false, false);
        if let PrePublishVerdict::Ok { passed_lints, .. } = v {
            assert_eq!(passed_lints, 0);
        }
    }

    #[test]
    fn passed_lints_four_when_all_pass() {
        let v = check(true, true, true, true);
        if let PrePublishVerdict::Ok { passed_lints, .. } = v {
            assert_eq!(passed_lints, 4);
        }
    }

    #[test]
    fn one_pass_three_missing() {
        let v = check(false, false, true, false);
        if let PrePublishVerdict::Ok {
            passed_lints,
            missing_lints,
        } = v
        {
            assert_eq!(passed_lints, 1);
            assert_eq!(missing_lints.len(), 3);
        }
    }
}
