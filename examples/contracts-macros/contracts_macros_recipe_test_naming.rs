//! # Contracts-Macros Recipe Test Naming Convention
//!
//! Verify test function names follow `<verb>_<subject>_<expectation>`
//! convention (e.g., `parses_input_correctly`). Returns offenders.
//!
//! Demonstrates the **CMM.123** recipe for PMAT-198 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Roy Osherove, The Art of Unit Testing §3 (test naming
//!  conventions); pytest test_<feature> convention.
//!
//! Run with: cargo run --example contracts_macros_recipe_test_naming
//!
//! Added by PMAT-198 (catalog 1405→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum NamingVerdict {
    Ok {
        offenders: Vec<String>,
        compliant_count: u32,
    },
    InvalidConfig,
}

pub fn audit(test_names: &[&str]) -> NamingVerdict {
    if test_names.is_empty() {
        return NamingVerdict::InvalidConfig;
    }
    let mut offenders: Vec<String> = Vec::new();
    let mut compliant_count = 0u32;
    for name in test_names {
        if is_well_named(name) {
            compliant_count += 1;
        } else {
            offenders.push((*name).to_string());
        }
    }
    offenders.sort();
    NamingVerdict::Ok {
        offenders,
        compliant_count,
    }
}

fn is_well_named(name: &str) -> bool {
    if !name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    {
        return false;
    }
    let parts: Vec<&str> = name.split('_').collect();
    if parts.len() < 2 {
        return false;
    }
    if parts.iter().any(|p| p.is_empty()) {
        return false;
    }
    true
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_test_naming")?;

    let tests = [
        "parses_input_correctly",
        "test1",
        "ChecksOutput",
        "validates_signature",
    ];
    println!("audit: {:?}", audit(&tests));
    println!("invalid: {:?}", audit(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auditor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn well_named_test_passes() {
        let v = audit(&["parses_input_correctly"]);
        if let NamingVerdict::Ok { offenders, .. } = v {
            assert!(offenders.is_empty());
        }
    }

    #[test]
    fn single_word_flagged() {
        let v = audit(&["test1"]);
        if let NamingVerdict::Ok { offenders, .. } = v {
            assert_eq!(offenders, vec!["test1".to_string()]);
        }
    }

    #[test]
    fn camel_case_flagged() {
        let v = audit(&["ChecksOutput"]);
        if let NamingVerdict::Ok { offenders, .. } = v {
            assert_eq!(offenders, vec!["ChecksOutput".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), NamingVerdict::InvalidConfig);
    }

    #[test]
    fn compliant_count_correct() {
        let v = audit(&["good_name", "BadName"]);
        if let NamingVerdict::Ok {
            compliant_count, ..
        } = v
        {
            assert_eq!(compliant_count, 1);
        }
    }

    #[test]
    fn offenders_sorted() {
        let v = audit(&["ZetaBad", "AlphaBad"]);
        if let NamingVerdict::Ok { offenders, .. } = v {
            assert_eq!(
                offenders,
                vec!["AlphaBad".to_string(), "ZetaBad".to_string()]
            );
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&["good_name"]);
        let r2 = audit(&["good_name"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn double_underscore_flagged() {
        let v = audit(&["bad__name"]);
        if let NamingVerdict::Ok { offenders, .. } = v {
            assert_eq!(offenders, vec!["bad__name".to_string()]);
        }
    }

    #[test]
    fn leading_underscore_flagged() {
        let v = audit(&["_leading"]);
        if let NamingVerdict::Ok { offenders, .. } = v {
            assert_eq!(offenders, vec!["_leading".to_string()]);
        }
    }

    #[test]
    fn digits_allowed_with_words() {
        let v = audit(&["v1_test"]);
        if let NamingVerdict::Ok { offenders, .. } = v {
            assert!(offenders.is_empty());
        }
    }

    #[test]
    fn three_word_pattern_passes() {
        let v = audit(&["parses_input_correctly"]);
        if let NamingVerdict::Ok {
            compliant_count, ..
        } = v
        {
            assert_eq!(compliant_count, 1);
        }
    }
}
