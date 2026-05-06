//! # Contracts-Macros YAML Kebab-Case Audit
//!
//! Verify YAML keys follow kebab-case convention (`some-key`, lower
//! ASCII letters/digits and hyphens). Returns offending keys.
//!
//! Demonstrates the **CMM.92** recipe for PMAT-188 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: kebab-case style (Python PEP-8 module naming, K8s
//!  manifest convention).
//!
//! Run with: cargo run --example contracts_macros_yaml_kebab_case_audit
//!
//! Added by PMAT-188 (catalog 1315→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CaseVerdict {
    Ok {
        offending: Vec<String>,
        ok_count: u32,
    },
    InvalidConfig,
}

pub fn audit(keys: &[&str]) -> CaseVerdict {
    if keys.is_empty() {
        return CaseVerdict::InvalidConfig;
    }
    let mut offending: Vec<String> = Vec::new();
    let mut ok_count = 0u32;
    for key in keys {
        if is_kebab_case(key) {
            ok_count += 1;
        } else {
            offending.push((*key).to_string());
        }
    }
    offending.sort();
    offending.dedup();
    CaseVerdict::Ok {
        offending,
        ok_count,
    }
}

fn is_kebab_case(s: &str) -> bool {
    if s.is_empty() || s.starts_with('-') || s.ends_with('-') || s.contains("--") {
        return false;
    }
    s.chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_kebab_case_audit")?;

    let keys = ["valid-key", "snake_case", "camelCase", "another-good-1"];
    println!("audit: {:?}", audit(&keys));
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
    fn kebab_case_passes() {
        let keys = ["my-key", "another-good-key"];
        let v = audit(&keys);
        if let CaseVerdict::Ok { offending, .. } = v {
            assert!(offending.is_empty());
        }
    }

    #[test]
    fn snake_case_flagged() {
        let keys = ["snake_case"];
        let v = audit(&keys);
        if let CaseVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["snake_case".to_string()]);
        }
    }

    #[test]
    fn camel_case_flagged() {
        let keys = ["camelCase"];
        let v = audit(&keys);
        if let CaseVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["camelCase".to_string()]);
        }
    }

    #[test]
    fn upper_letters_flagged() {
        let keys = ["MY-KEY"];
        let v = audit(&keys);
        if let CaseVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["MY-KEY".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), CaseVerdict::InvalidConfig);
    }

    #[test]
    fn leading_dash_flagged() {
        let keys = ["-leading"];
        let v = audit(&keys);
        if let CaseVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["-leading".to_string()]);
        }
    }

    #[test]
    fn trailing_dash_flagged() {
        let keys = ["trailing-"];
        let v = audit(&keys);
        if let CaseVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["trailing-".to_string()]);
        }
    }

    #[test]
    fn double_dash_flagged() {
        let keys = ["double--dash"];
        let v = audit(&keys);
        if let CaseVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["double--dash".to_string()]);
        }
    }

    #[test]
    fn ok_count_correct() {
        let keys = ["good-key", "BadKey"];
        let v = audit(&keys);
        if let CaseVerdict::Ok { ok_count, .. } = v {
            assert_eq!(ok_count, 1);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&["good-key"]);
        let r2 = audit(&["good-key"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn digits_allowed() {
        let keys = ["v1-key", "key-2-foo"];
        let v = audit(&keys);
        if let CaseVerdict::Ok { offending, .. } = v {
            assert!(offending.is_empty());
        }
    }

    #[test]
    fn offending_sorted() {
        let keys = ["zeta_bad", "alpha_bad"];
        let v = audit(&keys);
        if let CaseVerdict::Ok { offending, .. } = v {
            assert_eq!(
                offending,
                vec!["alpha_bad".to_string(), "zeta_bad".to_string()]
            );
        }
    }
}
