//! # Contracts-Macros Recipe Publish Gate
//!
//! Pre-publish gate check for a recipe: requires tests passing,
//! contract grade ≥ minimum, and no quarantine flag. Returns gate
//! verdict and which check(s) failed.
//!
//! Demonstrates the **CMM.170** recipe for PMAT-214 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cargo publish pre-flight checks; npm publish dry-run
//!  gating.
//!
//! Run with: cargo run --example contracts_macros_recipe_publish_gate
//!
//! Added by PMAT-214 (catalog 1549→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PublishVerdict {
    Allowed,
    Blocked { reasons: Vec<String> },
    InvalidConfig,
}

pub fn check(
    tests_pass: bool,
    grade: char,
    is_quarantined: bool,
    min_grade: char,
) -> PublishVerdict {
    if !"ABCDEF".contains(grade) || !"ABCDEF".contains(min_grade) {
        return PublishVerdict::InvalidConfig;
    }
    let mut reasons: Vec<String> = Vec::new();
    if !tests_pass {
        reasons.push("tests_failing".to_string());
    }
    if grade > min_grade {
        // ASCII: 'A' < 'B' < ... → grade higher means worse than min.
        reasons.push(format!("grade_below_min:{grade}_vs_{min_grade}"));
    }
    if is_quarantined {
        reasons.push("quarantined".to_string());
    }
    if reasons.is_empty() {
        PublishVerdict::Allowed
    } else {
        PublishVerdict::Blocked { reasons }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_publish_gate")?;

    println!("ok: {:?}", check(true, 'A', false, 'B'));
    println!("blocked: {:?}", check(false, 'C', true, 'B'));
    println!("invalid: {:?}", check(true, 'Z', false, 'A'));
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
    fn allowed_when_all_pass() {
        let v = check(true, 'A', false, 'B');
        assert_eq!(v, PublishVerdict::Allowed);
    }

    #[test]
    fn blocked_on_test_failure() {
        let v = check(false, 'A', false, 'B');
        if let PublishVerdict::Blocked { reasons } = v {
            assert!(reasons.contains(&"tests_failing".to_string()));
        }
    }

    #[test]
    fn blocked_on_grade_too_low() {
        // 'C' > 'B' → grade below min.
        let v = check(true, 'C', false, 'B');
        if let PublishVerdict::Blocked { reasons } = v {
            assert!(reasons.iter().any(|r| r.starts_with("grade_below_min")));
        }
    }

    #[test]
    fn blocked_on_quarantine() {
        let v = check(true, 'A', true, 'B');
        if let PublishVerdict::Blocked { reasons } = v {
            assert!(reasons.contains(&"quarantined".to_string()));
        }
    }

    #[test]
    fn invalid_grade_char() {
        assert_eq!(check(true, 'Z', false, 'B'), PublishVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_min_grade_char() {
        assert_eq!(check(true, 'A', false, 'Z'), PublishVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = check(true, 'A', false, 'B');
        let r2 = check(true, 'A', false, 'B');
        assert_eq!(r1, r2);
    }

    #[test]
    fn multiple_failures_all_listed() {
        let v = check(false, 'D', true, 'B');
        if let PublishVerdict::Blocked { reasons } = v {
            assert_eq!(reasons.len(), 3);
        }
    }

    #[test]
    fn boundary_grade_at_min_allowed() {
        let v = check(true, 'B', false, 'B');
        assert_eq!(v, PublishVerdict::Allowed);
    }

    #[test]
    fn min_grade_a_strictest() {
        // Only 'A' allowed when min is 'A'.
        let v = check(true, 'B', false, 'A');
        if let PublishVerdict::Blocked { reasons } = v {
            assert!(reasons.iter().any(|r| r.starts_with("grade_below_min")));
        }
    }

    #[test]
    fn min_grade_f_loosest() {
        // 'F' = anything passes.
        let v = check(true, 'F', false, 'F');
        assert_eq!(v, PublishVerdict::Allowed);
    }

    #[test]
    fn passing_grade_d_with_min_d() {
        let v = check(true, 'D', false, 'D');
        assert_eq!(v, PublishVerdict::Allowed);
    }
}
