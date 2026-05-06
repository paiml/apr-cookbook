//! # Contracts-Macros YAML Anchor Naming
//!
//! Verify YAML anchors follow `&snake_case` convention: lowercase
//! ASCII letters, digits, and underscores only. Returns offending
//! anchor names.
//!
//! Demonstrates the **CMM.104** recipe for PMAT-192 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 spec §6.9.1 (anchor names — char restrictions);
//!  PEP 8 snake_case for Python identifiers.
//!
//! Run with: cargo run --example contracts_macros_yaml_anchor_naming
//!
//! Added by PMAT-192 (catalog 1351→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AnchorVerdict {
    Ok {
        offending: Vec<String>,
        ok_count: u32,
    },
    InvalidConfig,
}

pub fn audit(anchors: &[&str]) -> AnchorVerdict {
    if anchors.is_empty() {
        return AnchorVerdict::InvalidConfig;
    }
    let mut offending: Vec<String> = Vec::new();
    let mut ok_count = 0u32;
    for anchor in anchors {
        if is_snake_case(anchor) {
            ok_count += 1;
        } else {
            offending.push((*anchor).to_string());
        }
    }
    offending.sort();
    offending.dedup();
    AnchorVerdict::Ok {
        offending,
        ok_count,
    }
}

fn is_snake_case(s: &str) -> bool {
    if s.is_empty() || s.starts_with('_') || s.ends_with('_') || s.contains("__") {
        return false;
    }
    let first = s.chars().next().unwrap();
    if !first.is_ascii_lowercase() {
        return false;
    }
    s.chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_anchor_naming")?;

    let anchors = [
        "good_anchor",
        "BadAnchor",
        "kebab-case",
        "_leading",
        "ok2_v1",
    ];
    println!("audit: {:?}", audit(&anchors));
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
    fn snake_case_passes() {
        let v = audit(&["good_name", "another_good"]);
        if let AnchorVerdict::Ok { offending, .. } = v {
            assert!(offending.is_empty());
        }
    }

    #[test]
    fn pascal_case_flagged() {
        let v = audit(&["PascalCase"]);
        if let AnchorVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["PascalCase".to_string()]);
        }
    }

    #[test]
    fn kebab_case_flagged() {
        let v = audit(&["kebab-case"]);
        if let AnchorVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["kebab-case".to_string()]);
        }
    }

    #[test]
    fn leading_underscore_flagged() {
        let v = audit(&["_leading"]);
        if let AnchorVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["_leading".to_string()]);
        }
    }

    #[test]
    fn trailing_underscore_flagged() {
        let v = audit(&["trailing_"]);
        if let AnchorVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["trailing_".to_string()]);
        }
    }

    #[test]
    fn double_underscore_flagged() {
        let v = audit(&["double__under"]);
        if let AnchorVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["double__under".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), AnchorVerdict::InvalidConfig);
    }

    #[test]
    fn digits_allowed() {
        let v = audit(&["v1_alpha", "step_2"]);
        if let AnchorVerdict::Ok { offending, .. } = v {
            assert!(offending.is_empty());
        }
    }

    #[test]
    fn ok_count_correct() {
        let v = audit(&["good_name", "BadName"]);
        if let AnchorVerdict::Ok { ok_count, .. } = v {
            assert_eq!(ok_count, 1);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&["good"]);
        let r2 = audit(&["good"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offending_sorted() {
        let v = audit(&["zeta_BAD", "alpha_BAD"]);
        if let AnchorVerdict::Ok { offending, .. } = v {
            assert_eq!(
                offending,
                vec!["alpha_BAD".to_string(), "zeta_BAD".to_string()]
            );
        }
    }

    #[test]
    fn digits_first_flagged() {
        let v = audit(&["1leading"]);
        if let AnchorVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["1leading".to_string()]);
        }
    }
}
