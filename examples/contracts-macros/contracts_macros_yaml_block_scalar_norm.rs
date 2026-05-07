//! # Contracts-Macros YAML Block Scalar Normalize
//!
//! Detect inconsistent block-scalar styles (`|` literal vs `>`
//! folded). Returns sorted offending keys whose style differs from
//! the policy.
//!
//! Demonstrates the **CMM.184** recipe for PMAT-219 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 §8 block-scalar styles; libyaml block-scalar
//!  style enums.
//!
//! Run with: cargo run --example contracts_macros_yaml_block_scalar_norm
//!
//! Added by PMAT-219 (catalog 1594→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BlockScalarVerdict {
    Ok {
        offending_keys: Vec<String>,
        clean: bool,
    },
    InvalidConfig,
}

/// Items: (key, style_char). Allowed styles: `|` (literal), `>` (folded).
pub fn check(items: &[(&str, char)], policy: char) -> BlockScalarVerdict {
    if items.is_empty() || (policy != '|' && policy != '>') {
        return BlockScalarVerdict::InvalidConfig;
    }
    for (_, style) in items {
        if *style != '|' && *style != '>' {
            return BlockScalarVerdict::InvalidConfig;
        }
    }
    let mut offenders: Vec<String> = items
        .iter()
        .filter(|(_, s)| *s != policy)
        .map(|(k, _)| (*k).to_string())
        .collect();
    offenders.sort();
    let clean = offenders.is_empty();
    BlockScalarVerdict::Ok {
        offending_keys: offenders,
        clean,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_block_scalar_norm")?;

    let items = [("desc", '|'), ("body", '>'), ("notes", '|')];
    println!("policy=|: {:?}", check(&items, '|'));
    println!("invalid: {:?}", check(&[], '|'));
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
    fn matching_policy_clean() {
        let v = check(&[("k", '|')], '|');
        if let BlockScalarVerdict::Ok { clean, .. } = v {
            assert!(clean);
        }
    }

    #[test]
    fn mismatch_flagged() {
        let v = check(&[("k", '>')], '|');
        if let BlockScalarVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["k".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[], '|'), BlockScalarVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_policy_char() {
        assert_eq!(check(&[("k", '|')], 'x'), BlockScalarVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_style_char() {
        assert_eq!(check(&[("k", 'x')], '|'), BlockScalarVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = check(&[("k", '|')], '|');
        let r2 = check(&[("k", '|')], '|');
        assert_eq!(r1, r2);
    }

    #[test]
    fn offenders_sorted() {
        let v = check(&[("zeta", '>'), ("alpha", '>')], '|');
        if let BlockScalarVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(
                offending_keys,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn folded_policy_supported() {
        let v = check(&[("k", '>')], '>');
        if let BlockScalarVerdict::Ok { clean, .. } = v {
            assert!(clean);
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, char)> = (0..30).map(|_| ("k", '>')).collect();
        let v = check(&items, '|');
        if let BlockScalarVerdict::Ok { offending_keys, .. } = v {
            // No dedup; each violator entry is preserved.
            assert_eq!(offending_keys.len(), 30);
        }
    }

    #[test]
    fn unicode_key_supported() {
        let v = check(&[("café", '>')], '|');
        if let BlockScalarVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["café".to_string()]);
        }
    }

    #[test]
    fn mixed_styles_flag_only_violators() {
        let v = check(&[("a", '|'), ("b", '>'), ("c", '|')], '|');
        if let BlockScalarVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["b".to_string()]);
        }
    }
}
