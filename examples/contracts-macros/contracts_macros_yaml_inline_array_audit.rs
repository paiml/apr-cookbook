//! # Contracts-Macros YAML Inline Array Audit
//!
//! Flag YAML keys using inline `[a, b, c]` flow form when block form
//! `- a\n- b` would be more readable for arrays exceeding `max_inline`.
//!
//! Demonstrates the **CMM.113** recipe for PMAT-195 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 spec §7.4 (flow vs block style); editor
//!  readability conventions.
//!
//! Run with: cargo run --example contracts_macros_yaml_inline_array_audit
//!
//! Added by PMAT-195 (catalog 1378→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone)]
pub enum ArrayStyle {
    Block,
    Inline,
}

#[derive(Debug, PartialEq)]
pub enum ArrayVerdict {
    Ok {
        offending: Vec<String>,
        block_count: u32,
        inline_count: u32,
    },
    InvalidConfig,
}

pub fn audit(arrays: &[(&str, ArrayStyle, u32)], max_inline: u32) -> ArrayVerdict {
    if arrays.is_empty() || max_inline == 0 {
        return ArrayVerdict::InvalidConfig;
    }
    let mut offending: Vec<String> = Vec::new();
    let mut block = 0u32;
    let mut inline = 0u32;
    for (name, style, len) in arrays {
        match style {
            ArrayStyle::Block => block += 1,
            ArrayStyle::Inline => {
                inline += 1;
                if *len > max_inline {
                    offending.push((*name).to_string());
                }
            }
        }
    }
    offending.sort();
    ArrayVerdict::Ok {
        offending,
        block_count: block,
        inline_count: inline,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_inline_array_audit")?;

    let arrays = [
        ("ok_inline", ArrayStyle::Inline, 3),
        ("too_many", ArrayStyle::Inline, 10),
        ("good_block", ArrayStyle::Block, 20),
    ];
    println!("audit max=5: {:?}", audit(&arrays, 5));
    println!("invalid: {:?}", audit(&[], 5));
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
    fn small_inline_no_flag() {
        let arrays = [("a", ArrayStyle::Inline, 3)];
        let v = audit(&arrays, 5);
        if let ArrayVerdict::Ok { offending, .. } = v {
            assert!(offending.is_empty());
        }
    }

    #[test]
    fn large_inline_flagged() {
        let arrays = [("a", ArrayStyle::Inline, 10)];
        let v = audit(&arrays, 5);
        if let ArrayVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["a".to_string()]);
        }
    }

    #[test]
    fn block_never_flagged() {
        let arrays = [("a", ArrayStyle::Block, 100)];
        let v = audit(&arrays, 5);
        if let ArrayVerdict::Ok { offending, .. } = v {
            assert!(offending.is_empty());
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[], 5), ArrayVerdict::InvalidConfig);
    }

    #[test]
    fn zero_max_rejected() {
        let arrays = [("a", ArrayStyle::Inline, 3)];
        assert_eq!(audit(&arrays, 0), ArrayVerdict::InvalidConfig);
    }

    #[test]
    fn count_block_correctly() {
        let arrays = [("a", ArrayStyle::Block, 5), ("b", ArrayStyle::Block, 3)];
        let v = audit(&arrays, 5);
        if let ArrayVerdict::Ok { block_count, .. } = v {
            assert_eq!(block_count, 2);
        }
    }

    #[test]
    fn count_inline_correctly() {
        let arrays = [("a", ArrayStyle::Inline, 5), ("b", ArrayStyle::Inline, 3)];
        let v = audit(&arrays, 5);
        if let ArrayVerdict::Ok { inline_count, .. } = v {
            assert_eq!(inline_count, 2);
        }
    }

    #[test]
    fn boundary_at_max_no_flag() {
        let arrays = [("a", ArrayStyle::Inline, 5)];
        let v = audit(&arrays, 5);
        if let ArrayVerdict::Ok { offending, .. } = v {
            assert!(offending.is_empty());
        }
    }

    #[test]
    fn one_over_max_flagged() {
        let arrays = [("a", ArrayStyle::Inline, 6)];
        let v = audit(&arrays, 5);
        if let ArrayVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["a".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let arrays = [("a", ArrayStyle::Inline, 3)];
        let r1 = audit(&arrays, 5);
        let r2 = audit(&arrays, 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offending_sorted() {
        let arrays = [
            ("zeta", ArrayStyle::Inline, 100),
            ("alpha", ArrayStyle::Inline, 100),
        ];
        let v = audit(&arrays, 5);
        if let ArrayVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }
}
