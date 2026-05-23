//! # Contracts-Macros YAML Indent Normalize
//!
//! Detect inconsistent YAML indent widths (mix of 2-space and
//! 4-space). Reports detected widths and `consistent` flag.
//!
//! Demonstrates the **CMM.71** recipe for PMAT-181 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 spec §6.1 (indent); EditorConfig.org spec.
//!
//! Run with: cargo run --example contracts_macros_yaml_indent_normalize
//!
//! Added by PMAT-181 (catalog 1252→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum IndentVerdict {
    Ok {
        widths_seen: Vec<u32>,
        consistent: bool,
    },
    InvalidConfig,
}

pub fn audit(lines: &[&str]) -> IndentVerdict {
    if lines.is_empty() {
        return IndentVerdict::InvalidConfig;
    }
    let mut widths: BTreeSet<u32> = BTreeSet::new();
    for line in lines {
        let mut leading: u32 = 0;
        for c in line.chars() {
            if c == ' ' {
                leading += 1;
            } else {
                break;
            }
        }
        if leading > 0 && line.trim_start() != *line {
            // Skip lines that are pure-blank or only whitespace.
            if !line.trim().is_empty() {
                widths.insert(leading);
            }
        }
    }
    let widths_seen: Vec<u32> = widths.into_iter().collect();
    let consistent = widths_seen.len() <= 1
        || widths_seen
            .iter()
            .all(|w| widths_seen.first().is_some_and(|first| w % first == 0));
    IndentVerdict::Ok {
        widths_seen,
        consistent,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_indent_normalize")?;

    let consistent = ["root:", "  key1: a", "  key2: b", "    nested: c"];
    println!("consistent: {:?}", audit(&consistent));
    let mixed = [
        "root:",
        "  two_space:",
        "    four_space: x",
        "   three_space: y",
    ];
    println!("mixed: {:?}", audit(&mixed));
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
    fn pure_two_space_consistent() {
        let lines = ["root:", "  a: 1", "    nested: 2"];
        let v = audit(&lines);
        if let IndentVerdict::Ok { consistent, .. } = v {
            assert!(consistent);
        }
    }

    #[test]
    fn three_space_outlier_inconsistent() {
        let lines = ["root:", "  a: 1", "   b: 2"];
        let v = audit(&lines);
        if let IndentVerdict::Ok { consistent, .. } = v {
            assert!(!consistent);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(audit(&[]), IndentVerdict::InvalidConfig);
    }

    #[test]
    fn no_indented_lines_consistent() {
        let lines = ["root:", "key1: a", "key2: b"];
        let v = audit(&lines);
        if let IndentVerdict::Ok {
            consistent,
            widths_seen,
        } = v
        {
            assert!(consistent);
            assert!(widths_seen.is_empty());
        }
    }

    #[test]
    fn widths_sorted() {
        let lines = ["a", "    deep:", "  shallow:"];
        let v = audit(&lines);
        if let IndentVerdict::Ok { widths_seen, .. } = v {
            assert_eq!(widths_seen, vec![2, 4]);
        }
    }

    #[test]
    fn deterministic() {
        let lines = ["a", "  b"];
        let r1 = audit(&lines);
        let r2 = audit(&lines);
        assert_eq!(r1, r2);
    }

    #[test]
    fn blank_lines_ignored() {
        let lines = ["root:", "  a: 1", "", "   ", "  b: 2"];
        let v = audit(&lines);
        if let IndentVerdict::Ok { widths_seen, .. } = v {
            assert_eq!(widths_seen, vec![2]);
        }
    }

    #[test]
    fn duplicate_widths_collapse() {
        let lines = ["a", "  b", "  c", "  d"];
        let v = audit(&lines);
        if let IndentVerdict::Ok { widths_seen, .. } = v {
            assert_eq!(widths_seen.len(), 1);
        }
    }

    #[test]
    fn deep_indent_multiple_of_first() {
        let lines = ["a", "  b", "    c", "      d"];
        let v = audit(&lines);
        if let IndentVerdict::Ok { consistent, .. } = v {
            assert!(consistent);
        }
    }

    #[test]
    fn one_indent_only_is_consistent() {
        let lines = ["a", "  b"];
        let v = audit(&lines);
        if let IndentVerdict::Ok { consistent, .. } = v {
            assert!(consistent);
        }
    }

    #[test]
    fn tabs_treated_as_no_indent() {
        // Tab is not a space; we only count leading spaces.
        let lines = ["a", "\tb"];
        let v = audit(&lines);
        if let IndentVerdict::Ok { widths_seen, .. } = v {
            assert!(widths_seen.is_empty());
        }
    }
}
