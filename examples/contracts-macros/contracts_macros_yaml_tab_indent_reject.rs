//! # Contracts-Macros YAML Tab-Indent Reject
//!
//! YAML 1.2 forbids tabs for indentation. Detect any indented line
//! using tabs. Returns sorted offending line numbers.
//!
//! Demonstrates the **CMM.178** recipe for PMAT-217 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 §6.1 indentation spaces; libyaml scanner
//!  reject-tab error.
//!
//! Run with: cargo run --example contracts_macros_yaml_tab_indent_reject
//!
//! Added by PMAT-217 (catalog 1576→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TabIndentVerdict {
    Ok {
        offending_lines: Vec<u32>,
        clean: bool,
    },
    InvalidConfig,
}

pub fn check(buffer: &str) -> TabIndentVerdict {
    if buffer.is_empty() {
        return TabIndentVerdict::InvalidConfig;
    }
    let mut offenders: Vec<u32> = Vec::new();
    for (i, line) in buffer.split('\n').enumerate() {
        if line.is_empty() {
            continue;
        }
        // Check leading whitespace for tabs.
        for (j, c) in line.char_indices() {
            if c != ' ' && c != '\t' {
                break;
            }
            if c == '\t' {
                offenders.push((i + 1) as u32);
                let _ = j;
                break;
            }
        }
    }
    let clean = offenders.is_empty();
    TabIndentVerdict::Ok {
        offending_lines: offenders,
        clean,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_tab_indent_reject")?;

    println!("clean: {:?}", check("a:\n  b: 1\n"));
    println!("tabs: {:?}", check("a:\n\tb: 1\n"));
    println!("invalid: {:?}", check(""));
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
    fn empty_input_rejected() {
        assert_eq!(check(""), TabIndentVerdict::InvalidConfig);
    }

    #[test]
    fn space_indent_clean() {
        let v = check("a:\n  b: 1\n");
        if let TabIndentVerdict::Ok { clean, .. } = v {
            assert!(clean);
        }
    }

    #[test]
    fn tab_indent_flagged() {
        let v = check("a:\n\tb: 1\n");
        if let TabIndentVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert_eq!(offending_lines, vec![2]);
        }
    }

    #[test]
    fn mixed_tab_in_indent_flagged() {
        let v = check("a:\n \tb: 1\n");
        if let TabIndentVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert_eq!(offending_lines, vec![2]);
        }
    }

    #[test]
    fn tab_in_value_not_flagged() {
        let v = check("a: x\ty\n");
        if let TabIndentVerdict::Ok { clean, .. } = v {
            assert!(clean);
        }
    }

    #[test]
    fn multiple_tab_lines_flagged() {
        let v = check("a:\n\tb: 1\nc:\n\td: 2\n");
        if let TabIndentVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert_eq!(offending_lines, vec![2, 4]);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check("a:\n\tb\n");
        let r2 = check("a:\n\tb\n");
        assert_eq!(r1, r2);
    }

    #[test]
    fn empty_lines_ignored() {
        let v = check("a:\n\n  b\n");
        if let TabIndentVerdict::Ok { clean, .. } = v {
            assert!(clean);
        }
    }

    #[test]
    fn unicode_value_handled() {
        let v = check("name: café\n");
        if let TabIndentVerdict::Ok { clean, .. } = v {
            assert!(clean);
        }
    }

    #[test]
    fn many_lines_handled() {
        let mut buf = String::new();
        for _ in 0..30 {
            buf.push_str("\tbad\n");
        }
        let v = check(&buf);
        if let TabIndentVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert_eq!(offending_lines.len(), 30);
        }
    }

    #[test]
    fn first_line_tab_flagged() {
        let v = check("\ta: 1\n");
        if let TabIndentVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert_eq!(offending_lines, vec![1]);
        }
    }

    #[test]
    fn tab_in_string_not_in_indent_clean() {
        // Leading content is not whitespace → no tab in indent.
        let v = check("a:\\tval\n");
        if let TabIndentVerdict::Ok { clean, .. } = v {
            assert!(clean);
        }
    }
}
