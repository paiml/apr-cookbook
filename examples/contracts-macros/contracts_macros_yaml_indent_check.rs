//! # Contracts-Macros YAML Indent Consistency Check
//!
//! Verify all indented YAML lines use consistent indentation (2-space
//! or 4-space, never mixed). Returns the dominant style and any
//! offending line numbers.
//!
//! Demonstrates the **CMM.60** recipe for PMAT-177 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: yamllint indentation rules.
//!
//! Run with: cargo run --example contracts_macros_yaml_indent_check
//!
//! Added by PMAT-177 (catalog 1216→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndentStyle {
    TwoSpaces,
    FourSpaces,
}

#[derive(Debug, PartialEq)]
pub enum IndentVerdict {
    Consistent {
        style: IndentStyle,
    },
    Inconsistent {
        dominant_style: IndentStyle,
        offending_lines: Vec<u32>,
    },
    EmptyDocument,
    InvalidIndent {
        line: u32,
    },
}

pub fn check(yaml: &str) -> IndentVerdict {
    if yaml.trim().is_empty() {
        return IndentVerdict::EmptyDocument;
    }
    let mut indents: Vec<(u32, u32)> = Vec::new();
    for (i, line) in yaml.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let leading_spaces = line.chars().take_while(|c| *c == ' ').count() as u32;
        if leading_spaces == 0 {
            continue;
        }
        indents.push(((i + 1) as u32, leading_spaces));
    }
    if indents.is_empty() {
        return IndentVerdict::Consistent {
            style: IndentStyle::TwoSpaces,
        };
    }
    // Determine GCD of all indents — that's the unit.
    let gcd = indents.iter().map(|(_, n)| *n).fold(0u32, gcd_u32);
    if gcd != 2 && gcd != 4 {
        // Reject odd or fractional indents.
        let bad_line = indents
            .iter()
            .find(|(_, n)| n % 2 != 0)
            .map_or(indents[0].0, |(line, _)| *line);
        return IndentVerdict::InvalidIndent { line: bad_line };
    }
    let style = if gcd == 4 {
        IndentStyle::FourSpaces
    } else {
        IndentStyle::TwoSpaces
    };
    let unit = if style == IndentStyle::FourSpaces {
        4
    } else {
        2
    };
    let offending_lines: Vec<u32> = indents
        .iter()
        .filter(|(_, n)| n % unit != 0)
        .map(|(line, _)| *line)
        .collect();
    if offending_lines.is_empty() {
        IndentVerdict::Consistent { style }
    } else {
        IndentVerdict::Inconsistent {
            dominant_style: style,
            offending_lines,
        }
    }
}

fn gcd_u32(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd_u32(b, a % b)
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_indent_check")?;

    let two_space = "key:\n  nested: value";
    println!("2-space: {:?}", check(two_space));

    let four_space = "key:\n    nested: value";
    println!("4-space: {:?}", check(four_space));

    let mixed = "a:\n  b: 1\n    c: 2";
    println!("mixed: {:?}", check(mixed));

    let invalid = "key:\n   bad: indent";
    println!("invalid: {:?}", check(invalid));

    println!("empty: {:?}", check("   "));
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
    fn two_space_consistent() {
        let v = check("a:\n  b: 1\n  c: 2");
        if let IndentVerdict::Consistent { style } = v {
            assert_eq!(style, IndentStyle::TwoSpaces);
        }
    }

    #[test]
    fn four_space_consistent() {
        let v = check("a:\n    b: 1");
        if let IndentVerdict::Consistent { style } = v {
            assert_eq!(style, IndentStyle::FourSpaces);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(check("   "), IndentVerdict::EmptyDocument);
    }

    #[test]
    fn three_space_invalid() {
        let v = check("a:\n   b: 1");
        assert!(matches!(v, IndentVerdict::InvalidIndent { .. }));
    }

    #[test]
    fn no_indent_consistent_two() {
        let v = check("flat: value");
        if let IndentVerdict::Consistent { style } = v {
            assert_eq!(style, IndentStyle::TwoSpaces);
        }
    }

    #[test]
    fn comments_skipped() {
        // Empty/comment lines don't affect.
        let v = check("a:\n\n  b: 1");
        assert!(matches!(v, IndentVerdict::Consistent { .. }));
    }

    #[test]
    fn mixed_two_and_four_consistent_under_2() {
        // 2 and 4 share GCD=2 → both valid 2-space indents at depths 1 and 2.
        let v = check("a:\n  b: 1\nc:\n    d: 2");
        assert!(matches!(v, IndentVerdict::Consistent { .. }));
    }

    #[test]
    fn six_alone_invalid_under_4() {
        // Indent 6 is not a multiple of 4; GCD(6) = 6 → InvalidIndent.
        let v = check("a:\n      b: 1");
        assert!(matches!(v, IndentVerdict::InvalidIndent { .. }));
    }

    #[test]
    fn deep_nest_two_space_ok() {
        let v = check("a:\n  b:\n    c:\n      d: 1");
        assert!(matches!(v, IndentVerdict::Consistent { .. }));
    }

    #[test]
    fn one_space_invalid() {
        let v = check("a:\n b: 1");
        assert!(matches!(v, IndentVerdict::InvalidIndent { .. }));
    }

    #[test]
    fn deterministic() {
        let yaml = "a:\n  b: 1";
        let a = check(yaml);
        let b = check(yaml);
        assert_eq!(a, b);
    }
}
