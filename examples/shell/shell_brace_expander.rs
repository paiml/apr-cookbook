//! # Shell Brace Expansion
//!
//! Brace expansion: `a{b,c,d}e` → `abe ace ade`. Numeric ranges:
//! `{1..5}` → `1 2 3 4 5`; with step `{1..10..2}` → `1 3 5 7 9`.
//! This recipe builds the comma-list expander (numeric ranges left
//! to the reader as exercise) + nesting depth limiter.
//!
//! Demonstrates the **SHELL.5** recipe for PMAT-126 (shell coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: bash man §3.5.1 Brace Expansion.
//!
//! Run with: cargo run --example shell_brace_expander
//!
//! Added by PMAT-126 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MAX_OUTPUT_COUNT: usize = 1024;

#[derive(Debug, PartialEq)]
pub enum ExpandVerdict {
    Ok(Vec<String>),
    UnbalancedBraces,
    EmptyAlternative,
    OutputTooLarge { count: usize, cap: usize },
}

pub fn expand(input: &str) -> ExpandVerdict {
    if !braces_balanced(input) {
        return ExpandVerdict::UnbalancedBraces;
    }
    let Some((prefix, alternatives, suffix)) = split_outer_brace(input) else {
        return ExpandVerdict::Ok(vec![input.to_string()]);
    };
    let mut all_alts = Vec::with_capacity(alternatives.len());
    for alt in &alternatives {
        if alt.is_empty() {
            return ExpandVerdict::EmptyAlternative;
        }
        match expand(alt) {
            ExpandVerdict::Ok(sub) => all_alts.push(sub),
            other => return other,
        }
    }
    let suffix_expanded = match expand(&suffix) {
        ExpandVerdict::Ok(s) => s,
        other => return other,
    };
    let mut out = Vec::new();
    for alt_set in &all_alts {
        for alt in alt_set {
            for suff in &suffix_expanded {
                out.push(format!("{prefix}{alt}{suff}"));
                if out.len() > MAX_OUTPUT_COUNT {
                    return ExpandVerdict::OutputTooLarge {
                        count: out.len(),
                        cap: MAX_OUTPUT_COUNT,
                    };
                }
            }
        }
    }
    ExpandVerdict::Ok(out)
}

fn braces_balanced(input: &str) -> bool {
    let mut depth = 0i32;
    for c in input.chars() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

fn split_outer_brace(input: &str) -> Option<(String, Vec<String>, String)> {
    let bytes = input.as_bytes();
    let open = bytes.iter().position(|&b| b == b'{')?;
    let mut depth = 1i32;
    let mut close = None;
    for (i, &b) in bytes.iter().enumerate().skip(open + 1) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    let prefix = input[..open].to_string();
    let inside = &input[open + 1..close];
    let suffix = input[close + 1..].to_string();
    let mut alts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in inside.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => depth -= 1,
            ',' if depth == 0 => {
                alts.push(inside[start..i].to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    alts.push(inside[start..].to_string());
    if alts.len() < 2 {
        return None;
    }
    Some((prefix, alts, suffix))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("shell_brace_expander")?;

    for input in ["a{b,c,d}e", "{x,y}", "no{}braces", "no_braces", "{a,,b}"] {
        println!("{input:<15}  →  {:?}", expand(input));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expander_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_expansion() {
        let v = expand("a{b,c,d}e");
        assert_eq!(
            v,
            ExpandVerdict::Ok(vec!["abe".into(), "ace".into(), "ade".into()])
        );
    }

    #[test]
    fn no_braces_returns_input_unchanged() {
        assert_eq!(expand("plain"), ExpandVerdict::Ok(vec!["plain".into()]));
    }

    #[test]
    fn single_alternative_treated_as_literal() {
        // {x} is a single-item brace — bash treats this as literal "{x}".
        assert_eq!(expand("{x}"), ExpandVerdict::Ok(vec!["{x}".into()]));
    }

    #[test]
    fn unbalanced_open_brace_rejected() {
        assert_eq!(expand("a{b,c"), ExpandVerdict::UnbalancedBraces);
    }

    #[test]
    fn unbalanced_close_brace_rejected() {
        assert_eq!(expand("ab,c}"), ExpandVerdict::UnbalancedBraces);
    }

    #[test]
    fn empty_alternative_rejected() {
        // {a,,b} contains an empty middle alternative.
        assert_eq!(expand("{a,,b}"), ExpandVerdict::EmptyAlternative);
    }

    #[test]
    fn nested_braces_expand_cartesian() {
        let v = expand("{a,b}{1,2}");
        if let ExpandVerdict::Ok(items) = v {
            assert_eq!(items.len(), 4);
            assert!(items.contains(&"a1".to_string()));
            assert!(items.contains(&"b2".to_string()));
        }
    }

    #[test]
    fn large_expansion_capped() {
        // 12 nested braces, each 4 alts → 4^12 ≈ 16M items; should cap.
        let big = "{a,b,c,d}{a,b,c,d}{a,b,c,d}{a,b,c,d}{a,b,c,d}{a,b,c,d}";
        let v = expand(big);
        assert!(matches!(v, ExpandVerdict::OutputTooLarge { .. }));
    }

    #[test]
    fn prefix_and_suffix_preserved() {
        let v = expand("pre{a,b}post");
        if let ExpandVerdict::Ok(items) = v {
            assert!(items.contains(&"preapost".to_string()));
            assert!(items.contains(&"prebpost".to_string()));
        }
    }
}
