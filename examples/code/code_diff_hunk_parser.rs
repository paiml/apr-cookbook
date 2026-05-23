//! # Code Unified-Diff Hunk Parser
//!
//! Unified diff hunks open with `@@ -a,b +c,d @@`: a/b = old start +
//! length, c/d = new start + length. Length defaults to 1 when
//! omitted (`@@ -5 +5 @@`). This recipe builds the parser.
//!
//! Demonstrates the **CODE.8** recipe for PMAT-125 (code coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GNU diffutils §unified format.
//!
//! Run with: cargo run --example code_diff_hunk_parser
//!
//! Added by PMAT-125 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq)]
pub struct HunkHeader {
    pub old_start: u32,
    pub old_len: u32,
    pub new_start: u32,
    pub new_len: u32,
}

#[derive(Debug, PartialEq)]
pub enum ParseError {
    NotAHunkHeader,
    MissingMinus,
    MissingPlus,
    InvalidNumber,
    MissingTrailingMarker,
}

pub fn parse(line: &str) -> std::result::Result<HunkHeader, ParseError> {
    let trimmed = line.strip_prefix("@@ ").ok_or(ParseError::NotAHunkHeader)?;
    let inner = trimmed.strip_suffix(" @@").or_else(|| {
        // Allow trailing context after @@: "... @@ section_label"
        trimmed.find(" @@").map(|i| &trimmed[..i])
    });
    let inner = inner.ok_or(ParseError::MissingTrailingMarker)?;
    let mut parts = inner.split_whitespace();
    let minus = parts.next().ok_or(ParseError::MissingMinus)?;
    let plus = parts.next().ok_or(ParseError::MissingPlus)?;
    let (old_start, old_len) =
        parse_range(minus.strip_prefix('-').ok_or(ParseError::MissingMinus)?)?;
    let (new_start, new_len) = parse_range(plus.strip_prefix('+').ok_or(ParseError::MissingPlus)?)?;
    Ok(HunkHeader {
        old_start,
        old_len,
        new_start,
        new_len,
    })
}

fn parse_range(s: &str) -> std::result::Result<(u32, u32), ParseError> {
    let (start_str, len_str) = match s.split_once(',') {
        Some((a, b)) => (a, b),
        None => (s, "1"),
    };
    let start = start_str
        .parse::<u32>()
        .map_err(|_| ParseError::InvalidNumber)?;
    let len = len_str
        .parse::<u32>()
        .map_err(|_| ParseError::InvalidNumber)?;
    Ok((start, len))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("code_diff_hunk_parser")?;

    for line in [
        "@@ -1,7 +1,9 @@",
        "@@ -5 +5 @@",
        "@@ -10,3 +12,3 @@ fn foo() {",
        "@@ malformed @@",
        "not a hunk",
    ] {
        println!("{line:<40}  →  {:?}", parse(line));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_hunk_header_parsed() {
        let h = parse("@@ -1,7 +1,9 @@").unwrap();
        assert_eq!(
            h,
            HunkHeader {
                old_start: 1,
                old_len: 7,
                new_start: 1,
                new_len: 9
            }
        );
    }

    #[test]
    fn omitted_length_defaults_to_one() {
        let h = parse("@@ -5 +5 @@").unwrap();
        assert_eq!(h.old_len, 1);
        assert_eq!(h.new_len, 1);
    }

    #[test]
    fn trailing_section_label_accepted() {
        let h = parse("@@ -10,3 +12,3 @@ fn foo() {").unwrap();
        assert_eq!(h.old_start, 10);
        assert_eq!(h.new_start, 12);
    }

    #[test]
    fn non_hunk_line_rejected() {
        assert_eq!(parse("not a hunk"), Err(ParseError::NotAHunkHeader));
    }

    #[test]
    fn missing_minus_rejected() {
        // First component must be `-N[,M]`.
        let v = parse("@@ +1,7 +1,9 @@");
        assert!(matches!(v, Err(ParseError::MissingMinus)));
    }

    #[test]
    fn missing_plus_rejected() {
        let v = parse("@@ -1,7 -1,9 @@");
        assert!(matches!(v, Err(ParseError::MissingPlus)));
    }

    #[test]
    fn non_numeric_rejected() {
        let v = parse("@@ -a,b +1,9 @@");
        assert!(matches!(v, Err(ParseError::InvalidNumber)));
    }

    #[test]
    fn missing_trailing_marker_rejected() {
        let v = parse("@@ -1,7 +1,9");
        assert!(matches!(v, Err(ParseError::MissingTrailingMarker)));
    }

    #[test]
    fn boundary_zero_length_hunk() {
        // @@ -0,0 +1,5 @@ = file creation hunk.
        let h = parse("@@ -0,0 +1,5 @@").unwrap();
        assert_eq!(h.old_len, 0);
        assert_eq!(h.new_len, 5);
    }
}
