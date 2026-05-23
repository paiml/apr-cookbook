//! # apr hex — `--slice <START:END>` Range Parser
//!
//! `apr hex <FILE> --slice 0:3` selects a partial tensor read by giving
//! a Python-style `[start:end)` half-open interval. This recipe builds
//! the parser and asserts the contract: end > start, both must be
//! non-negative, single-`:` syntax only, malformed input returns an
//! error rather than silently using the full range.
//!
//! Demonstrates the **HEX.5** recipe for PMAT-100 (apr hex coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender HEX-002 + Python slice convention
//!
//! Run with: cargo run --example cli_hex_slice_range_parser
//!
//! Added by PMAT-100 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceRange {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, PartialEq)]
pub enum SliceVerdict {
    Ok(SliceRange),
    MalformedSyntax,
    EndNotAfterStart,
    NegativeIndex,
}

pub fn parse_slice(s: &str) -> SliceVerdict {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return SliceVerdict::MalformedSyntax;
    }
    if parts[0].trim().starts_with('-') || parts[1].trim().starts_with('-') {
        return SliceVerdict::NegativeIndex;
    }
    let Ok(start) = parts[0].trim().parse::<u64>() else {
        return SliceVerdict::MalformedSyntax;
    };
    let Ok(end) = parts[1].trim().parse::<u64>() else {
        return SliceVerdict::MalformedSyntax;
    };
    if end <= start {
        return SliceVerdict::EndNotAfterStart;
    }
    SliceVerdict::Ok(SliceRange { start, end })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_hex_slice_range_parser")?;

    let cases = [
        "0:3", "100:200", "0:0", "10:5", "1:3:5", "abc:def", "-1:5", "1: 5",
    ];
    for c in cases {
        println!("{c:>10}  →  {:?}", parse_slice(c));
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
    fn happy_slice_parses() {
        let v = parse_slice("0:3");
        assert_eq!(v, SliceVerdict::Ok(SliceRange { start: 0, end: 3 }));
    }

    #[test]
    fn end_equal_to_start_rejected() {
        // [0:0) is empty by Python convention — caller probably meant something else.
        assert_eq!(parse_slice("0:0"), SliceVerdict::EndNotAfterStart);
    }

    #[test]
    fn end_less_than_start_rejected() {
        assert_eq!(parse_slice("10:5"), SliceVerdict::EndNotAfterStart);
    }

    #[test]
    fn malformed_three_part_slice_rejected() {
        // Python supports "0:10:2" (step) but apr hex does not.
        assert_eq!(parse_slice("1:3:5"), SliceVerdict::MalformedSyntax);
    }

    #[test]
    fn missing_colon_rejected() {
        assert_eq!(parse_slice("0"), SliceVerdict::MalformedSyntax);
    }

    #[test]
    fn negative_index_rejected() {
        // No Python-style negative indexing.
        assert_eq!(parse_slice("-1:5"), SliceVerdict::NegativeIndex);
        assert_eq!(parse_slice("0:-1"), SliceVerdict::NegativeIndex);
    }

    #[test]
    fn whitespace_around_numbers_trimmed() {
        let v = parse_slice("  100 : 200 ");
        assert_eq!(
            v,
            SliceVerdict::Ok(SliceRange {
                start: 100,
                end: 200
            })
        );
    }

    #[test]
    fn nonnumeric_returns_malformed() {
        assert_eq!(parse_slice("abc:def"), SliceVerdict::MalformedSyntax);
        assert_eq!(parse_slice(":5"), SliceVerdict::MalformedSyntax);
        assert_eq!(parse_slice("5:"), SliceVerdict::MalformedSyntax);
    }
}
