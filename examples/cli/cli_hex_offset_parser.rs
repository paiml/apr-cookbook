//! # apr hex — `--offset` Parser (Decimal + Hex)
//!
//! `apr hex <FILE> --offset <N>` accepts decimal (e.g., `1024`) or
//! hex-prefixed (e.g., `0x400`) byte offsets. This recipe builds the
//! parser and asserts the contract: leading `0x` triggers hex parsing,
//! plain digits parse as decimal, malformed input returns None (not 0),
//! and offsets must fit in u64.
//!
//! Demonstrates the **HEX.4** recipe for PMAT-100 (apr hex coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender HEX-001 + xxd(1) offset convention
//!
//! Run with: cargo run --example cli_hex_offset_parser
//!
//! Added by PMAT-100 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

pub fn parse_offset(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        return u64::from_str_radix(hex, 16).ok();
    }
    s.parse::<u64>().ok()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_hex_offset_parser")?;

    let cases = [
        "0",
        "1024",
        "0x400",
        "0xDEADBEEF",
        "0X1A",
        "garbage",
        "",
        "0x",
    ];
    for c in cases {
        println!("{c:>15}  →  {:?}", parse_offset(c));
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
    fn decimal_parses() {
        assert_eq!(parse_offset("0"), Some(0));
        assert_eq!(parse_offset("1024"), Some(1024));
        assert_eq!(parse_offset("18446744073709551615"), Some(u64::MAX));
    }

    #[test]
    fn hex_with_0x_prefix_parses() {
        assert_eq!(parse_offset("0x400"), Some(1024));
        assert_eq!(parse_offset("0xDEADBEEF"), Some(0xDEAD_BEEF));
    }

    #[test]
    fn hex_uppercase_x_also_parses() {
        // CLI flexibility: `0X1A` should work the same as `0x1a`.
        assert_eq!(parse_offset("0X1A"), Some(26));
    }

    #[test]
    fn empty_string_returns_none() {
        assert!(parse_offset("").is_none());
    }

    #[test]
    fn whitespace_only_returns_none() {
        assert!(parse_offset("   ").is_none());
    }

    #[test]
    fn garbage_returns_none_not_zero() {
        // Critical: don't silently parse "garbage" as 0 — operator's typo
        // would otherwise dump from byte 0 instead of the intended offset.
        assert!(parse_offset("garbage").is_none());
        assert!(parse_offset("0xZZZ").is_none());
        assert!(parse_offset("123abc").is_none());
    }

    #[test]
    fn naked_0x_returns_none() {
        // "0x" alone has no hex digits — must not silently default to 0.
        assert!(parse_offset("0x").is_none());
    }

    #[test]
    fn overflow_returns_none() {
        // Past u64::MAX must not panic or silently wrap.
        assert!(parse_offset("99999999999999999999").is_none());
    }

    #[test]
    fn leading_whitespace_trimmed() {
        assert_eq!(parse_offset("  1024  "), Some(1024));
    }
}
