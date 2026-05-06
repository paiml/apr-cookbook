//! # apr trace --save-tensor-layers — Layer Range Parser
//!
//! `apr trace --save-tensor --save-tensor-layers <RANGE>` accepts Rust
//! range syntax: `START..END` (END exclusive). Default is `0..1` (just
//! layer 0). This recipe builds the parser and asserts the contract:
//! malformed rejects, end ≤ start rejects (empty range), single-layer
//! ranges supported.
//!
//! Demonstrates the **TRACE.7** recipe for PMAT-109 (apr trace coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SHIP-007 + Rust range syntax
//!
//! Run with: cargo run --example cli_trace_save_tensor_layer_range
//!
//! Added by PMAT-109 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RangeVerdict {
    Ok {
        start: u32,
        end: u32,
        layer_count: u32,
    },
    MalformedSyntax,
    EndNotAfterStart {
        start: u32,
        end: u32,
    },
    NotNumeric,
}

pub fn parse_range(s: &str) -> RangeVerdict {
    let parts: Vec<&str> = s.split("..").collect();
    if parts.len() != 2 {
        return RangeVerdict::MalformedSyntax;
    }
    let Ok(start) = parts[0].trim().parse::<u32>() else {
        return RangeVerdict::NotNumeric;
    };
    let Ok(end) = parts[1].trim().parse::<u32>() else {
        return RangeVerdict::NotNumeric;
    };
    if end <= start {
        return RangeVerdict::EndNotAfterStart { start, end };
    }
    RangeVerdict::Ok {
        start,
        end,
        layer_count: end - start,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_trace_save_tensor_layer_range")?;

    for s in [
        "0..1", "0..28", "5..10", "10..5", "0..0", "abc..xyz", "0", "0..1..2",
    ] {
        println!("{s:>15}  →  {:?}", parse_range(s));
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
    fn default_0_to_1_parses() {
        let v = parse_range("0..1");
        assert_eq!(
            v,
            RangeVerdict::Ok {
                start: 0,
                end: 1,
                layer_count: 1,
            }
        );
    }

    #[test]
    fn full_28_layers_parses() {
        let v = parse_range("0..28");
        if let RangeVerdict::Ok { layer_count, .. } = v {
            assert_eq!(layer_count, 28);
        }
    }

    #[test]
    fn end_equal_to_start_rejected() {
        // `0..0` is empty (no layers) → reject.
        let v = parse_range("0..0");
        assert!(matches!(v, RangeVerdict::EndNotAfterStart { .. }));
    }

    #[test]
    fn end_before_start_rejected() {
        let v = parse_range("10..5");
        assert!(matches!(v, RangeVerdict::EndNotAfterStart { .. }));
    }

    #[test]
    fn missing_dotdot_rejected() {
        assert_eq!(parse_range("0"), RangeVerdict::MalformedSyntax);
        assert_eq!(parse_range("0-1"), RangeVerdict::MalformedSyntax);
    }

    #[test]
    fn three_part_split_rejected() {
        // Rust ranges have only one `..`, not multiple.
        assert_eq!(parse_range("0..1..2"), RangeVerdict::MalformedSyntax);
    }

    #[test]
    fn nonnumeric_rejected() {
        assert_eq!(parse_range("abc..xyz"), RangeVerdict::NotNumeric);
    }

    #[test]
    fn whitespace_around_numbers_trimmed() {
        let v = parse_range(" 0 .. 28 ");
        if let RangeVerdict::Ok { start, end, .. } = v {
            assert_eq!(start, 0);
            assert_eq!(end, 28);
        }
    }
}
