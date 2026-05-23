//! # apr prune — `--remove-layers <RANGE>` Parser
//!
//! `apr prune --method depth --remove-layers <RANGE>` accepts ranges
//! like `20-24` (inclusive). Multiple ranges can be comma-separated:
//! `5,10-12,18`. This recipe builds the parser and asserts the contract:
//! ranges expand to integer sets, malformed input rejects, no overlap
//! with the empty set.
//!
//! Demonstrates the **PRUNE.7** recipe for PMAT-104 (apr prune coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GH-247 + Linux range-spec convention
//!
//! Run with: cargo run --example cli_prune_remove_layers_parser
//!
//! Added by PMAT-104 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq, Eq)]
pub enum ParseVerdict {
    Ok(BTreeSet<u32>),
    Empty,
    MalformedToken(String),
    EndBeforeStart { start: u32, end: u32 },
}

pub fn parse_layer_spec(s: &str) -> ParseVerdict {
    if s.trim().is_empty() {
        return ParseVerdict::Empty;
    }
    let mut layers: BTreeSet<u32> = BTreeSet::new();
    for token in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        if let Some((a, b)) = token.split_once('-') {
            let Ok(start) = a.trim().parse::<u32>() else {
                return ParseVerdict::MalformedToken(token.into());
            };
            let Ok(end) = b.trim().parse::<u32>() else {
                return ParseVerdict::MalformedToken(token.into());
            };
            if end < start {
                return ParseVerdict::EndBeforeStart { start, end };
            }
            for n in start..=end {
                layers.insert(n);
            }
        } else {
            let Ok(n) = token.parse::<u32>() else {
                return ParseVerdict::MalformedToken(token.into());
            };
            layers.insert(n);
        }
    }
    if layers.is_empty() {
        ParseVerdict::Empty
    } else {
        ParseVerdict::Ok(layers)
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_prune_remove_layers_parser")?;

    for s in ["20-24", "5,10-12,18", "1", "10-5", "abc-def", "", "20- 24"] {
        println!("{s:>15}  →  {:?}", parse_layer_spec(s));
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
    fn single_range_expands() {
        let v = parse_layer_spec("20-24");
        if let ParseVerdict::Ok(set) = v {
            assert_eq!(
                set.iter().copied().collect::<Vec<_>>(),
                vec![20, 21, 22, 23, 24]
            );
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn multi_range_with_singletons() {
        let v = parse_layer_spec("5,10-12,18");
        if let ParseVerdict::Ok(set) = v {
            assert_eq!(
                set.iter().copied().collect::<Vec<_>>(),
                vec![5, 10, 11, 12, 18]
            );
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn duplicates_deduped_via_btreeset() {
        let v = parse_layer_spec("5-7,6-8");
        if let ParseVerdict::Ok(set) = v {
            assert_eq!(set.iter().copied().collect::<Vec<_>>(), vec![5, 6, 7, 8]);
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        assert_eq!(parse_layer_spec(""), ParseVerdict::Empty);
        assert_eq!(parse_layer_spec("   "), ParseVerdict::Empty);
    }

    #[test]
    fn end_before_start_rejected() {
        assert_eq!(
            parse_layer_spec("10-5"),
            ParseVerdict::EndBeforeStart { start: 10, end: 5 }
        );
    }

    #[test]
    fn nonnumeric_returns_malformed() {
        let v = parse_layer_spec("abc-def");
        assert!(matches!(v, ParseVerdict::MalformedToken(_)));
    }

    #[test]
    fn whitespace_around_numbers_in_range_trimmed() {
        // "20- 24" must parse as 20..=24.
        let v = parse_layer_spec("20- 24");
        if let ParseVerdict::Ok(set) = v {
            assert!(set.contains(&20));
            assert!(set.contains(&24));
        } else {
            panic!("expected Ok, got {v:?}");
        }
    }

    #[test]
    fn single_layer_no_range() {
        let v = parse_layer_spec("42");
        if let ParseVerdict::Ok(set) = v {
            assert_eq!(set.len(), 1);
            assert!(set.contains(&42));
        }
    }
}
