//! # apr finetune — `--checkpoint-format` CSV Parser
//!
//! `apr finetune --checkpoint-format apr,safetensors` writes both formats
//! per checkpoint (default). This recipe builds the CSV parser and asserts
//! the contract: known formats deduped, unknown formats surface as
//! warnings (not silent skip), empty CSV yields default `apr` only.
//!
//! Demonstrates the **FINETUNE.5** recipe for PMAT-104 (apr finetune coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GH-244 + dual-format checkpoint convention
//!
//! Run with: cargo run --example cli_finetune_checkpoint_format_csv
//!
//! Added by PMAT-104 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CheckpointFormat {
    Apr,
    SafeTensors,
}

impl CheckpointFormat {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "apr" => Some(CheckpointFormat::Apr),
            "safetensors" => Some(CheckpointFormat::SafeTensors),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct FormatPlan {
    pub formats: BTreeSet<CheckpointFormat>,
    pub unknown: Vec<String>,
}

pub fn parse_csv(s: &str) -> FormatPlan {
    if s.trim().is_empty() {
        let mut formats = BTreeSet::new();
        formats.insert(CheckpointFormat::Apr);
        return FormatPlan {
            formats,
            unknown: Vec::new(),
        };
    }
    let mut formats = BTreeSet::new();
    let mut unknown = Vec::new();
    for token in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        match CheckpointFormat::from_str_strict(token) {
            Some(f) => {
                formats.insert(f);
            }
            None => unknown.push(token.to_string()),
        }
    }
    FormatPlan { formats, unknown }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_finetune_checkpoint_format_csv")?;

    let cases = [
        "apr,safetensors",
        "apr",
        "safetensors,apr,apr",
        "torch",
        "",
        "apr, ggml",
    ];
    for c in cases {
        println!("{c:>22}  →  {:?}", parse_csv(c));
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
    fn both_formats_parsed() {
        let p = parse_csv("apr,safetensors");
        assert_eq!(p.formats.len(), 2);
        assert!(p.formats.contains(&CheckpointFormat::Apr));
        assert!(p.formats.contains(&CheckpointFormat::SafeTensors));
    }

    #[test]
    fn duplicates_deduped() {
        // BTreeSet → automatic dedup.
        let p = parse_csv("apr,apr,apr");
        assert_eq!(p.formats.len(), 1);
    }

    #[test]
    fn empty_csv_defaults_to_apr_only() {
        // Default is "apr" alone — preserves canonical format selection.
        let p = parse_csv("");
        assert_eq!(p.formats.len(), 1);
        assert!(p.formats.contains(&CheckpointFormat::Apr));
    }

    #[test]
    fn whitespace_trimmed() {
        let p = parse_csv(" apr , safetensors ");
        assert_eq!(p.formats.len(), 2);
    }

    #[test]
    fn unknown_token_separated_from_known() {
        let p = parse_csv("apr,torch,safetensors");
        assert_eq!(p.formats.len(), 2);
        assert_eq!(p.unknown, vec!["torch".to_string()]);
    }

    #[test]
    fn empty_tokens_in_csv_skipped() {
        let p = parse_csv("apr,,,safetensors");
        assert_eq!(p.formats.len(), 2);
        assert!(p.unknown.is_empty());
    }

    #[test]
    fn output_iterates_in_canonical_order() {
        // BTreeSet → Apr < SafeTensors deterministically.
        let p = parse_csv("safetensors,apr");
        let v: Vec<_> = p.formats.iter().collect();
        assert_eq!(v[0], &CheckpointFormat::Apr);
        assert_eq!(v[1], &CheckpointFormat::SafeTensors);
    }
}
