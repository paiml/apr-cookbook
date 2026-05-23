//! # apr trace --save-tensor — Stage CSV Dispatcher
//!
//! `apr trace --save-tensor <STAGES>` accepts a comma-separated list of
//! stage names (`embedding,qkv_matmul,attention`, …) or the literal
//! `all`. This recipe builds the dispatcher and asserts the contract:
//! `all` expands to the full set, dedup, unknown stages surface as warnings.
//!
//! Demonstrates the **TRACE.8** recipe for PMAT-109 (apr trace coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SHIP-007 + apr-cli-trace-save-tensor-v1.yaml
//!
//! Run with: cargo run --example cli_trace_stage_csv_dispatcher
//!
//! Added by PMAT-109 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

const KNOWN_STAGES: &[&str] = &[
    "embedding",
    "qkv_matmul",
    "attention",
    "mlp",
    "norm",
    "softmax",
    "logits",
];

#[derive(Debug, PartialEq, Eq)]
pub struct StagePlan {
    pub stages: BTreeSet<String>,
    pub unknown: Vec<String>,
}

pub fn parse_stages(s: &str) -> StagePlan {
    let trimmed = s.trim();
    if trimmed == "all" {
        return StagePlan {
            stages: KNOWN_STAGES.iter().map(|s| (*s).to_string()).collect(),
            unknown: Vec::new(),
        };
    }
    let mut stages = BTreeSet::new();
    let mut unknown = Vec::new();
    for token in trimmed.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        if KNOWN_STAGES.contains(&token) {
            stages.insert(token.to_string());
        } else {
            unknown.push(token.to_string());
        }
    }
    StagePlan { stages, unknown }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_trace_stage_csv_dispatcher")?;

    for s in [
        "embedding",
        "embedding,qkv_matmul",
        "all",
        "embedding,emb,attention",
        "",
    ] {
        let plan = parse_stages(s);
        println!("--save-tensor {s:>30}");
        println!("  stages: {:?}", plan.stages);
        if !plan.unknown.is_empty() {
            println!("  unknown: {:?}", plan.unknown);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn single_stage_parses() {
        let p = parse_stages("embedding");
        assert_eq!(p.stages.len(), 1);
        assert!(p.stages.contains("embedding"));
    }

    #[test]
    fn multiple_stages_parsed() {
        let p = parse_stages("embedding,qkv_matmul,attention");
        assert_eq!(p.stages.len(), 3);
    }

    #[test]
    fn all_keyword_expands_to_full_set() {
        let p = parse_stages("all");
        assert_eq!(p.stages.len(), KNOWN_STAGES.len());
        for known in KNOWN_STAGES {
            assert!(p.stages.contains(*known));
        }
    }

    #[test]
    fn unknown_stage_separated() {
        let p = parse_stages("embedding,emb,attention");
        assert_eq!(p.stages.len(), 2);
        assert_eq!(p.unknown, vec!["emb".to_string()]);
    }

    #[test]
    fn empty_input_yields_empty_plan() {
        let p = parse_stages("");
        assert!(p.stages.is_empty());
        assert!(p.unknown.is_empty());
    }

    #[test]
    fn duplicates_deduped() {
        let p = parse_stages("embedding,embedding,embedding");
        assert_eq!(p.stages.len(), 1);
    }

    #[test]
    fn whitespace_trimmed() {
        let p = parse_stages(" embedding , qkv_matmul ");
        assert_eq!(p.stages.len(), 2);
    }
}
