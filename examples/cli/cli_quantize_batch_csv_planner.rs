//! # apr quantize — `--batch` Multi-Scheme Planner
//!
//! `apr quantize <FILE> --batch int4,int8,fp16` runs multiple schemes in
//! one invocation. This recipe builds the parser + per-scheme output-path
//! generator and asserts the contract: dedup schemes, reject unknowns,
//! generate predictable filenames `<stem>.<scheme>.<ext>`.
//!
//! Demonstrates the **QUANTIZE.13** recipe for PMAT-105 (apr quantize coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GH-243
//!
//! Run with: cargo run --example cli_quantize_batch_csv_planner
//!
//! Added by PMAT-105 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;
use std::path::PathBuf;

const KNOWN_SCHEMES: &[&str] = &["int8", "int4", "fp16", "q4k"];

#[derive(Debug, PartialEq)]
pub struct BatchPlan {
    pub schemes: BTreeSet<String>,
    pub outputs: Vec<PathBuf>,
    pub unknown: Vec<String>,
}

pub fn plan_batch(input_stem: &str, batch_csv: &str) -> BatchPlan {
    let mut schemes = BTreeSet::new();
    let mut unknown = Vec::new();
    for token in batch_csv
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        if KNOWN_SCHEMES.contains(&token) {
            schemes.insert(token.to_string());
        } else {
            unknown.push(token.to_string());
        }
    }
    let outputs: Vec<PathBuf> = schemes
        .iter()
        .map(|s| PathBuf::from(format!("{input_stem}.{s}.apr")))
        .collect();
    BatchPlan {
        schemes,
        outputs,
        unknown,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_quantize_batch_csv_planner")?;

    for csv in [
        "int4,int8,fp16",
        "int4,int4,int4",
        "int4,bogus,q4k",
        "",
        " int4 , int8 ",
    ] {
        let plan = plan_batch("model", csv);
        println!("--batch {csv:>22}");
        println!("  schemes: {:?}", plan.schemes);
        println!("  outputs: {:?}", plan.outputs);
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
    fn planner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn three_schemes_yield_three_outputs() {
        let p = plan_batch("m", "int4,int8,fp16");
        assert_eq!(p.schemes.len(), 3);
        assert_eq!(p.outputs.len(), 3);
        assert!(p.unknown.is_empty());
    }

    #[test]
    fn duplicate_schemes_deduped() {
        let p = plan_batch("m", "int4,int4,int4");
        assert_eq!(p.schemes.len(), 1);
        assert_eq!(p.outputs.len(), 1);
    }

    #[test]
    fn unknown_scheme_separated_from_known() {
        let p = plan_batch("m", "int4,bogus,q4k");
        assert_eq!(p.schemes.len(), 2);
        assert_eq!(p.unknown, vec!["bogus".to_string()]);
    }

    #[test]
    fn empty_csv_yields_empty_plan() {
        let p = plan_batch("m", "");
        assert!(p.schemes.is_empty());
        assert!(p.outputs.is_empty());
        assert!(p.unknown.is_empty());
    }

    #[test]
    fn whitespace_in_csv_trimmed() {
        let p = plan_batch("m", "  int4 , int8  ");
        assert_eq!(p.schemes.len(), 2);
    }

    #[test]
    fn output_naming_per_scheme() {
        let p = plan_batch("model", "int4");
        assert_eq!(p.outputs[0].to_string_lossy(), "model.int4.apr");
    }

    #[test]
    fn outputs_sorted_by_scheme_via_btreeset() {
        // BTreeSet → "fp16" < "int4" < "int8" < "q4k" alphabetical.
        let p = plan_batch("m", "q4k,fp16,int8,int4");
        let names: Vec<String> = p
            .outputs
            .iter()
            .map(|p| p.to_string_lossy().into())
            .collect();
        assert_eq!(names[0], "m.fp16.apr");
        assert_eq!(names[1], "m.int4.apr");
        assert_eq!(names[2], "m.int8.apr");
        assert_eq!(names[3], "m.q4k.apr");
    }
}
