//! # apr check — 10-Stage Pipeline Integrity Smoke
//!
//! `apr check <MODEL.apr>` runs a 10-stage integrity pipeline:
//!   1. magic        APR2 header bytes
//!   2. version      header version field
//!   3. crc32        section CRCs
//!   4. tensor-shape shape table well-formedness
//!   5. tensor-dtype dtype enum validity
//!   6. quantization quant scheme matches dtype
//!   7. tokenizer    tokenizer block parseable (if present)
//!   8. provenance   SPDX validity (if present)
//!   9. signature    detached sig verification (if present)
//!  10. contract     attached pv contract round-trips
//!
//! This recipe models the pipeline as an ordered list of stages and asserts
//! the smoke contract: every stage runs in order, a fail short-circuits, and
//! the report is a `Vec<StageOutcome>` (one per stage, no holes).
//!
//! Demonstrates the **CHECK.1** recipe for PMAT-088 (apr check coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CHECK-001 + APR v2 format spec
//!
//! Run with: cargo run --example cli_check_pipeline_integrity_smoke
//!
//! Added by PMAT-088 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
enum StageStatus {
    Pass,
    Fail(&'static str),
    Skipped(&'static str),
}

#[derive(Debug, Clone, PartialEq)]
struct StageOutcome {
    name: &'static str,
    status: StageStatus,
}

const PIPELINE: &[&str] = &[
    "magic",
    "version",
    "crc32",
    "tensor-shape",
    "tensor-dtype",
    "quantization",
    "tokenizer",
    "provenance",
    "signature",
    "contract",
];

fn run_pipeline<F>(stage_eval: F) -> Vec<StageOutcome>
where
    F: Fn(&'static str) -> StageStatus,
{
    let mut out = Vec::with_capacity(PIPELINE.len());
    for &name in PIPELINE {
        let status = stage_eval(name);
        let stop = matches!(status, StageStatus::Fail(_));
        out.push(StageOutcome { name, status });
        if stop {
            break; // short-circuit: a failed stage means later stages can't trust their inputs
        }
    }
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_check_pipeline_integrity_smoke")?;

    // Healthy model: every stage passes.
    let healthy = run_pipeline(|_name| StageStatus::Pass);
    println!("healthy model: {} stages, all pass", healthy.len());

    // Tokenizer-less model: stage 7 is Skipped, the rest pass.
    let no_tokenizer = run_pipeline(|name| match name {
        "tokenizer" => StageStatus::Skipped("model has no tokenizer block"),
        _ => StageStatus::Pass,
    });
    println!(
        "tokenizer-less model: stage 7 = {:?}",
        no_tokenizer[6].status
    );

    // CRC corruption short-circuits at stage 3.
    let corrupt = run_pipeline(|name| match name {
        "crc32" => StageStatus::Fail("section CRC mismatch"),
        _ => StageStatus::Pass,
    });
    println!(
        "corrupt model: short-circuited at {} stages (max 10)",
        corrupt.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn healthy_pipeline_runs_all_ten_stages() {
        let report = run_pipeline(|_| StageStatus::Pass);
        assert_eq!(report.len(), 10);
        assert!(report.iter().all(|s| s.status == StageStatus::Pass));
    }

    #[test]
    fn pipeline_runs_in_declared_order() {
        let report = run_pipeline(|_| StageStatus::Pass);
        let names: Vec<&'static str> = report.iter().map(|s| s.name).collect();
        assert_eq!(names, PIPELINE);
    }

    #[test]
    fn fail_short_circuits_remainder() {
        let report = run_pipeline(|name| match name {
            "tensor-shape" => StageStatus::Fail("rank mismatch"),
            _ => StageStatus::Pass,
        });
        // Stages 1..=4 ran (magic, version, crc32, tensor-shape).
        assert_eq!(report.len(), 4);
        assert!(matches!(report[3].status, StageStatus::Fail(_)));
    }

    #[test]
    fn skipped_stages_do_not_short_circuit() {
        let report = run_pipeline(|name| match name {
            "signature" => StageStatus::Skipped("no detached signature attached"),
            _ => StageStatus::Pass,
        });
        // Skipped does not stop the pipeline; we still see all 10 stages.
        assert_eq!(report.len(), 10);
        assert!(matches!(report[8].status, StageStatus::Skipped(_)));
    }
}
