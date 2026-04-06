#![allow(unused_imports)]
//! # Recipe: CI/CD Model Deployment Pipeline
//!
//! **Category**: Advanced - End-to-End Workflow
//! **CLI Equivalent**: `apr pipeline`
//! Contract: contracts/recipe-iiur-v1.yaml
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## Learning Objective
//! Simulate a full CI/CD pipeline for model deployment composing six stages:
//! build, validate, QA gates, benchmark, publish, and report. Demonstrates
//! how to enforce quality gates, latency budgets, and size budgets before
//! promoting a model to production.
//!
//! ## Run Command
//! ```bash
//! cargo run --example cicd_model_pipeline
//! ```
//!
//! ## Toyota Way Principles
//! - **Jidoka** (Quality built-in): Fail-fast pipeline stops on first defect
//! - **Poka-yoke** (Mistake-proofing): NaN scan, size budget, accuracy threshold
//! - **Heijunka** (Level scheduling): Deterministic stages with wall-clock timing
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;
use rand::Rng;
use std::fmt;
use std::time::Instant;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

mod helpers;
#[allow(unused_imports, clippy::wildcard_imports)]
use helpers::*;

// ============================================================================
// Pipeline Orchestrator
// ============================================================================

/// Run the full 6-stage CI/CD pipeline, collecting results.
///
/// If `inject_nan` is true, the build stage injects NaN values into the model
/// weights, causing downstream validation or QA stages to fail.
fn run_pipeline(
    ctx: &mut RecipeContext,
    model_name: &str,
    version: &str,
    inject_nan: bool,
) -> Result<PipelineReport> {
    let mut report = PipelineReport::new(model_name, version);

    // Stage 1: Build
    let (stage, built) = stage_build(ctx, model_name, inject_nan)?;
    report.push(stage);

    // Stage 2: Validate
    let stage = run_or_skip(&report, || stage_validate(&built));
    report.push(stage);

    // Stage 3: QA Gates
    let stage = run_or_skip(&report, || stage_qa_gates(&built));
    report.push(stage);

    // Stage 4: Benchmark
    let stage = run_or_skip(&report, || stage_benchmark(ctx, &built));
    report.push(stage);

    // Stage 5: Publish
    let stage = run_or_skip(&report, || stage_publish(&built, model_name, version, ctx));
    report.push(stage);

    // Stage 6: Report (always runs -- it just summarises)
    let stage = stage_report(&report);
    report.push(stage);

    Ok(report)
}

/// Run a stage function if no prior failure, otherwise produce a Skip stage.
fn run_or_skip<F>(report: &PipelineReport, f: F) -> PipelineStage
where
    F: FnOnce() -> Result<PipelineStage>,
{
    if report.has_failure() {
        return make_skip_stage(f);
    }
    match f() {
        Ok(stage) => stage,
        Err(e) => PipelineStage {
            name: "unknown".to_string(),
            status: StageStatus::Fail,
            duration_ms: 0.0,
            detail: format!("internal error: {e}"),
        },
    }
}

/// Produce a skip placeholder.
fn make_skip_stage<F>(_f: F) -> PipelineStage
where
    F: FnOnce() -> Result<PipelineStage>,
{
    PipelineStage {
        name: "Skipped".to_string(),
        status: StageStatus::Skip,
        duration_ms: 0.0,
        detail: "prior stage failed".to_string(),
    }
}

fn main() -> Result<()> {
    println!("========================================================");
    println!("  CI/CD Model Deployment Pipeline");
    println!("  Build -> Validate -> QA -> Bench -> Publish -> Report");
    println!("========================================================");
    println!();

    let mut ctx = RecipeContext::new("cicd_model_pipeline")?;

    // --- Run 1: Valid model (expect DEPLOY) ---
    println!("--- Run 1: Valid model ---");
    let report_ok = run_pipeline(&mut ctx, "cicd-demo-v2", "1.0.0", false)?;
    print_report(&report_ok);

    // --- Run 2: Model with NaN (expect REJECT) ---
    println!("\n--- Run 2: Model with NaN injection ---");
    let report_bad = run_pipeline(&mut ctx, "cicd-demo-v2-bad", "1.0.0-bad", true)?;
    print_report(&report_bad);

    ctx.record_metric("run1_pass", report_ok.pass_count() as i64);
    ctx.record_metric("run2_fail", report_bad.fail_count() as i64);
    ctx.record_string_metric("run1_verdict", report_ok.verdict.to_string());
    ctx.record_string_metric("run2_verdict", report_bad.verdict.to_string());

    println!("\nCI/CD pipeline demo complete.");
    Ok(())
}

/// Print the pipeline report table.
fn print_report(report: &PipelineReport) {
    println!();
    println!(
        "+------------+--------+------------+------------------------------------------------+"
    );
    println!(
        "| Stage      | Status | Duration   | Detail                                         |"
    );
    println!(
        "+------------+--------+------------+------------------------------------------------+"
    );

    for stage in &report.stages {
        let status_tag = match stage.status {
            StageStatus::Pass => "PASS",
            StageStatus::Fail => "FAIL",
            StageStatus::Skip => "SKIP",
        };
        let short_detail: String = if stage.detail.len() > 48 {
            format!("{}...", &stage.detail[..45])
        } else {
            stage.detail.clone()
        };
        println!(
            "| {:<10} | {:<6} | {:>7.2} ms | {:<48} |",
            stage.name, status_tag, stage.duration_ms, short_detail
        );
    }

    println!(
        "+------------+--------+------------+------------------------------------------------+"
    );
    println!(
        "| Model: {:<15} Version: {:<8} Verdict: {:<7} Total: {:>7.2} ms       |",
        report.model_name,
        report.version,
        report.verdict,
        report.total_ms(),
    );
    println!(
        "+------------+--------+------------+------------------------------------------------+"
    );
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_pipeline_deploys() {
        let mut ctx = RecipeContext::new("test_cicd_valid").expect("context");
        let report = run_pipeline(&mut ctx, "test-model", "1.0.0", false).expect("pipeline");
        assert_eq!(report.verdict, PipelineVerdict::Deploy);
        assert_eq!(report.stages.len(), 6);
        assert_eq!(report.fail_count(), 0);
        assert_eq!(report.skip_count(), 0);
    }

    #[test]
    fn test_nan_pipeline_rejects() {
        let mut ctx = RecipeContext::new("test_cicd_nan").expect("context");
        let report = run_pipeline(&mut ctx, "bad-model", "0.0.1", true).expect("pipeline");
        assert_eq!(report.verdict, PipelineVerdict::Reject);
        assert!(report.fail_count() >= 1);
        assert!(report.skip_count() >= 1);
    }

    #[test]
    fn test_build_stage_produces_valid_bundle() {
        let mut ctx = RecipeContext::new("test_build").expect("context");
        let (stage, built) = stage_build(&mut ctx, "build-test", false).expect("build");
        assert_eq!(stage.status, StageStatus::Pass);
        assert!(built.bytes.len() > 64);
        assert_eq!(&built.bytes[0..4], b"APR2");
        // ~10K params: 64*128 + 128 + 128*10 + 10 = 9418
        assert!(built.n_params > 9000);
        assert!(built.n_params < 11000);
    }

    #[test]
    fn test_validate_clean_model() {
        let mut ctx = RecipeContext::new("test_validate_clean").expect("context");
        let (_stage, built) = stage_build(&mut ctx, "clean", false).expect("build");
        let stage = stage_validate(&built).expect("validate");
        assert_eq!(stage.status, StageStatus::Pass);
        assert!(stage.detail.contains("nan=0"));
    }

    #[test]
    fn test_validate_nan_model() {
        let mut ctx = RecipeContext::new("test_validate_nan").expect("context");
        let (_stage, built) = stage_build(&mut ctx, "nan-model", true).expect("build");
        let stage = stage_validate(&built).expect("validate");
        assert_eq!(stage.status, StageStatus::Fail);
        assert!(stage.detail.contains("NaN"));
    }

    #[test]
    fn test_qa_gates_clean_model() {
        let mut ctx = RecipeContext::new("test_qa_clean").expect("context");
        let (_stage, built) = stage_build(&mut ctx, "qa-clean", false).expect("build");
        let stage = stage_qa_gates(&built).expect("qa");
        assert_eq!(stage.status, StageStatus::Pass);
        assert!(stage.detail.contains("4/4"));
    }

    #[test]
    fn test_qa_gates_nan_model_fails_integrity() {
        let mut ctx = RecipeContext::new("test_qa_nan").expect("context");
        let (_stage, built) = stage_build(&mut ctx, "qa-nan", true).expect("build");
        let stage = stage_qa_gates(&built).expect("qa");
        assert_eq!(stage.status, StageStatus::Fail);
        assert!(stage.detail.contains("FAIL"));
    }

    #[test]
    fn test_benchmark_within_budget() {
        let mut ctx = RecipeContext::new("test_bench").expect("context");
        let (_stage, built) = stage_build(&mut ctx, "bench-model", false).expect("build");
        let stage = stage_benchmark(&mut ctx, &built).expect("bench");
        assert_eq!(stage.status, StageStatus::Pass);
        assert!(stage.detail.contains("inf/sec"));
    }

    #[test]
    fn test_publish_writes_manifest() {
        let mut ctx = RecipeContext::new("test_publish").expect("context");
        let (_stage, built) = stage_build(&mut ctx, "pub-model", false).expect("build");
        let stage = stage_publish(&built, "pub-model", "2.0.0", &ctx).expect("publish");
        assert_eq!(stage.status, StageStatus::Pass);

        let manifest_path = ctx.path("manifest.json");
        assert!(manifest_path.exists());
        let contents = std::fs::read_to_string(&manifest_path).expect("read manifest");
        assert!(contents.contains("pub-model"));
        assert!(contents.contains("2.0.0"));
    }

    #[test]
    fn test_pipeline_deterministic() {
        let mut ctx1 = RecipeContext::new("test_cicd_det").expect("ctx1");
        let mut ctx2 = RecipeContext::new("test_cicd_det").expect("ctx2");
        let r1 = run_pipeline(&mut ctx1, "det", "1.0.0", false).expect("r1");
        let r2 = run_pipeline(&mut ctx2, "det", "1.0.0", false).expect("r2");
        assert_eq!(r1.stages.len(), r2.stages.len());
        for (a, b) in r1.stages.iter().zip(r2.stages.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.status, b.status);
        }
    }
}
