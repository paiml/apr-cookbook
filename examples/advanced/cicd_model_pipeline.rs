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

// ============================================================================
// Data Structures
// ============================================================================

/// Status of a single pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageStatus {
    /// Stage completed successfully.
    Pass,
    /// Stage failed with a defect.
    Fail,
    /// Stage was skipped because a prior stage failed.
    Skip,
}

impl fmt::Display for StageStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Fail => write!(f, "FAIL"),
            Self::Skip => write!(f, "SKIP"),
        }
    }
}

/// Result of a single pipeline stage.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Human-readable stage name.
    pub name: String,
    /// Outcome of this stage.
    pub status: StageStatus,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: f64,
    /// Short detail string describing what happened.
    pub detail: String,
}

/// Final verdict for the pipeline run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineVerdict {
    /// All stages passed; model is safe to deploy.
    Deploy,
    /// At least one stage failed; model is rejected.
    Reject,
}

impl fmt::Display for PipelineVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Deploy => write!(f, "DEPLOY"),
            Self::Reject => write!(f, "REJECT"),
        }
    }
}

/// Aggregated pipeline report.
#[derive(Debug, Clone)]
pub struct PipelineReport {
    /// Ordered list of pipeline stages.
    pub stages: Vec<PipelineStage>,
    /// Name of the model under test.
    pub model_name: String,
    /// Model version string.
    pub version: String,
    /// Overall verdict.
    pub verdict: PipelineVerdict,
}

impl PipelineReport {
    fn new(model_name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            stages: Vec::new(),
            model_name: model_name.into(),
            version: version.into(),
            verdict: PipelineVerdict::Deploy,
        }
    }

    fn push(&mut self, stage: PipelineStage) {
        if stage.status == StageStatus::Fail {
            self.verdict = PipelineVerdict::Reject;
        }
        self.stages.push(stage);
    }

    fn has_failure(&self) -> bool {
        self.verdict == PipelineVerdict::Reject
    }

    fn pass_count(&self) -> usize {
        self.stages
            .iter()
            .filter(|s| s.status == StageStatus::Pass)
            .count()
    }

    fn fail_count(&self) -> usize {
        self.stages
            .iter()
            .filter(|s| s.status == StageStatus::Fail)
            .count()
    }

    fn skip_count(&self) -> usize {
        self.stages
            .iter()
            .filter(|s| s.status == StageStatus::Skip)
            .count()
    }

    fn total_ms(&self) -> f64 {
        self.stages.iter().map(|s| s.duration_ms).sum()
    }
}

/// Manifest generated during the publish stage.
#[derive(Debug, Clone)]
pub struct PublishManifest {
    /// Model name.
    pub name: String,
    /// Model version.
    pub version: String,
    /// Bundle size in bytes.
    pub size_bytes: usize,
    /// BLAKE3 checksum hex string.
    pub checksum: String,
    /// ISO-8601 timestamp string.
    pub timestamp: String,
}

/// Lightweight holder for the model bundle produced in the build stage.
struct BuiltModel {
    /// Raw APR v2 bytes.
    bytes: Vec<u8>,
    /// Number of parameters (weight elements), used in tests and reporting.
    #[allow(dead_code)]
    n_params: usize,
}

// ============================================================================
// Main Entry Point
// ============================================================================

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

/// Produce a skip placeholder. We call `f` only to name the stage, but since
/// the closure hasn't been called yet we use a helper pattern.
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

// ============================================================================
// Stage 1: Build
// ============================================================================

/// Build a synthetic APR v2 model bundle (2-layer neural net, ~10K params).
fn stage_build(
    ctx: &mut RecipeContext,
    name: &str,
    inject_nan: bool,
) -> Result<(PipelineStage, BuiltModel)> {
    let start = Instant::now();

    let d_in: usize = 64;
    let d_hidden: usize = 128;
    let d_out: usize = 10;

    let rng = ctx.rng();
    let mut weight1: Vec<f32> = (0..d_in * d_hidden)
        .map(|_| rng.gen_range(-0.5_f32..0.5))
        .collect();
    let bias1: Vec<f32> = (0..d_hidden)
        .map(|_| rng.gen_range(-0.1_f32..0.1))
        .collect();
    let weight2: Vec<f32> = (0..d_hidden * d_out)
        .map(|_| rng.gen_range(-0.5_f32..0.5))
        .collect();
    let bias2: Vec<f32> = (0..d_out).map(|_| rng.gen_range(-0.1_f32..0.1)).collect();

    if inject_nan {
        // Inject NaN at several positions to trigger validation failure.
        let positions = [0, weight1.len() / 2, weight1.len() - 1];
        for &pos in &positions {
            if pos < weight1.len() {
                weight1[pos] = f32::NAN;
            }
        }
    }

    let n_params = weight1.len() + bias1.len() + weight2.len() + bias2.len();

    let bundle_bytes = ModelBundleV2::new()
        .with_name(name)
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor(
            "fc1.weight",
            vec![d_hidden, d_in],
            float_vec_to_bytes(&weight1),
        )
        .add_tensor("fc1.bias", vec![d_hidden], float_vec_to_bytes(&bias1))
        .add_tensor(
            "fc2.weight",
            vec![d_out, d_hidden],
            float_vec_to_bytes(&weight2),
        )
        .add_tensor("fc2.bias", vec![d_out], float_vec_to_bytes(&bias2))
        .build();

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    let detail = format!(
        "{} params, 4 tensors, {} bytes{}",
        n_params,
        bundle_bytes.len(),
        if inject_nan { ", NaN injected" } else { "" }
    );

    let stage = PipelineStage {
        name: "Build".to_string(),
        status: StageStatus::Pass,
        duration_ms: elapsed,
        detail,
    };

    let built = BuiltModel {
        bytes: bundle_bytes,
        n_params,
    };

    Ok((stage, built))
}

// ============================================================================
// Stage 2: Validate
// ============================================================================

/// Structural validation: magic bytes, header integrity, NaN scan, shape
/// consistency.
fn stage_validate(model: &BuiltModel) -> Result<PipelineStage> {
    let start = Instant::now();
    let bytes = &model.bytes;
    let mut issues: Vec<String> = Vec::new();

    // Check 1: Magic bytes
    let magic_ok = bytes.len() >= 4 && &bytes[0..4] == b"APR2";
    if !magic_ok {
        issues.push("bad magic bytes".to_string());
    }

    // Check 2: Header integrity (minimum 64-byte header for v2)
    let header_ok = bytes.len() >= 64;
    if !header_ok {
        issues.push("header too short".to_string());
    }

    // Check 3: NaN scan
    let loaded = BundledModelV2::from_bytes(bytes)
        .map_err(|e| CookbookError::invalid_format(format!("validate parse failed: {e}")))?;
    let payload = loaded.decompress()?;
    let nan_count = count_nan_in_payload(&payload);
    if nan_count > 0 {
        issues.push(format!("{nan_count} NaN values detected"));
    }

    // Check 4: Shape consistency (payload must be f32-aligned)
    let shape_ok = payload.len() % 4 == 0;
    if !shape_ok {
        issues.push("payload not f32-aligned".to_string());
    }

    let all_ok = issues.is_empty();
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    let detail = if all_ok {
        format!(
            "magic=OK, header=OK, nan=0, shape=aligned ({} bytes)",
            payload.len()
        )
    } else {
        issues.join("; ")
    };

    Ok(PipelineStage {
        name: "Validate".to_string(),
        status: if all_ok {
            StageStatus::Pass
        } else {
            StageStatus::Fail
        },
        duration_ms: elapsed,
        detail,
    })
}

// ============================================================================
// Stage 3: QA Gates
// ============================================================================

/// Run four quality gates: format, integrity, size budget, accuracy threshold.
fn stage_qa_gates(model: &BuiltModel) -> Result<PipelineStage> {
    let start = Instant::now();
    let bytes = &model.bytes;
    let mut gates_passed: u32 = 0;
    let mut gate_details: Vec<String> = Vec::new();
    let total_gates: u32 = 4;

    // Gate 1: Format check (valid APR v2 parse)
    let format_ok = BundledModelV2::from_bytes(bytes).is_ok();
    record_gate(&mut gates_passed, &mut gate_details, "format", format_ok);

    // Gate 2: Integrity check (no NaN in decompressed payload)
    let integrity_ok = check_integrity(bytes);
    record_gate(
        &mut gates_passed,
        &mut gate_details,
        "integrity",
        integrity_ok,
    );

    // Gate 3: Size budget (< 1 MB)
    let size_budget_bytes: usize = 1_000_000;
    let size_ok = bytes.len() < size_budget_bytes;
    record_gate(&mut gates_passed, &mut gate_details, "size<1MB", size_ok);

    // Gate 4: Accuracy threshold (simulated > 70%)
    let accuracy = simulate_accuracy(model);
    let accuracy_ok = accuracy > 0.70;
    record_gate(
        &mut gates_passed,
        &mut gate_details,
        &format!("accuracy={:.1}%>70%", accuracy * 100.0),
        accuracy_ok,
    );

    let all_ok = gates_passed == total_gates;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let detail = format!(
        "{}/{} gates: {}",
        gates_passed,
        total_gates,
        gate_details.join(", ")
    );

    Ok(PipelineStage {
        name: "QA Gates".to_string(),
        status: if all_ok {
            StageStatus::Pass
        } else {
            StageStatus::Fail
        },
        duration_ms: elapsed,
        detail,
    })
}

/// Record a single gate result.
fn record_gate(passed: &mut u32, details: &mut Vec<String>, name: &str, ok: bool) {
    if ok {
        *passed += 1;
        details.push(format!("{name}:OK"));
    } else {
        details.push(format!("{name}:FAIL"));
    }
}

/// Check integrity by decompressing and scanning for NaN.
fn check_integrity(bytes: &[u8]) -> bool {
    let Ok(loaded) = BundledModelV2::from_bytes(bytes) else {
        return false;
    };
    let Ok(payload) = loaded.decompress() else {
        return false;
    };
    count_nan_in_payload(&payload) == 0
}

/// Simulate model accuracy using weight statistics. A valid model (no NaN,
/// reasonable variance) scores above the 70% threshold. Models with NaN
/// or degenerate weights score zero.
fn simulate_accuracy(model: &BuiltModel) -> f64 {
    let Ok(loaded) = BundledModelV2::from_bytes(&model.bytes) else {
        return 0.0;
    };
    let Ok(payload) = loaded.decompress() else {
        return 0.0;
    };
    let weights = bytes_to_float_vec(&payload);

    // If any weight is NaN, accuracy is zero.
    if weights.iter().any(|w| w.is_nan()) {
        return 0.0;
    }

    if weights.is_empty() {
        return 0.0;
    }

    // Compute weight statistics: mean absolute value and variance.
    let n = weights.len() as f64;
    let mean_abs: f64 = weights.iter().map(|w| f64::from(w.abs())).sum::<f64>() / n;
    let variance: f64 = weights
        .iter()
        .map(|w| {
            let v = f64::from(*w);
            (v - 0.0) * (v - 0.0)
        })
        .sum::<f64>()
        / n;

    // A well-initialised model with non-zero variance and reasonable magnitude
    // simulates ~85% accuracy. Degenerate models (all zeros) get ~50%.
    if mean_abs < 1e-10 {
        0.50
    } else {
        // Clamp simulated accuracy between 0.50 and 0.95 based on variance.
        let raw = 0.75 + 0.20 * (variance.sqrt().min(1.0));
        raw.clamp(0.50, 0.95)
    }
}

// ============================================================================
// Stage 4: Benchmark
// ============================================================================

/// Run 50 forward-pass simulations, compute throughput, verify latency < 100ms.
fn stage_benchmark(ctx: &mut RecipeContext, model: &BuiltModel) -> Result<PipelineStage> {
    let start = Instant::now();

    let loaded = BundledModelV2::from_bytes(&model.bytes)
        .map_err(|e| CookbookError::invalid_format(format!("bench parse failed: {e}")))?;
    let payload = loaded.decompress()?;
    let weights = bytes_to_float_vec(&payload);

    let iterations: usize = 50;
    let input_dim: usize = 64;
    let latency_budget_ms: f64 = 100.0;
    let rng = ctx.rng();

    let bench_start = Instant::now();
    for _ in 0..iterations {
        let input: Vec<f32> = (0..input_dim)
            .map(|_| rng.gen_range(-1.0_f32..1.0))
            .collect();
        let _ = simulated_forward(&input, &weights);
    }
    let bench_elapsed = bench_start.elapsed();

    let total_bench_ms = bench_elapsed.as_secs_f64() * 1000.0;
    let throughput = iterations as f64 / bench_elapsed.as_secs_f64();
    let latency_ok = total_bench_ms < latency_budget_ms;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let detail = format!(
        "{} iters in {:.2}ms, {:.0} inf/sec, latency {}",
        iterations,
        total_bench_ms,
        throughput,
        if latency_ok { "OK" } else { "OVER BUDGET" },
    );

    Ok(PipelineStage {
        name: "Benchmark".to_string(),
        status: if latency_ok {
            StageStatus::Pass
        } else {
            StageStatus::Fail
        },
        duration_ms: elapsed,
        detail,
    })
}

// ============================================================================
// Stage 5: Publish
// ============================================================================

/// Simulate publishing: compute checksum, generate manifest, write to temp dir.
fn stage_publish(
    model: &BuiltModel,
    name: &str,
    version: &str,
    ctx: &RecipeContext,
) -> Result<PipelineStage> {
    let start = Instant::now();

    let checksum = blake3::hash(&model.bytes);
    let checksum_hex = checksum.to_hex().to_string();
    let timestamp = "2026-02-25T00:00:00Z".to_string();

    let manifest = PublishManifest {
        name: name.to_string(),
        version: version.to_string(),
        size_bytes: model.bytes.len(),
        checksum: checksum_hex.clone(),
        timestamp: timestamp.clone(),
    };

    let manifest_json = format!(
        "{{\"name\":\"{}\",\"version\":\"{}\",\"size_bytes\":{},\"checksum\":\"{}\",\"timestamp\":\"{}\"}}",
        manifest.name, manifest.version, manifest.size_bytes, manifest.checksum, manifest.timestamp
    );

    // Write manifest and model to temp directory.
    let manifest_path = ctx.path("manifest.json");
    let model_path = ctx.path(&format!("{}-{}.apr", name, version));

    std::fs::write(&manifest_path, manifest_json.as_bytes())?;
    std::fs::write(&model_path, &model.bytes)?;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let detail = format!(
        "checksum={}, manifest={}, model={}",
        &checksum_hex[..16],
        manifest_path.display(),
        model_path.display(),
    );

    Ok(PipelineStage {
        name: "Publish".to_string(),
        status: StageStatus::Pass,
        duration_ms: elapsed,
        detail,
    })
}

// ============================================================================
// Stage 6: Report
// ============================================================================

/// Generate the summary report stage. This always runs regardless of failures.
fn stage_report(report: &PipelineReport) -> PipelineStage {
    let start = Instant::now();

    let pass = report.pass_count();
    let fail = report.fail_count();
    let skip = report.skip_count();
    let verdict = report.verdict;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let detail = format!(
        "pass={}, fail={}, skip={}, verdict={}",
        pass, fail, skip, verdict
    );

    PipelineStage {
        name: "Report".to_string(),
        status: StageStatus::Pass,
        duration_ms: elapsed,
        detail,
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Serialize a `Vec<f32>` into little-endian bytes.
fn float_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Deserialize little-endian bytes back into `Vec<f32>`.
fn bytes_to_float_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(arr)
        })
        .collect()
}

/// Count NaN values in raw f32 payload bytes.
fn count_nan_in_payload(payload: &[u8]) -> usize {
    bytes_to_float_vec(payload)
        .iter()
        .filter(|v| v.is_nan())
        .count()
}

/// Simulated forward pass: dot product of input with first `input.len()` weights.
fn simulated_forward(input: &[f32], weights: &[f32]) -> f32 {
    let len = input.len().min(weights.len());
    let mut acc: f64 = 0.0;
    for i in 0..len {
        acc += f64::from(input[i]) * f64::from(weights[i]);
    }
    acc as f32
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
