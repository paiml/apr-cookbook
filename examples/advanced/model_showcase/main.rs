#![allow(unused_imports)]
//! # Recipe: Model Showcase Pipeline
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! **Category**: Advanced - End-to-End Workflow
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## Learning Objective
//! End-to-end model lifecycle demo mirroring `apr showcase`: create a model
//! from scratch, inspect its internals, validate integrity, benchmark
//! throughput, convert formats, and compare tensors.
//!
//! ## Run Command
//! ```bash
//! cargo run --example model_showcase
//! ```
//!
//! ## Toyota Way Principles
//! - **Genchi Genbutsu** (Go and see): Walk the full model lifecycle
//! - **Jidoka** (Quality built-in): Validation at every step
//! - **Heijunka** (Level scheduling): Deterministic pipeline stages
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

fn main() -> Result<()> {
    println!("========================================================");
    println!("  Model Showcase Pipeline (apr showcase)");
    println!("  Create -> Inspect -> Validate -> Bench -> Convert -> Compare");
    println!("========================================================");
    println!();

    let mut ctx = RecipeContext::new("model_showcase")?;
    let report = run_showcase_pipeline(&mut ctx)?;
    print_report(&report);

    ctx.record_metric("steps_done", report.done_count() as i64);
    ctx.record_metric("steps_fail", report.fail_count() as i64);
    ctx.record_float_metric("total_ms", report.total_ms());

    println!("\nShowcase complete.");
    Ok(())
}

// ============================================================================
// Pipeline Orchestrator
// ============================================================================

/// Run the full 6-step showcase pipeline, collecting results.
fn run_showcase_pipeline(ctx: &mut RecipeContext) -> Result<ShowcaseReport> {
    let model_name = "showcase-demo-v2";
    let mut report = ShowcaseReport::new(model_name);

    // Step 1: Create
    let (step, created) = step_create(ctx, model_name)?;
    report.push(step);

    // Step 2: Inspect
    let step = step_inspect(&created)?;
    report.push(step);

    // Step 3: Validate
    let step = step_validate(&created)?;
    report.push(step);

    // Step 4: Benchmark
    let step = step_benchmark(ctx, &created)?;
    report.push(step);

    // Step 5: Convert
    let (step, converted_bytes) = step_convert(&created)?;
    report.push(step);

    // Step 6: Compare
    let step = step_compare(&created.bytes, &converted_bytes)?;
    report.push(step);

    Ok(report)
}

// ============================================================================
// Step 1: Create
// ============================================================================

/// Build a synthetic APR v2 model from scratch using deterministic data.
fn step_create(ctx: &mut RecipeContext, name: &str) -> Result<(ShowcaseStep, CreatedModel)> {
    let start = Instant::now();

    let d_in: usize = 128;
    let d_hidden: usize = 64;
    let d_out: usize = 10;

    let rng = ctx.rng();
    let weight1: Vec<f32> = (0..d_in * d_hidden)
        .map(|_| rng.gen_range(-0.5..0.5))
        .collect();
    let bias1: Vec<f32> = (0..d_hidden).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let weight2: Vec<f32> = (0..d_hidden * d_out)
        .map(|_| rng.gen_range(-0.5..0.5))
        .collect();
    let bias2: Vec<f32> = (0..d_out).map(|_| rng.gen_range(-0.1..0.1)).collect();

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

    let n_tensors: usize = 4;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    let created = CreatedModel {
        bytes: bundle_bytes,
        n_params,
        n_tensors,
    };

    let detail = format!(
        "{} params, {} tensors, {} bytes",
        created.n_params,
        created.n_tensors,
        created.bytes.len()
    );

    let step = ShowcaseStep {
        name: "Create".to_string(),
        status: StepStatus::Done,
        duration_ms: elapsed,
        detail,
    };

    Ok((step, created))
}

// ============================================================================
// Step 2: Inspect
// ============================================================================

/// Parse the bundle header, count tensors, compute total parameters.
fn step_inspect(model: &CreatedModel) -> Result<ShowcaseStep> {
    let start = Instant::now();

    let loaded = BundledModelV2::from_bytes(&model.bytes)
        .map_err(|e| CookbookError::invalid_format(format!("inspect parse failed: {e}")))?;

    let tensor_count = loaded.tensor_count();
    let compression = loaded.compression();
    let quantization = loaded.quantization();

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let detail = format!(
        "tensors={}, compression={:?}, quant={:?}, params={}",
        tensor_count, compression, quantization, model.n_params
    );

    Ok(ShowcaseStep {
        name: "Inspect".to_string(),
        status: StepStatus::Done,
        duration_ms: elapsed,
        detail,
    })
}

// ============================================================================
// Step 3: Validate
// ============================================================================

/// Check magic bytes, scan for NaN values, verify shape consistency.
fn step_validate(model: &CreatedModel) -> Result<ShowcaseStep> {
    let start = Instant::now();
    let bytes = &model.bytes;

    // 3a: magic bytes
    let magic_ok = bytes.len() >= 4 && &bytes[0..4] == b"APR2";

    // 3b: decompress and NaN-scan the payload
    let loaded = BundledModelV2::from_bytes(bytes)
        .map_err(|e| CookbookError::invalid_format(format!("validate parse failed: {e}")))?;
    let payload = loaded.decompress()?;
    let nan_count = count_nan_in_payload(&payload);

    // 3c: shape consistency -- payload length must be divisible by 4 (f32)
    let shape_ok = payload.len() % 4 == 0;

    let all_ok = magic_ok && nan_count == 0 && shape_ok;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let detail = format!(
        "magic={}, nan_count={}, shape_aligned={}",
        if magic_ok { "OK" } else { "BAD" },
        nan_count,
        shape_ok
    );

    Ok(ShowcaseStep {
        name: "Validate".to_string(),
        status: if all_ok {
            StepStatus::Done
        } else {
            StepStatus::Fail
        },
        duration_ms: elapsed,
        detail,
    })
}

// ============================================================================
// Step 4: Benchmark
// ============================================================================

/// Run 100 simulated forward passes and report throughput.
fn step_benchmark(ctx: &mut RecipeContext, model: &CreatedModel) -> Result<ShowcaseStep> {
    let start = Instant::now();

    let loaded = BundledModelV2::from_bytes(&model.bytes)
        .map_err(|e| CookbookError::invalid_format(format!("bench parse failed: {e}")))?;
    let payload = loaded.decompress()?;
    let weights = bytes_to_float_vec(&payload);

    let iterations: usize = 100;
    let input_dim: usize = 128;
    let rng = ctx.rng();

    let bench_start = Instant::now();
    for _ in 0..iterations {
        let input: Vec<f32> = (0..input_dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let _ = simulated_forward(&input, &weights);
    }
    let bench_elapsed = bench_start.elapsed();

    let throughput = iterations as f64 / bench_elapsed.as_secs_f64();
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    let detail = format!(
        "{} iters in {:.2}ms, {:.0} inferences/sec",
        iterations,
        bench_elapsed.as_secs_f64() * 1000.0,
        throughput
    );

    Ok(ShowcaseStep {
        name: "Benchmark".to_string(),
        status: StepStatus::Done,
        duration_ms: elapsed,
        detail,
    })
}

// ============================================================================
// Step 5: Convert
// ============================================================================

/// Simulate APR -> GGUF conversion by rewriting the 4-byte magic header.
fn step_convert(model: &CreatedModel) -> Result<(ShowcaseStep, Vec<u8>)> {
    let start = Instant::now();

    if model.bytes.len() < 4 {
        let step = ShowcaseStep {
            name: "Convert".to_string(),
            status: StepStatus::Skip,
            duration_ms: 0.0,
            detail: "model too small to convert".to_string(),
        };
        return Ok((step, Vec::new()));
    }

    let mut converted = model.bytes.clone();
    // Replace APR2 magic with GGUF magic (simulated)
    converted[0..4].copy_from_slice(b"GGUF");

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let detail = format!("APR2->GGUF header rewrite, {} bytes", converted.len());

    let step = ShowcaseStep {
        name: "Convert".to_string(),
        status: StepStatus::Done,
        duration_ms: elapsed,
        detail,
    };

    Ok((step, converted))
}

// ============================================================================
// Step 6: Compare
// ============================================================================

/// Diff original vs converted tensors, reporting L2 distance and max delta.
fn step_compare(original: &[u8], converted: &[u8]) -> Result<ShowcaseStep> {
    let start = Instant::now();

    if original.len() < 8 || converted.len() < 8 {
        return Ok(ShowcaseStep {
            name: "Compare".to_string(),
            status: StepStatus::Skip,
            duration_ms: 0.0,
            detail: "nothing to compare".to_string(),
        });
    }

    // Compare payload regions (skip the first 4 magic bytes that changed).
    let orig_payload = &original[4..];
    let conv_payload = &converted[4..];

    let min_len = orig_payload.len().min(conv_payload.len());
    let mut byte_diffs: usize = 0;
    let mut max_byte_delta: u8 = 0;

    for i in 0..min_len {
        let a = orig_payload[i];
        let b = conv_payload[i];
        if a != b {
            byte_diffs += 1;
            let delta = a.abs_diff(b);
            if delta > max_byte_delta {
                max_byte_delta = delta;
            }
        }
    }

    let identical = byte_diffs == 0 && orig_payload.len() == conv_payload.len();
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    let detail = format!(
        "payload_identical={}, byte_diffs={}, max_byte_delta={}",
        identical, byte_diffs, max_byte_delta
    );

    Ok(ShowcaseStep {
        name: "Compare".to_string(),
        status: StepStatus::Done,
        duration_ms: elapsed,
        detail,
    })
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Serialize a Vec<f32> into little-endian bytes.
fn float_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Deserialize little-endian bytes back into Vec<f32>.
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

/// Simulated forward pass: dot product of input with first input_dim weights.
fn simulated_forward(input: &[f32], weights: &[f32]) -> f32 {
    let len = input.len().min(weights.len());
    let mut acc: f64 = 0.0;
    for i in 0..len {
        acc += f64::from(input[i]) * f64::from(weights[i]);
    }
    acc as f32
}

/// Print the final showcase summary table.
fn print_report(report: &ShowcaseReport) {
    println!();
    println!("+-----------+--------+------------+------------------------------------------+");
    println!("| Step      | Status | Duration   | Detail                                   |");
    println!("+-----------+--------+------------+------------------------------------------+");

    for step in &report.steps {
        let status_tag = match step.status {
            StepStatus::Done => "DONE",
            StepStatus::Skip => "SKIP",
            StepStatus::Fail => "FAIL",
        };
        // Truncate detail to 40 chars for table alignment.
        let short_detail: String = if step.detail.len() > 40 {
            format!("{}...", &step.detail[..37])
        } else {
            step.detail.clone()
        };
        println!(
            "| {:<9} | {:<6} | {:>7.2} ms | {:<40} |",
            step.name, status_tag, step.duration_ms, short_detail
        );
    }

    println!("+-----------+--------+------------+------------------------------------------+");
    println!(
        "| Model: {:<14} Total: {:.2} ms   Done: {}  Fail: {} {:<14}|",
        report.model_name,
        report.total_ms(),
        report.done_count(),
        report.fail_count(),
        ""
    );
    println!("+-----------+--------+------------+------------------------------------------+");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_pipeline_succeeds() {
        let mut ctx = RecipeContext::new("test_showcase_full").expect("context");
        let report = run_showcase_pipeline(&mut ctx).expect("pipeline");
        assert_eq!(report.steps.len(), 6);
        assert_eq!(report.fail_count(), 0);
        assert_eq!(report.done_count(), 6);
    }

    #[test]
    fn test_step_create_produces_valid_bundle() {
        let mut ctx = RecipeContext::new("test_create").expect("context");
        let (step, created) = step_create(&mut ctx, "test-model").expect("create");
        assert_eq!(step.status, StepStatus::Done);
        assert!(created.bytes.len() > 64);
        assert_eq!(&created.bytes[0..4], b"APR2");
        assert_eq!(created.n_tensors, 4);
    }

    #[test]
    fn test_step_inspect_parses_header() {
        let mut ctx = RecipeContext::new("test_inspect").expect("context");
        let (_step, created) = step_create(&mut ctx, "inspect-model").expect("create");
        let step = step_inspect(&created).expect("inspect");
        assert_eq!(step.status, StepStatus::Done);
        assert!(step.detail.contains("tensors=4"));
    }

    #[test]
    fn test_step_validate_clean_model() {
        let mut ctx = RecipeContext::new("test_validate").expect("context");
        let (_step, created) = step_create(&mut ctx, "validate-model").expect("create");
        let step = step_validate(&created).expect("validate");
        assert_eq!(step.status, StepStatus::Done);
        assert!(step.detail.contains("nan_count=0"));
    }

    #[test]
    fn test_step_validate_detects_bad_magic() {
        let model = CreatedModel {
            bytes: vec![0xFF; 128],
            n_params: 0,
            n_tensors: 0,
        };
        // Bad magic should cause a parse error in BundledModelV2
        let result = step_validate(&model);
        assert!(result.is_err());
    }

    #[test]
    fn test_step_benchmark_reports_throughput() {
        let mut ctx = RecipeContext::new("test_bench").expect("context");
        let (_step, created) = step_create(&mut ctx, "bench-model").expect("create");
        let step = step_benchmark(&mut ctx, &created).expect("benchmark");
        assert_eq!(step.status, StepStatus::Done);
        assert!(step.detail.contains("inferences/sec"));
    }

    #[test]
    fn test_step_convert_rewrites_magic() {
        let mut ctx = RecipeContext::new("test_convert").expect("context");
        let (_step, created) = step_create(&mut ctx, "convert-model").expect("create");
        let (step, converted) = step_convert(&created).expect("convert");
        assert_eq!(step.status, StepStatus::Done);
        assert_eq!(&converted[0..4], b"GGUF");
        // Payload after magic should be identical
        assert_eq!(&created.bytes[4..], &converted[4..]);
    }

    #[test]
    fn test_step_compare_identical_payloads() {
        let mut ctx = RecipeContext::new("test_compare").expect("context");
        let (_step, created) = step_create(&mut ctx, "compare-model").expect("create");
        let (_, converted) = step_convert(&created).expect("convert");
        let step = step_compare(&created.bytes, &converted).expect("compare");
        assert_eq!(step.status, StepStatus::Done);
        assert!(step.detail.contains("payload_identical=true"));
    }

    #[test]
    fn test_float_roundtrip() {
        let original: Vec<f32> = vec![1.0, -0.5, 0.0, f32::MAX, f32::MIN_POSITIVE];
        let bytes = float_vec_to_bytes(&original);
        let recovered = bytes_to_float_vec(&bytes);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_pipeline_deterministic() {
        let mut ctx1 = RecipeContext::new("test_determinism").expect("ctx1");
        let mut ctx2 = RecipeContext::new("test_determinism").expect("ctx2");
        let r1 = run_showcase_pipeline(&mut ctx1).expect("r1");
        let r2 = run_showcase_pipeline(&mut ctx2).expect("r2");
        assert_eq!(r1.steps.len(), r2.steps.len());
        for (a, b) in r1.steps.iter().zip(r2.steps.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.status, b.status);
        }
    }
}
