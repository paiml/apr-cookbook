//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports, clippy::wildcard_imports)]
use super::types::*;
use apr_cookbook::prelude::*;
use rand::Rng;
use std::time::Instant;

// ============================================================================
// Stage 1: Build
// ============================================================================

/// Build a synthetic APR v2 model bundle (2-layer neural net, ~10K params).
pub fn stage_build(
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
pub fn stage_validate(model: &BuiltModel) -> Result<PipelineStage> {
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
pub fn stage_qa_gates(model: &BuiltModel) -> Result<PipelineStage> {
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
pub fn record_gate(passed: &mut u32, details: &mut Vec<String>, name: &str, ok: bool) {
    if ok {
        *passed += 1;
        details.push(format!("{name}:OK"));
    } else {
        details.push(format!("{name}:FAIL"));
    }
}

/// Check integrity by decompressing and scanning for NaN.
pub fn check_integrity(bytes: &[u8]) -> bool {
    let Ok(loaded) = BundledModelV2::from_bytes(bytes) else {
        return false;
    };
    let Ok(payload) = loaded.decompress() else {
        return false;
    };
    count_nan_in_payload(&payload) == 0
}

/// Simulate model accuracy using weight statistics.
pub fn simulate_accuracy(model: &BuiltModel) -> f64 {
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

    if mean_abs < 1e-10 {
        0.50
    } else {
        let raw = 0.75 + 0.20 * (variance.sqrt().min(1.0));
        raw.clamp(0.50, 0.95)
    }
}

// ============================================================================
// Stage 4: Benchmark
// ============================================================================

/// Run 50 forward-pass simulations, compute throughput, verify latency < 100ms.
pub fn stage_benchmark(ctx: &mut RecipeContext, model: &BuiltModel) -> Result<PipelineStage> {
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
pub fn stage_publish(
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
pub fn stage_report(report: &PipelineReport) -> PipelineStage {
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
pub fn float_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Deserialize little-endian bytes back into `Vec<f32>`.
pub fn bytes_to_float_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(arr)
        })
        .collect()
}

/// Count NaN values in raw f32 payload bytes.
pub fn count_nan_in_payload(payload: &[u8]) -> usize {
    bytes_to_float_vec(payload)
        .iter()
        .filter(|v| v.is_nan())
        .count()
}

/// Simulated forward pass: dot product of input with first `input.len()` weights.
pub fn simulated_forward(input: &[f32], weights: &[f32]) -> f32 {
    let len = input.len().min(weights.len());
    let mut acc: f64 = 0.0;
    for i in 0..len {
        acc += f64::from(input[i]) * f64::from(weights[i]);
    }
    acc as f32
}
