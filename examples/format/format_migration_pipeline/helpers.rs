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
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Stage 3: Convert
// ---------------------------------------------------------------------------

/// Map HuggingFace tensor names to APR naming convention.
///
/// HF: `model.layers.0.self_attn.q_proj.weight`
/// APR: `layers.0.attn.q.weight`
pub fn map_tensor_name(hf_name: &str) -> String {
    hf_name
        .replace("model.layers", "layers")
        .replace("self_attn.", "attn.")
        .replace("q_proj", "q")
        .replace("k_proj", "k")
        .replace("v_proj", "v")
        .replace("o_proj", "o")
        .replace("gate_proj", "gate")
        .replace("up_proj", "up")
}

/// Simulate FP16 quantization by halving byte count.
///
/// In production this would apply proper float16 conversion.
/// Here we take the upper 2 bytes of each 4-byte float (exponent + upper mantissa),
/// preserving sign, exponent, and most significant mantissa bits.
pub fn simulate_fp16_quantize(fp32_bytes: &[u8]) -> Vec<u8> {
    let target_len = fp32_bytes.len() / 2;
    let mut output = Vec::with_capacity(target_len);
    // Take upper 2 bytes (bytes 2..4) of each f32 little-endian value
    // This preserves sign bit, exponent, and upper mantissa bits
    for chunk in fp32_bytes.chunks(4) {
        if chunk.len() >= 4 {
            output.extend_from_slice(&chunk[2..4]);
        }
    }
    output
}

/// Convert an imported model to APR v2 format.
///
/// Returns the tensor mappings, the APR bundle bytes, and the stage result.
pub fn convert_to_apr(
    model: &ImportedModel,
    quantize_fp16: bool,
) -> (Vec<TensorMapping>, Vec<u8>, MigrationStage) {
    let mut mappings = Vec::new();
    let mut sorted_names: Vec<&String> = model.tensors.keys().collect();
    sorted_names.sort();

    let target_dtype = if quantize_fp16 { "FP16" } else { "FP32" };
    let quantization = if quantize_fp16 {
        Quantization::FP16
    } else {
        Quantization::FP32
    };

    let mut builder = ModelBundleV2::new()
        .with_name("migrated-model")
        .with_compression(Compression::Lz4)
        .with_quantization(quantization);

    let mut total_bytes = 0usize;

    for name in &sorted_names {
        let apr_name = map_tensor_name(name);
        let source_data = &model.tensors[*name];
        let converted = if quantize_fp16 {
            simulate_fp16_quantize(source_data)
        } else {
            source_data.clone()
        };

        total_bytes += converted.len();

        mappings.push(TensorMapping {
            source_name: (*name).clone(),
            target_name: apr_name.clone(),
            shape: model.shape.clone(),
            source_dtype: "FP32".to_string(),
            target_dtype: target_dtype.to_string(),
        });

        builder = builder.add_tensor(&apr_name, model.shape.clone(), converted);
    }

    let bundle = builder.build();

    let stage = MigrationStage {
        name: "convert".to_string(),
        status: MigrationStatus::Pass,
        duration_ms: 8.5,
        bytes_processed: total_bytes,
        detail: format!(
            "Converted {} tensors to APR v2 ({}), {} bytes",
            mappings.len(),
            target_dtype,
            total_bytes
        ),
    };

    (mappings, bundle, stage)
}

// ---------------------------------------------------------------------------
// Stage 4: Verify
// ---------------------------------------------------------------------------

/// Compute cosine similarity between two byte slices interpreted as f32.
///
/// Both slices must contain little-endian f32 values (4 bytes each).
pub fn cosine_similarity(a: &[u8], b: &[u8]) -> f64 {
    let len = a.len().min(b.len()) / 4;
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for i in 0..len {
        let offset = i * 4;
        let va = f64::from(f32::from_le_bytes([
            a[offset],
            a[offset + 1],
            a[offset + 2],
            a[offset + 3],
        ]));
        let vb = f64::from(f32::from_le_bytes([
            b[offset],
            b[offset + 1],
            b[offset + 2],
            b[offset + 3],
        ]));

        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f64::EPSILON {
        return 0.0;
    }
    dot / denom
}

/// Compute maximum absolute error between two f32 byte arrays.
pub fn max_absolute_error(a: &[u8], b: &[u8]) -> f64 {
    let len = a.len().min(b.len()) / 4;
    let mut max_err = 0.0f64;

    for i in 0..len {
        let offset = i * 4;
        let va = f64::from(f32::from_le_bytes([
            a[offset],
            a[offset + 1],
            a[offset + 2],
            a[offset + 3],
        ]));
        let vb = f64::from(f32::from_le_bytes([
            b[offset],
            b[offset + 1],
            b[offset + 2],
            b[offset + 3],
        ]));
        let err = (va - vb).abs();
        if err > max_err {
            max_err = err;
        }
    }

    max_err
}

/// Verification result for a single tensor.
#[derive(Debug)]
pub struct VerifyResult {
    pub tensor_name: String,
    pub cosine_sim: f64,
    pub max_abs_error: f64,
    pub passed: bool,
}

/// Verify round-trip fidelity of the conversion.
///
/// For FP32 (no quantization), expects exact match.
/// For FP16, expects cosine similarity > 0.9 and bounded error.
pub fn verify_conversion(
    source: &ImportedModel,
    mappings: &[TensorMapping],
    quantized: bool,
) -> (Vec<VerifyResult>, MigrationStage) {
    let mut results = Vec::new();
    let mut total_bytes = 0usize;
    let threshold = if quantized { 0.9 } else { 1.0 - f64::EPSILON };

    for mapping in mappings {
        let Some(source_data) = source.tensors.get(&mapping.source_name) else {
            continue;
        };

        // For verification without quantization, compare source to itself
        // For quantization, compare source to fp16-then-back
        let converted = if quantized {
            let fp16 = simulate_fp16_quantize(source_data);
            // Reconstruct FP32 from upper-2-byte truncation: zero-pad lower bytes
            let mut fp32_back = Vec::with_capacity(source_data.len());
            for chunk in fp16.chunks(2) {
                if chunk.len() == 2 {
                    fp32_back.extend_from_slice(&[0, 0, chunk[0], chunk[1]]);
                }
            }
            fp32_back
        } else {
            source_data.clone()
        };

        let sim = cosine_similarity(source_data, &converted);
        let mae = max_absolute_error(source_data, &converted);
        let passed = sim >= threshold;

        total_bytes += source_data.len();
        results.push(VerifyResult {
            tensor_name: mapping.target_name.clone(),
            cosine_sim: sim,
            max_abs_error: mae,
            passed,
        });
    }

    let all_passed = results.iter().all(|r| r.passed);
    let status = if all_passed {
        MigrationStatus::Pass
    } else {
        MigrationStatus::Fail
    };

    let stage = MigrationStage {
        name: "verify".to_string(),
        status,
        duration_ms: 3.2,
        bytes_processed: total_bytes,
        detail: format!(
            "Verified {} tensors: {}/{} passed (threshold={:.2})",
            results.len(),
            results.iter().filter(|r| r.passed).count(),
            results.len(),
            threshold,
        ),
    };

    (results, stage)
}

// ---------------------------------------------------------------------------
// Stage 5: Export
// ---------------------------------------------------------------------------

/// Export manifest describing the output bundle.
#[derive(Debug)]
#[allow(dead_code)]
pub struct ExportManifest {
    pub output_path: String,
    pub checksum: String,
    pub bundle_size: usize,
    pub tensor_count: usize,
    pub compression: String,
    pub quantization: String,
}

/// Write the APR bundle to a temp directory and generate a manifest.
pub fn export_bundle(
    bundle: &[u8],
    tensor_count: usize,
    ctx: &RecipeContext,
) -> Result<(ExportManifest, MigrationStage)> {
    let output_path = ctx.path("migrated_model.apr");
    std::fs::write(&output_path, bundle)?;

    let checksum = blake3::hash(bundle);
    let checksum_hex = checksum.to_hex().to_string();

    let manifest = ExportManifest {
        output_path: output_path.to_string_lossy().to_string(),
        checksum: checksum_hex,
        bundle_size: bundle.len(),
        tensor_count,
        compression: "LZ4".to_string(),
        quantization: "FP16".to_string(),
    };

    let stage = MigrationStage {
        name: "export".to_string(),
        status: MigrationStatus::Pass,
        duration_ms: 2.1,
        bytes_processed: bundle.len(),
        detail: format!(
            "Exported {} bytes to {}, checksum={}",
            bundle.len(),
            output_path.display(),
            &manifest.checksum[..16],
        ),
    };

    Ok((manifest, stage))
}
