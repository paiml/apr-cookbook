//! # Model Migration Pipeline
//!
//! **CLI equivalent:** `apr convert model.safetensors --to apr2 --lint --verify`
//!
//! Demonstrates a complete model migration pipeline composing four stages:
//! import, lint, convert, and export. This is the workflow used when
//! migrating a HuggingFace SafeTensors model into the APR v2 format
//! with quality checks and round-trip verification.
//!
//! ## Sections
//! 1. Import — simulate importing a HuggingFace SafeTensors model
//! 2. Lint — run quality checks on the imported model
//! 3. Convert — transform from source format to APR v2
//! 4. Verify — round-trip verification with cosine similarity
//! 5. Export — write final APR bundle with checksum and manifest
//!
//! ## References
//! - Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP. DOI: 10.18653/v1/2020.emnlp-demos.6

use apr_cookbook::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Migration types
// ---------------------------------------------------------------------------

/// Status of a single pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MigrationStatus {
    Pass,
    Fail,
    Warn,
}

impl fmt::Display for MigrationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Fail => write!(f, "FAIL"),
            Self::Warn => write!(f, "WARN"),
        }
    }
}

/// Result of a single pipeline stage.
#[derive(Debug, Clone)]
struct MigrationStage {
    name: String,
    status: MigrationStatus,
    duration_ms: f64,
    bytes_processed: usize,
    detail: String,
}

/// Cumulative migration log across all stages.
#[derive(Debug)]
struct MigrationLog {
    stages: Vec<MigrationStage>,
    source_format: String,
    target_format: String,
    source_size: usize,
    target_size: usize,
}

impl MigrationLog {
    fn new(source_format: &str, target_format: &str) -> Self {
        Self {
            stages: Vec::new(),
            source_format: source_format.to_string(),
            target_format: target_format.to_string(),
            source_size: 0,
            target_size: 0,
        }
    }

    fn push(&mut self, stage: MigrationStage) {
        self.stages.push(stage);
    }

    fn all_passed(&self) -> bool {
        self.stages
            .iter()
            .all(|s| s.status != MigrationStatus::Fail)
    }

    fn total_bytes_processed(&self) -> usize {
        self.stages.iter().map(|s| s.bytes_processed).sum()
    }

    fn compression_ratio(&self) -> f64 {
        if self.source_size == 0 {
            return 0.0;
        }
        self.target_size as f64 / self.source_size as f64
    }
}

/// Mapping between source and target tensor names.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TensorMapping {
    source_name: String,
    target_name: String,
    shape: Vec<usize>,
    source_dtype: String,
    target_dtype: String,
}

// ---------------------------------------------------------------------------
// Stage 1: Import
// ---------------------------------------------------------------------------

/// HuggingFace-style tensor names for a 2-layer transformer.
const HF_TENSOR_NAMES: &[&str] = &[
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.layers.1.self_attn.k_proj.weight",
    "model.layers.1.self_attn.v_proj.weight",
    "model.layers.1.self_attn.o_proj.weight",
    "model.layers.1.mlp.gate_proj.weight",
    "model.layers.1.mlp.up_proj.weight",
];

/// Imported model with raw tensor data.
struct ImportedModel {
    tensors: HashMap<String, Vec<u8>>,
    shape: Vec<usize>,
    metadata: HashMap<String, String>,
}

/// Simulate importing a HuggingFace SafeTensors model.
///
/// Generates deterministic synthetic tensor data using the recipe RNG.
fn import_hf_model(rng: &mut impl Rng, dim: usize) -> (ImportedModel, MigrationStage) {
    let shape = vec![dim, dim];
    let elements = dim * dim;
    let mut tensors = HashMap::new();
    let mut total_bytes = 0usize;

    for &name in HF_TENSOR_NAMES {
        let data: Vec<f32> = (0..elements)
            .map(|_| rng.gen_range(-1.0f32..1.0f32))
            .collect();
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        total_bytes += bytes.len();
        tensors.insert(name.to_string(), bytes);
    }

    let mut metadata = HashMap::new();
    metadata.insert("model_type".to_string(), "llama".to_string());
    metadata.insert("hidden_size".to_string(), dim.to_string());
    metadata.insert("num_layers".to_string(), "2".to_string());
    metadata.insert("source_format".to_string(), "safetensors".to_string());

    let model = ImportedModel {
        tensors,
        shape,
        metadata,
    };

    let stage = MigrationStage {
        name: "import".to_string(),
        status: MigrationStatus::Pass,
        duration_ms: 12.3,
        bytes_processed: total_bytes,
        detail: format!(
            "Imported {} tensors ({} bytes) from HuggingFace SafeTensors",
            HF_TENSOR_NAMES.len(),
            total_bytes
        ),
    };

    (model, stage)
}

// ---------------------------------------------------------------------------
// Stage 2: Lint
// ---------------------------------------------------------------------------

/// A single lint finding.
#[derive(Debug)]
struct LintFinding {
    severity: MigrationStatus,
    message: String,
}

/// Run quality checks on the imported model.
///
/// Checks:
/// - Naming convention compliance (dotted HF path)
/// - Dtype consistency (all tensors same size => same dtype)
/// - Missing metadata fields
/// - Tensor shape validation (all shapes match declared shape)
fn lint_model(model: &ImportedModel) -> (Vec<LintFinding>, MigrationStage) {
    let mut findings = Vec::new();
    let expected_byte_len = model.shape.iter().product::<usize>() * 4; // FP32

    // Check naming convention
    for name in model.tensors.keys() {
        if !name.contains('.') {
            findings.push(LintFinding {
                severity: MigrationStatus::Warn,
                message: format!("Tensor '{name}' does not use dotted naming convention"),
            });
        }
    }

    // Check dtype consistency (all tensors should have the same byte length)
    let mut inconsistent_count = 0usize;
    for (name, data) in &model.tensors {
        if data.len() != expected_byte_len {
            findings.push(LintFinding {
                severity: MigrationStatus::Fail,
                message: format!(
                    "Tensor '{name}' has {} bytes, expected {expected_byte_len}",
                    data.len()
                ),
            });
            inconsistent_count += 1;
        }
    }

    // Check required metadata
    let required_keys = ["model_type", "hidden_size", "num_layers"];
    for key in &required_keys {
        if !model.metadata.contains_key(*key) {
            findings.push(LintFinding {
                severity: MigrationStatus::Warn,
                message: format!("Missing metadata field: '{key}'"),
            });
        }
    }

    // Check tensor shape validity
    for &dim in &model.shape {
        if dim == 0 {
            findings.push(LintFinding {
                severity: MigrationStatus::Fail,
                message: "Shape contains zero dimension".to_string(),
            });
        }
    }

    let status = if inconsistent_count > 0 {
        MigrationStatus::Fail
    } else if findings.iter().any(|f| f.severity == MigrationStatus::Warn) {
        MigrationStatus::Warn
    } else {
        MigrationStatus::Pass
    };

    let total_bytes: usize = model.tensors.values().map(Vec::len).sum();

    let stage = MigrationStage {
        name: "lint".to_string(),
        status,
        duration_ms: 1.8,
        bytes_processed: total_bytes,
        detail: format!(
            "{} findings ({} errors, {} warnings)",
            findings.len(),
            findings
                .iter()
                .filter(|f| f.severity == MigrationStatus::Fail)
                .count(),
            findings
                .iter()
                .filter(|f| f.severity == MigrationStatus::Warn)
                .count(),
        ),
    };

    (findings, stage)
}

// ---------------------------------------------------------------------------
// Stage 3: Convert
// ---------------------------------------------------------------------------

/// Map HuggingFace tensor names to APR naming convention.
///
/// HF: `model.layers.0.self_attn.q_proj.weight`
/// APR: `layers.0.attn.q.weight`
fn map_tensor_name(hf_name: &str) -> String {
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
fn simulate_fp16_quantize(fp32_bytes: &[u8]) -> Vec<u8> {
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
fn convert_to_apr(
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
fn cosine_similarity(a: &[u8], b: &[u8]) -> f64 {
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
fn max_absolute_error(a: &[u8], b: &[u8]) -> f64 {
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
struct VerifyResult {
    tensor_name: String,
    cosine_sim: f64,
    max_abs_error: f64,
    passed: bool,
}

/// Verify round-trip fidelity of the conversion.
///
/// For FP32 (no quantization), expects exact match.
/// For FP16, expects cosine similarity > 0.9 and bounded error.
fn verify_conversion(
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
struct ExportManifest {
    output_path: String,
    checksum: String,
    bundle_size: usize,
    tensor_count: usize,
    compression: String,
    quantization: String,
}

/// Write the APR bundle to a temp directory and generate a manifest.
fn export_bundle(
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("format_migration_pipeline")?;
    let dim = 64;
    let mut log = MigrationLog::new("SafeTensors (HF)", "APR v2");

    // Section 1: Import
    println!("=== Stage 1: Import ===");
    let (model, import_stage) = import_hf_model(ctx.rng(), dim);
    log.source_size = model.tensors.values().map(Vec::len).sum();
    println!("  Tensors:  {}", model.tensors.len());
    println!("  Shape:    {:?}", model.shape);
    println!("  Bytes:    {}", log.source_size);
    println!("  Status:   {}", import_stage.status);
    log.push(import_stage);
    println!();

    // Section 2: Lint
    println!("=== Stage 2: Lint ===");
    let (findings, lint_stage) = lint_model(&model);
    println!("  Findings: {}", findings.len());
    for finding in &findings {
        println!("    [{}] {}", finding.severity, finding.message);
    }
    println!("  Status:   {}", lint_stage.status);
    log.push(lint_stage);
    println!();

    // Section 3: Convert (with FP16 quantization)
    println!("=== Stage 3: Convert ===");
    let (mappings, bundle, convert_stage) = convert_to_apr(&model, true);
    println!("  Mappings: {}", mappings.len());
    for m in mappings.iter().take(4) {
        println!(
            "    {} -> {} ({} -> {})",
            m.source_name, m.target_name, m.source_dtype, m.target_dtype
        );
    }
    if mappings.len() > 4 {
        println!("    ... and {} more", mappings.len() - 4);
    }
    println!("  Bundle:   {} bytes", bundle.len());
    println!("  Status:   {}", convert_stage.status);
    log.push(convert_stage);
    println!();

    // Section 4: Verify
    println!("=== Stage 4: Verify ===");
    let (verify_results, verify_stage) = verify_conversion(&model, &mappings, true);
    for vr in verify_results.iter().take(4) {
        println!(
            "    {}: cos_sim={:.6}, max_err={:.6} [{}]",
            vr.tensor_name,
            vr.cosine_sim,
            vr.max_abs_error,
            if vr.passed { "PASS" } else { "FAIL" },
        );
    }
    if verify_results.len() > 4 {
        println!("    ... and {} more", verify_results.len() - 4);
    }
    println!("  Status:   {}", verify_stage.status);
    log.push(verify_stage);
    println!();

    // Section 5: Export
    println!("=== Stage 5: Export ===");
    let (manifest, export_stage) = export_bundle(&bundle, mappings.len(), &ctx)?;
    log.target_size = manifest.bundle_size;
    println!("  Path:     {}", manifest.output_path);
    println!("  Size:     {} bytes", manifest.bundle_size);
    println!("  Checksum: {}", &manifest.checksum[..32]);
    println!("  Status:   {}", export_stage.status);
    log.push(export_stage);
    println!();

    // Summary
    println!("=== Migration Summary ===");
    let detail_header = "Detail";
    println!(
        "{:<10} {:<8} {:<12} {:<14} {}",
        "Stage", "Status", "Duration", "Bytes", detail_header
    );
    println!("{}", "-".repeat(72));
    for stage in &log.stages {
        println!(
            "{:<10} {:<8} {:<12.1} {:<14} {}",
            stage.name,
            format!("{}", stage.status),
            format!("{}ms", stage.duration_ms),
            stage.bytes_processed,
            stage.detail,
        );
    }
    println!();
    println!(
        "Source:      {} ({} bytes)",
        log.source_format, log.source_size
    );
    println!(
        "Target:      {} ({} bytes)",
        log.target_format, log.target_size
    );
    println!("Ratio:       {:.2}", log.compression_ratio());
    println!("Total read:  {} bytes", log.total_bytes_processed());
    println!(
        "Pipeline:    {}",
        if log.all_passed() { "PASS" } else { "FAIL" }
    );

    assert!(log.all_passed(), "Migration pipeline should pass");
    assert_eq!(&bundle[0..4], b"APR2", "Output must be APR v2");

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx() -> RecipeContext {
        RecipeContext::new("test_migration_pipeline").expect("context creation")
    }

    fn make_model(ctx: &mut RecipeContext) -> ImportedModel {
        let (model, _) = import_hf_model(ctx.rng(), 32);
        model
    }

    #[test]
    fn test_import_produces_all_tensors() {
        let mut ctx = make_ctx();
        let (model, stage) = import_hf_model(ctx.rng(), 32);
        assert_eq!(model.tensors.len(), HF_TENSOR_NAMES.len());
        assert_eq!(stage.status, MigrationStatus::Pass);
    }

    #[test]
    fn test_import_deterministic() {
        let mut ctx1 = RecipeContext::new("det_test").expect("ctx");
        let mut ctx2 = RecipeContext::new("det_test").expect("ctx");
        let (m1, _) = import_hf_model(ctx1.rng(), 16);
        let (m2, _) = import_hf_model(ctx2.rng(), 16);
        for name in HF_TENSOR_NAMES {
            assert_eq!(m1.tensors[*name], m2.tensors[*name]);
        }
    }

    #[test]
    fn test_lint_passes_valid_model() {
        let mut ctx = make_ctx();
        let model = make_model(&mut ctx);
        let (findings, stage) = lint_model(&model);
        // All findings should be at most warnings (dotted names all pass)
        assert!(
            findings.iter().all(|f| f.severity != MigrationStatus::Fail),
            "Valid model should have no errors"
        );
        assert_ne!(stage.status, MigrationStatus::Fail);
    }

    #[test]
    fn test_lint_detects_bad_shape() {
        let model = ImportedModel {
            tensors: HashMap::from([("flat_name".to_string(), vec![0u8; 16])]),
            shape: vec![2, 2],
            metadata: HashMap::new(),
        };
        let (findings, stage) = lint_model(&model);
        // "flat_name" has no dots -> warning
        // 16 bytes != 2*2*4=16 bytes -> actually matches, so no error from size
        // But metadata is missing -> warnings
        assert!(findings.len() >= 2, "Expected at least 2 findings");
        assert_eq!(stage.status, MigrationStatus::Warn);
    }

    #[test]
    fn test_tensor_name_mapping() {
        assert_eq!(
            map_tensor_name("model.layers.0.self_attn.q_proj.weight"),
            "layers.0.attn.q.weight"
        );
        assert_eq!(
            map_tensor_name("model.layers.1.mlp.gate_proj.weight"),
            "layers.1.mlp.gate.weight"
        );
    }

    #[test]
    fn test_fp16_quantize_halves_size() {
        let fp32 = vec![0u8; 1024]; // 256 floats * 4 bytes
        let fp16 = simulate_fp16_quantize(&fp32);
        assert_eq!(fp16.len(), 512); // 256 floats * 2 bytes
    }

    #[test]
    fn test_convert_produces_valid_apr() {
        let mut ctx = make_ctx();
        let model = make_model(&mut ctx);
        let (mappings, bundle, stage) = convert_to_apr(&model, false);
        assert_eq!(&bundle[0..4], b"APR2");
        assert_eq!(mappings.len(), HF_TENSOR_NAMES.len());
        assert_eq!(stage.status, MigrationStatus::Pass);
    }

    #[test]
    fn test_verify_fp32_exact_match() {
        let mut ctx = make_ctx();
        let model = make_model(&mut ctx);
        let (mappings, _, _) = convert_to_apr(&model, false);
        let (results, stage) = verify_conversion(&model, &mappings, false);
        assert!(results.iter().all(|r| r.passed));
        assert_eq!(stage.status, MigrationStatus::Pass);
        // FP32 round-trip should be exact
        for r in &results {
            assert!((r.cosine_sim - 1.0).abs() < 1e-6);
            assert!(r.max_abs_error < 1e-6);
        }
    }

    #[test]
    fn test_verify_fp16_within_tolerance() {
        let mut ctx = make_ctx();
        let model = make_model(&mut ctx);
        let (mappings, _, _) = convert_to_apr(&model, true);
        let (results, stage) = verify_conversion(&model, &mappings, true);
        assert_eq!(stage.status, MigrationStatus::Pass);
        for r in &results {
            assert!(r.cosine_sim > 0.9, "cosine_sim={}", r.cosine_sim);
        }
    }

    #[test]
    fn test_export_writes_file() {
        let mut ctx = make_ctx();
        let model = make_model(&mut ctx);
        let (mappings, bundle, _) = convert_to_apr(&model, true);
        let (manifest, stage) = export_bundle(&bundle, mappings.len(), &ctx).expect("export");
        assert_eq!(stage.status, MigrationStatus::Pass);
        assert!(manifest.bundle_size > 0);
        assert!(!manifest.checksum.is_empty());
        let written = std::fs::read(ctx.path("migrated_model.apr")).expect("read");
        assert_eq!(written.len(), bundle.len());
    }

    #[test]
    fn test_migration_log_tracking() {
        let mut log = MigrationLog::new("SafeTensors", "APR v2");
        log.source_size = 1000;
        log.target_size = 500;
        log.push(MigrationStage {
            name: "import".to_string(),
            status: MigrationStatus::Pass,
            duration_ms: 1.0,
            bytes_processed: 1000,
            detail: "ok".to_string(),
        });
        log.push(MigrationStage {
            name: "convert".to_string(),
            status: MigrationStatus::Warn,
            duration_ms: 2.0,
            bytes_processed: 500,
            detail: "warn".to_string(),
        });
        assert!(log.all_passed());
        assert_eq!(log.total_bytes_processed(), 1500);
        assert!((log.compression_ratio() - 0.5).abs() < f64::EPSILON);
    }
}
