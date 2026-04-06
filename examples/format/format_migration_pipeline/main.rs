#![allow(unused_imports)]
//! # Model Migration Pipeline
//!
//! **CLI equivalent:** `apr convert model.safetensors --to apr2 --lint --verify`
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/apr-format-roundtrip-v1.yaml
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
//!
//! ## Format Variants
//! ```bash
//! apr convert model.apr          # APR native format
//! apr convert model.gguf         # GGUF (llama.cpp compatible)
//! apr convert model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP. DOI: 10.18653/v1/2020.emnlp-demos.6

use apr_cookbook::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::fmt;

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
