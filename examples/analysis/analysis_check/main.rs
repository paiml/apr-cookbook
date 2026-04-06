#![allow(unused_imports)]
//! # APR Model Pre-Flight Check
//!
//! CLI equivalent: `apr check model.apr`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Runs a 10-stage sequential pre-flight health check pipeline on an APR
//! model file. Each stage produces a pass/fail/skip result with detail.
//! The final report summarizes overall model readiness for deployment.
//!
//!
//! ## Format Variants
//! ```bash
//! apr check model.apr          # APR native format
//! apr check model.gguf         # GGUF (llama.cpp compatible)
//! apr check model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
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
    let ctx = RecipeContext::new("analysis_check")?;
    println!("=== APR Model Pre-Flight Check ===\n");

    // --- Section 1: Build a synthetic model to check ---
    let seed = hash_name_to_seed("check-model");
    let weight_bytes = generate_model_payload(seed, 64 * 32);
    let bias_bytes = generate_model_payload(seed + 1, 32);
    let embed_bytes = generate_model_payload(seed + 2, 128 * 32);

    let bundle = ModelBundleV2::new()
        .with_name("check-target")
        .with_description("Synthetic model for pre-flight check demo")
        .with_compression(Compression::None)
        .with_quantization(Quantization::FP32)
        .add_tensor("fc.weight", vec![64, 32], weight_bytes)
        .add_tensor("fc.bias", vec![32], bias_bytes)
        .add_tensor("embed.weight", vec![128, 32], embed_bytes)
        .build();

    let model_path = ctx.path("check-target.apr");
    std::fs::write(&model_path, &bundle)?;
    println!("Model: check-target ({} bytes)\n", bundle.len(),);

    // --- Section 2: Run 10-stage check on valid model ---
    println!("--- Clean Model Check ---");
    let report = run_check_pipeline("check-target", &bundle);
    print_report(&report);

    assert_eq!(
        report.verdict(),
        CheckVerdict::Pass,
        "Clean model should pass"
    );

    // --- Section 3: Check a model with bad magic ---
    println!("\n--- Corrupted Magic Check ---");
    let mut bad_magic = bundle.clone();
    bad_magic[0] = b'X';
    let bad_report = run_check_pipeline("bad-magic", &bad_magic);
    print_report(&bad_report);

    assert_eq!(bad_report.verdict(), CheckVerdict::Fail);

    // --- Section 4: Check a model with injected NaN ---
    println!("\n--- NaN-Injected Model Check ---");
    let mut nan_model = bundle.clone();
    let payload_off = get_payload_start(&nan_model);
    if payload_off + 4 <= nan_model.len() {
        let nan_bits: u32 = 0x7FC0_0000;
        nan_model[payload_off..payload_off + 4].copy_from_slice(&nan_bits.to_le_bytes());
    }
    let nan_report = run_check_pipeline("nan-injected", &nan_model);
    print_report(&nan_report);

    assert_eq!(nan_report.verdict(), CheckVerdict::Fail);

    // --- Section 5: Summary ---
    println!("\n--- Overall Summary ---");
    println!("Clean model:   {}", report.verdict());
    println!("Bad magic:     {}", bad_report.verdict());
    println!("NaN injected:  {}", nan_report.verdict());
    println!("\nPre-flight check pipeline complete.");

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_bundle() -> Vec<u8> {
        let seed = hash_name_to_seed("check-test");
        let w = generate_model_payload(seed, 16 * 16);
        let b = generate_model_payload(seed + 1, 16);
        ModelBundleV2::new()
            .with_name("check-test")
            .with_description("test bundle")
            .with_compression(Compression::None)
            .with_quantization(Quantization::FP32)
            .add_tensor("weight", vec![16, 16], w)
            .add_tensor("bias", vec![16], b)
            .build()
    }

    #[test]
    fn test_clean_model_passes_all() {
        let bundle = make_valid_bundle();
        let report = run_check_pipeline("test-clean", &bundle);
        assert_eq!(
            report.failed_count(),
            0,
            "Clean model should have zero failures: {:?}",
            report
                .stages
                .iter()
                .filter(|s| !s.passed && !s.skipped)
                .map(|s| &s.name)
                .collect::<Vec<_>>()
        );
        assert_eq!(report.verdict(), CheckVerdict::Pass);
    }

    #[test]
    fn test_bad_magic_fails_stage1() {
        let mut bundle = make_valid_bundle();
        bundle[0] = b'Z';
        let report = run_check_pipeline("bad-magic", &bundle);
        let stage1 = &report.stages[0];
        assert!(!stage1.passed, "Bad magic should fail stage 1");
        assert!(!stage1.skipped);
        assert_eq!(report.verdict(), CheckVerdict::Fail);
    }

    #[test]
    fn test_short_file_fails_header() {
        let report = run_check_pipeline("tiny", &[0x41, 0x50, 0x52, 0x32]);
        let header_stage = &report.stages[1];
        assert!(
            !header_stage.passed || header_stage.skipped,
            "4-byte file should fail header integrity"
        );
    }

    #[test]
    fn test_zero_tensor_count_fails() {
        let mut bundle = make_valid_bundle();
        // Overwrite tensor count at bytes [8..12] with zero
        bundle[8] = 0;
        bundle[9] = 0;
        bundle[10] = 0;
        bundle[11] = 0;
        let report = run_check_pipeline("zero-tensors", &bundle);
        let stage3 = &report.stages[2];
        assert!(!stage3.passed, "Zero tensor count should fail");
    }

    #[test]
    fn test_unknown_dtype_fails() {
        let mut bundle = make_valid_bundle();
        // Overwrite dtype byte with an invalid code
        bundle[7] = 0xFF;
        let report = run_check_pipeline("bad-dtype", &bundle);
        let stage5 = &report.stages[4];
        assert!(!stage5.passed, "Unknown dtype should fail");
    }

    #[test]
    fn test_nan_injection_fails_scan() {
        let mut bundle = make_valid_bundle();
        let off = get_payload_start(&bundle);
        if off + 4 <= bundle.len() {
            let nan_bits: u32 = 0x7FC0_0000;
            bundle[off..off + 4].copy_from_slice(&nan_bits.to_le_bytes());
        }
        let report = run_check_pipeline("nan-model", &bundle);
        let stage7 = &report.stages[6];
        assert!(!stage7.passed, "NaN should fail stage 7");
    }

    #[test]
    fn test_sparsity_stage_always_passes() {
        let bundle = make_valid_bundle();
        let report = run_check_pipeline("sparsity", &bundle);
        let stage8 = &report.stages[7];
        assert!(
            stage8.passed || stage8.skipped,
            "Sparsity is informational and should pass or skip"
        );
    }

    #[test]
    fn test_checksum_nonzero_for_valid_model() {
        let bundle = make_valid_bundle();
        let report = run_check_pipeline("checksum", &bundle);
        let stage10 = &report.stages[9];
        assert!(
            stage10.passed,
            "Valid model should have non-degenerate checksum"
        );
    }

    #[test]
    fn test_report_counts_correct() {
        let mut report = CheckReport::new("counts-test");
        report.add(StageResult::pass("a", "ok"));
        report.add(StageResult::fail("b", "bad"));
        report.add(StageResult::skip("c", "n/a"));
        report.add(StageResult::pass("d", "ok"));

        assert_eq!(report.passed_count(), 2);
        assert_eq!(report.failed_count(), 1);
        assert_eq!(report.skipped_count(), 1);
        assert_eq!(report.verdict(), CheckVerdict::Fail);
    }

    #[test]
    fn test_verdict_warn_on_skips_only() {
        let mut report = CheckReport::new("warn-test");
        report.add(StageResult::pass("a", "ok"));
        report.add(StageResult::skip("b", "skipped"));
        report.add(StageResult::pass("c", "ok"));

        assert_eq!(report.verdict(), CheckVerdict::Warn);
    }
}
