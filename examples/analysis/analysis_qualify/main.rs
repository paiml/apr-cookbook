#![allow(unused_imports)]
//! # APR Model Qualification — CLI equivalent: `apr qualify model.apr`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Runs 11 diagnostic gates (smoke tests) to qualify a model for deployment.
//! Each gate produces a Pass/Fail/Skip result with timing. The final report
//! assigns a qualification tier: Smoke (all pass), Qualified (8+ pass),
//! or Rejected.
//!
//!
//! ## Format Variants
//! ```bash
//! apr inspect model.apr          # APR native format
//! apr inspect model.gguf         # GGUF (llama.cpp compatible)
//! apr inspect model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use std::fmt;
use std::time::Instant;

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_qualify")?;
    println!("=== APR Model Qualification ===");

    // --- Section 1: Build a valid synthetic model ---
    let dim: usize = 32;
    let seed = hash_name_to_seed("qualify-model");
    let weight_bytes = generate_model_payload(seed, dim * dim);
    let bias_bytes = generate_model_payload(seed + 1, dim);

    let bundle = ModelBundleV2::new()
        .with_name("qualify-target")
        .with_description("Synthetic model for qualification")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("weight", vec![dim, dim], weight_bytes)
        .add_tensor("bias", vec![dim], bias_bytes)
        .build();

    let model_path = ctx.path("qualify-target.apr");
    std::fs::write(&model_path, &bundle)?;
    println!("Model: qualify-target ({} bytes)", bundle.len());

    // --- Section 2: Qualify the valid model ---
    println!("\n--- Qualifying valid model ---");
    let gates = run_all_gates(&bundle);
    let report = build_report("qualify-target", gates);
    print_report(&report);

    // --- Section 3: Qualify a corrupted model (bad magic) ---
    println!("\n--- Qualifying corrupted model (bad magic) ---");
    let mut bad_magic = bundle.clone();
    bad_magic[0] = b'X';
    let corrupt_report = build_report("corrupt-magic", run_all_gates(&bad_magic));
    print_report(&corrupt_report);

    // --- Section 4: Qualify model with NaN injected (uncompressed for clean injection) ---
    println!("\n--- Qualifying model with NaN ---");
    let mut nan_payload = generate_model_payload(seed, dim * dim);
    // Inject NaN into the first 4 bytes of the weight payload
    nan_payload[0..4].copy_from_slice(&0x7FC0_0000_u32.to_le_bytes());
    let nan_bundle = ModelBundleV2::new()
        .with_name("nan-injected")
        .with_compression(Compression::None)
        .with_quantization(Quantization::FP32)
        .add_tensor("weight", vec![dim, dim], nan_payload)
        .add_tensor("bias", vec![dim], generate_model_payload(seed + 1, dim))
        .build();
    let nan_report = build_report("nan-injected", run_all_gates(&nan_bundle));
    print_report(&nan_report);

    // --- Section 5: Summary ---
    println!("\n--- Summary ---");
    println!("  {:<20} tier={}", report.model_name, report.tier);
    println!(
        "  {:<20} tier={}",
        corrupt_report.model_name, corrupt_report.tier
    );
    println!("  {:<20} tier={}", nan_report.model_name, nan_report.tier);

    println!("\nQualification complete.");
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
        let seed = hash_name_to_seed("qualify-test");
        let payload = generate_model_payload(seed, 32 * 32);
        ModelBundleV2::new()
            .with_name("qualify-test")
            .with_description("test model for qualification")
            .with_compression(Compression::Lz4)
            .with_quantization(Quantization::FP32)
            .add_tensor("weight", vec![32, 32], payload)
            .build()
    }

    fn make_uncompressed_bundle() -> Vec<u8> {
        let seed = hash_name_to_seed("qualify-test-raw");
        let payload = generate_model_payload(seed, 32 * 32);
        ModelBundleV2::new()
            .with_name("qualify-raw")
            .with_compression(Compression::None)
            .with_quantization(Quantization::FP32)
            .add_tensor("weight", vec![32, 32], payload)
            .build()
    }

    // -- Gate-level tests --

    #[test]
    fn test_gate_format_valid_pass_and_fail() {
        let bundle = make_valid_bundle();
        assert_eq!(gate_format_valid(&bundle).status, GateStatus::Pass);

        let mut bad = bundle;
        bad[0] = b'Z';
        assert_eq!(gate_format_valid(&bad).status, GateStatus::Fail);
    }

    #[test]
    fn test_gate_header_parseable_pass_and_fail() {
        let bundle = make_valid_bundle();
        assert_eq!(gate_header_parseable(&bundle).status, GateStatus::Pass);

        // Too short
        assert_eq!(gate_header_parseable(&[0; 10]).status, GateStatus::Fail);

        // Unknown version
        let mut bad = bundle.clone();
        bad[4] = 99;
        assert_eq!(gate_header_parseable(&bad).status, GateStatus::Fail);
    }

    #[test]
    fn test_gate_tensor_loadable_pass_and_fail() {
        let bundle = make_valid_bundle();
        assert_eq!(gate_tensor_loadable(&bundle).status, GateStatus::Pass);

        // Zero out tensor count
        let mut bad = bundle;
        bad[8..12].copy_from_slice(&0_u32.to_le_bytes());
        assert_eq!(gate_tensor_loadable(&bad).status, GateStatus::Fail);
    }

    #[test]
    fn test_gate_no_nan_pass_and_fail() {
        let bundle = make_valid_bundle();
        assert_eq!(gate_no_nan(&bundle).status, GateStatus::Pass);

        // Use uncompressed bundle so NaN injection works on raw payload
        let mut bad = make_uncompressed_bundle();
        let po = payload_offset(&bad);
        if po + 4 <= bad.len() {
            bad[po..po + 4].copy_from_slice(&0x7FC0_0000_u32.to_le_bytes());
        }
        assert_eq!(gate_no_nan(&bad).status, GateStatus::Fail);
    }

    #[test]
    fn test_gate_no_inf_pass_and_fail() {
        let bundle = make_valid_bundle();
        assert_eq!(gate_no_inf(&bundle).status, GateStatus::Pass);

        let mut bad = make_uncompressed_bundle();
        let po = payload_offset(&bad);
        if po + 4 <= bad.len() {
            bad[po..po + 4].copy_from_slice(&0x7F80_0000_u32.to_le_bytes());
        }
        assert_eq!(gate_no_inf(&bad).status, GateStatus::Fail);
    }

    #[test]
    fn test_gate_size_reasonable_pass_and_fail() {
        let bundle = make_valid_bundle();
        assert_eq!(gate_size_reasonable(&bundle).status, GateStatus::Pass);

        // Too small
        assert_eq!(gate_size_reasonable(&[0; 10]).status, GateStatus::Fail);
    }

    #[test]
    fn test_gate_shape_consistent_pass_and_skip() {
        let bundle = make_valid_bundle();
        assert_eq!(gate_shape_consistent(&bundle).status, GateStatus::Pass);

        // Too short to inspect
        assert_eq!(gate_shape_consistent(&[0; 10]).status, GateStatus::Skip);
    }

    #[test]
    fn test_gate_dtype_supported_pass_and_fail() {
        let bundle = make_valid_bundle();
        assert_eq!(gate_dtype_supported(&bundle).status, GateStatus::Pass);

        let mut bad = bundle;
        bad[7] = 99;
        assert_eq!(gate_dtype_supported(&bad).status, GateStatus::Fail);
    }

    #[test]
    fn test_gate_compression_decodable_pass_and_skip() {
        let bundle = make_valid_bundle();
        // LZ4 compressed — should pass
        assert_eq!(gate_compression_decodable(&bundle).status, GateStatus::Pass);

        // Uncompressed model — should skip
        let uncompressed = ModelBundleV2::new()
            .with_name("uncompressed")
            .with_compression(Compression::None)
            .with_quantization(Quantization::FP32)
            .add_tensor("w", vec![4, 4], generate_model_payload(1, 16))
            .build();
        assert_eq!(
            gate_compression_decodable(&uncompressed).status,
            GateStatus::Skip
        );
    }

    #[test]
    fn test_tier_computation() {
        // All pass => Smoke
        let all_pass: Vec<GateResult> = (0..11)
            .map(|i| GateResult::new(&format!("g{i}"), GateStatus::Pass, 0.1, "ok"))
            .collect();
        assert_eq!(compute_tier(&all_pass), QualifyTier::Smoke);

        // 10 pass + 1 skip => Qualified
        let mut with_skip = all_pass.clone();
        with_skip[10] = GateResult::new("g10", GateStatus::Skip, 0.1, "skipped");
        assert_eq!(compute_tier(&with_skip), QualifyTier::Qualified);

        // 7 pass + 4 fail => Rejected
        let mut rejected = all_pass;
        for g in rejected.iter_mut().skip(7) {
            *g = GateResult::new(&g.name.clone(), GateStatus::Fail, 0.1, "bad");
        }
        assert_eq!(compute_tier(&rejected), QualifyTier::Rejected);
    }

    #[test]
    fn test_full_report_valid_model() {
        let bundle = make_valid_bundle();
        let gates = run_all_gates(&bundle);
        let report = build_report("test-model", gates);

        // Valid model should reach Smoke or Qualified
        assert_ne!(report.tier, QualifyTier::Rejected);
        assert_eq!(report.model_name, "test-model");
        assert_eq!(report.gates.len(), 11);

        // No gate should have failed on a valid bundle
        let failures: Vec<_> = report
            .gates
            .iter()
            .filter(|g| g.status == GateStatus::Fail)
            .collect();
        assert!(failures.is_empty(), "Unexpected failures: {failures:?}");
    }
}
