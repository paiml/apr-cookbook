#![allow(unused_imports)]
//! # APR QA Capability Check — CLI equivalent: `apr qa_capability model.apr`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Gate 0 pre-flight check: validates that hardware supports a model's required
//! operations before loading weights. Prevents wasted time loading 70B models
//! onto hardware that cannot run them.
//!
//!
//! ## Format Variants
//! ```bash
//! apr qa model.apr          # APR native format
//! apr qa model.gguf         # GGUF (llama.cpp compatible)
//! apr qa model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use std::collections::HashSet;
use std::fmt;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_qa_capability")?;
    println!("=== APR QA Capability Check ===\n");

    // --- Section 1: Define model architectures ---
    println!("--- Model Architectures ---\n");
    let models = define_model_architectures();
    for arch in &models {
        println!(
            "  {}: {} ops [{}]",
            arch.model_name,
            arch.required_ops.len(),
            arch.required_ops.join(", "),
        );
    }

    // --- Section 2: Define hardware profiles ---
    println!("\n--- Hardware Profiles ---\n");
    let profiles = define_hardware_profiles();
    for hw in &profiles {
        let mut sorted_ops: Vec<_> = hw.supported_ops.iter().cloned().collect();
        sorted_ops.sort();
        println!(
            "  {}: {} ops [{}]",
            hw.name,
            hw.supported_ops.len(),
            sorted_ops.join(", "),
        );
    }

    // --- Section 3: Capability matrix ---
    println!("\n--- Capability Matrix ---\n");
    let results = check_all_capabilities(&models, &profiles);
    print_capability_matrix(&models, &profiles, &results);

    // --- Section 4: Failure details and fallback recommendations ---
    println!("\n--- Failure Details & Fallback Recommendations ---\n");
    print_failure_details(&results);

    // --- Section 5: Summary statistics ---
    println!("\n--- Summary ---");
    let total = results.len();
    let pass_count = results
        .iter()
        .filter(|r| r.status == CapStatus::Supported)
        .count();
    let partial_count = results
        .iter()
        .filter(|r| r.status == CapStatus::Partial)
        .count();
    let fail_count = results
        .iter()
        .filter(|r| r.status == CapStatus::Unsupported)
        .count();
    println!("  Total checks:  {total}");
    println!("  Supported:     {pass_count}");
    println!("  Partial:       {partial_count}");
    println!("  Unsupported:   {fail_count}");

    println!("\nCapability check complete.");
    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_arch() -> ArchConstraints {
        define_model_architectures()
            .into_iter()
            .find(|a| a.model_name == "LLaMA")
            .expect("LLaMA architecture must exist")
    }

    fn whisper_arch() -> ArchConstraints {
        define_model_architectures()
            .into_iter()
            .find(|a| a.model_name == "Whisper")
            .expect("Whisper architecture must exist")
    }

    fn sd_arch() -> ArchConstraints {
        define_model_architectures()
            .into_iter()
            .find(|a| a.model_name == "StableDiffusion")
            .expect("StableDiffusion architecture must exist")
    }

    fn full_gpu() -> HardwareProfile {
        define_hardware_profiles()
            .into_iter()
            .find(|h| h.name == "Full GPU")
            .expect("Full GPU profile must exist")
    }

    fn basic_gpu() -> HardwareProfile {
        define_hardware_profiles()
            .into_iter()
            .find(|h| h.name == "Basic GPU")
            .expect("Basic GPU profile must exist")
    }

    fn cpu_only() -> HardwareProfile {
        define_hardware_profiles()
            .into_iter()
            .find(|h| h.name == "CPU Only")
            .expect("CPU Only profile must exist")
    }

    // Test 1: Full GPU supports all models
    #[test]
    fn test_full_gpu_supports_all_models() {
        let hw = full_gpu();
        for arch in &define_model_architectures() {
            let result = check_capability(arch, &hw);
            assert_eq!(
                result.status,
                CapStatus::Supported,
                "{} should be fully supported on Full GPU, missing: {:?}",
                arch.model_name,
                result.missing_ops,
            );
        }
    }

    // Test 2: Basic GPU fails on complex models
    #[test]
    fn test_basic_gpu_fails_stable_diffusion() {
        let result = check_capability(&sd_arch(), &basic_gpu());
        assert_eq!(
            result.status,
            CapStatus::Unsupported,
            "StableDiffusion needs too many ops for Basic GPU",
        );
        assert!(result.missing_ops.len() > 3);
    }

    // Test 3: CPU Only partially supports LLaMA (has rmsnorm, silu but not flash_attention, rope)
    #[test]
    fn test_cpu_only_partial_llama() {
        let result = check_capability(&llama_arch(), &cpu_only());
        assert_eq!(
            result.status,
            CapStatus::Partial,
            "CPU Only should partially support LLaMA",
        );
        assert!(
            result.missing_ops.contains(&"flash_attention".to_string()),
            "flash_attention should be missing on CPU Only",
        );
        assert!(
            result.missing_ops.contains(&"rope".to_string()),
            "rope should be missing on CPU Only",
        );
    }

    // Test 4: Classify status thresholds
    #[test]
    fn test_classify_status_thresholds() {
        assert_eq!(classify_status(6, 0), CapStatus::Supported);
        assert_eq!(classify_status(6, 1), CapStatus::Partial);
        assert_eq!(classify_status(6, 3), CapStatus::Partial);
        assert_eq!(classify_status(6, 4), CapStatus::Unsupported);
        assert_eq!(classify_status(6, 6), CapStatus::Unsupported);
    }

    // Test 5: Empty ops edge case
    #[test]
    fn test_classify_status_zero_ops() {
        // A model with zero required ops is trivially supported
        assert_eq!(classify_status(0, 0), CapStatus::Supported);
    }

    // Test 6: check_all_capabilities returns correct count
    #[test]
    fn test_check_all_count() {
        let models = define_model_architectures();
        let profiles = define_hardware_profiles();
        let results = check_all_capabilities(&models, &profiles);
        assert_eq!(
            results.len(),
            models.len() * profiles.len(),
            "Should have one result per (model, hardware) pair",
        );
    }

    // Test 7: Whisper on CPU Only is partial (missing sinusoidal_pe)
    #[test]
    fn test_whisper_cpu_only_partial() {
        let result = check_capability(&whisper_arch(), &cpu_only());
        assert_eq!(result.status, CapStatus::Partial);
        assert!(
            result.missing_ops.contains(&"sinusoidal_pe".to_string()),
            "sinusoidal_pe should be missing on CPU Only",
        );
    }

    // Test 8: Fallback recommendations exist for all missing ops
    #[test]
    fn test_fallback_recommendations_non_empty() {
        let models = define_model_architectures();
        let profiles = define_hardware_profiles();
        let results = check_all_capabilities(&models, &profiles);
        for r in &results {
            for op in &r.missing_ops {
                let fallback = fallback_for_op(op);
                assert!(
                    !fallback.is_empty(),
                    "Fallback for '{op}' should not be empty",
                );
            }
        }
    }

    // Test 9: CapStatus display formatting
    #[test]
    fn test_cap_status_display() {
        assert_eq!(format!("{}", CapStatus::Supported), "PASS");
        assert_eq!(format!("{}", CapStatus::Partial), "PARTIAL");
        assert_eq!(format!("{}", CapStatus::Unsupported), "FAIL");
    }

    // Test 10: Supported result has empty missing_ops
    #[test]
    fn test_supported_result_has_no_missing_ops() {
        let models = define_model_architectures();
        let hw = full_gpu();
        for arch in &models {
            let result = check_capability(arch, &hw);
            assert!(
                result.missing_ops.is_empty(),
                "{} on Full GPU should have no missing ops, got: {:?}",
                arch.model_name,
                result.missing_ops,
            );
        }
    }
}
