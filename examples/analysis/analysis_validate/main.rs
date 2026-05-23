#![allow(unused_imports)]
//! # APR Model Validation
//!
//! CLI equivalent: `apr validate model.apr`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Performs a comprehensive 100-point model validation and integrity check.
//! Each check contributes to a pass/fail/warn score, giving a clear picture
//! of model health before deployment.
//!
//!
//! ## Format Variants
//! ```bash
//! apr validate model.apr          # APR native format
//! apr validate model.gguf         # GGUF (llama.cpp compatible)
//! apr validate model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use std::fmt;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_validate")?;

    // --- Section 1: Create test model ---
    println!("=== APR Model Validator ===\n");

    let seed = hash_name_to_seed("validate-model");
    let weight_bytes = generate_model_payload(seed, 128 * 64);
    let bias_bytes = generate_model_payload(seed + 1, 64);

    let bundle = ModelBundleV2::new()
        .with_name("validation-test")
        .with_description("Model for validation demo")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("weight", vec![128, 64], weight_bytes)
        .add_tensor("bias", vec![64], bias_bytes)
        .build();

    let model_path = ctx.path("validation-test.apr");
    std::fs::write(&model_path, &bundle)?;
    println!(
        "Created test model: {} ({} bytes)\n",
        model_path.display(),
        bundle.len()
    );

    // --- Section 2: Run validation on valid model ---
    println!("--- Validating Clean Model ---");
    let result = validate_model(&bundle);
    print_validation_result(&result);

    // --- Section 3: Validate a corrupted model ---
    println!("\n--- Validating Corrupted Model (bad magic) ---");
    let mut corrupted = bundle.clone();
    corrupted[0] = b'X';
    let corrupt_result = validate_model(&corrupted);
    print_validation_result(&corrupt_result);

    // --- Section 4: Validate model with NaN ---
    println!("\n--- Validating Model with NaN ---");
    let mut nan_model = bundle.clone();
    inject_nan_at(&mut nan_model, 80); // inject NaN in payload
    let nan_result = validate_model(&nan_model);
    print_validation_result(&nan_result);

    // --- Section 5: Summary ---
    println!("\n--- Validation Summary ---");
    println!(
        "Clean model:     score={}/100, passed={}, failed={}, warnings={}",
        result.score(),
        result.passed,
        result.failed,
        result.warnings
    );
    println!(
        "Corrupted model: score={}/100, passed={}, failed={}, warnings={}",
        corrupt_result.score(),
        corrupt_result.passed,
        corrupt_result.failed,
        corrupt_result.warnings
    );
    println!(
        "NaN model:       score={}/100, passed={}, failed={}, warnings={}",
        nan_result.score(),
        nan_result.passed,
        nan_result.failed,
        nan_result.warnings
    );

    assert!(result.score() >= 80, "Valid model should score >= 80");
    assert!(
        !corrupt_result.all_passed(),
        "Corrupted model must have failures"
    );

    ctx.report()?;
    Ok(())
}

fn print_validation_result(result: &ValidationResult) {
    println!("\n{:<25} {:<6} Detail", "Check", "Status");
    println!("{}", "-".repeat(75));
    for check in &result.checks {
        println!("{:<25} {:<6} {}", check.name, check.status, check.detail);
    }
    println!(
        "\nScore: {}/100  (passed={}, failed={}, warnings={})",
        result.score(),
        result.passed,
        result.failed,
        result.warnings,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_bundle() -> Vec<u8> {
        let seed = hash_name_to_seed("test-valid");
        let payload = generate_model_payload(seed, 256);
        ModelBundleV2::new()
            .with_name("test-valid")
            .with_description("valid test model")
            .with_compression(Compression::Lz4)
            .with_quantization(Quantization::FP32)
            .add_tensor("weight", vec![16, 16], payload)
            .build()
    }

    #[test]
    fn test_valid_model_passes_all() {
        let result = validate_model(&make_valid_bundle());
        let fails: Vec<_> = result
            .checks
            .iter()
            .filter(|c| c.status == CheckStatus::Fail)
            .collect();
        assert!(
            result.all_passed(),
            "Valid model should pass all checks, but failed: {fails:?}"
        );
    }

    #[test]
    fn test_corrupt_magic_fails() {
        let mut bundle = make_valid_bundle();
        bundle[0] = b'Z';
        let result = validate_model(&bundle);
        let magic_check = result
            .checks
            .iter()
            .find(|c| c.name == "magic_bytes")
            .unwrap();
        assert_eq!(magic_check.status, CheckStatus::Fail);
    }

    #[test]
    fn test_empty_file_fails() {
        let result = validate_model(&[]);
        assert!(result.failed > 0);
    }

    #[test]
    fn test_tiny_file_fails() {
        let result = validate_model(&[0x41, 0x50, 0x52, 0x32]); // Just "APR2"
        let size_check = result
            .checks
            .iter()
            .find(|c| c.name == "minimum_size")
            .unwrap();
        assert_eq!(size_check.status, CheckStatus::Fail);
    }

    #[test]
    fn test_nan_detected() {
        let mut bundle = make_valid_bundle();
        inject_nan_at(&mut bundle, 80);
        let result = validate_model(&bundle);
        let nan_check = result.checks.iter().find(|c| c.name == "no_nan").unwrap();
        assert_eq!(nan_check.status, CheckStatus::Fail);
    }

    #[test]
    fn test_inf_detected() {
        let mut bundle = make_valid_bundle();
        inject_inf_at(&mut bundle, 80);
        let result = validate_model(&bundle);
        let inf_check = result.checks.iter().find(|c| c.name == "no_inf").unwrap();
        assert_eq!(inf_check.status, CheckStatus::Warn);
    }

    #[test]
    fn test_score_calculation_all_pass() {
        let mut r = ValidationResult::new();
        r.add("a", CheckStatus::Pass, "ok");
        r.add("b", CheckStatus::Pass, "ok");
        assert_eq!(r.score(), 100);
    }

    #[test]
    fn test_score_calculation_mixed() {
        let mut r = ValidationResult::new();
        r.add("a", CheckStatus::Pass, "ok");
        r.add("b", CheckStatus::Fail, "bad");
        // 1 pass (100) + 1 fail (0) / 2 = 50
        assert_eq!(r.score(), 50);
    }

    #[test]
    fn test_score_with_warnings() {
        let mut r = ValidationResult::new();
        r.add("a", CheckStatus::Pass, "ok");
        r.add("b", CheckStatus::Warn, "meh");
        // 1 pass (100) + 1 warn (50) / 2 = 75
        assert_eq!(r.score(), 75);
    }
}
