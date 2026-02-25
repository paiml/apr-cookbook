//! # APR Model Validation
//!
//! CLI equivalent: `apr validate model.apr`
//!
//! Performs a comprehensive 100-point model validation and integrity check.
//! Each check contributes to a pass/fail/warn score, giving a clear picture
//! of model health before deployment.

use apr_cookbook::prelude::*;
use std::fmt;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckStatus {
    Pass,
    Fail,
    Warn,
}

impl fmt::Display for CheckStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckStatus::Pass => write!(f, "PASS"),
            CheckStatus::Fail => write!(f, "FAIL"),
            CheckStatus::Warn => write!(f, "WARN"),
        }
    }
}

#[derive(Debug, Clone)]
struct Check {
    name: String,
    status: CheckStatus,
    detail: String,
}

#[derive(Debug, Clone)]
struct ValidationResult {
    passed: u32,
    failed: u32,
    warnings: u32,
    checks: Vec<Check>,
}

impl ValidationResult {
    fn new() -> Self {
        Self {
            passed: 0,
            failed: 0,
            warnings: 0,
            checks: Vec::new(),
        }
    }

    fn add(&mut self, name: &str, status: CheckStatus, detail: &str) {
        match status {
            CheckStatus::Pass => self.passed += 1,
            CheckStatus::Fail => self.failed += 1,
            CheckStatus::Warn => self.warnings += 1,
        }
        self.checks.push(Check {
            name: name.to_string(),
            status,
            detail: detail.to_string(),
        });
    }

    fn score(&self) -> u32 {
        let total = self.passed + self.failed + self.warnings;
        if total == 0 {
            return 0;
        }
        // Each pass = full points, warn = half, fail = 0
        let earned = self.passed * 100 + self.warnings * 50;
        earned / total
    }

    fn all_passed(&self) -> bool {
        self.failed == 0
    }
}

// ---------------------------------------------------------------------------
// Validation checks
// ---------------------------------------------------------------------------

fn validate_model(bytes: &[u8]) -> ValidationResult {
    let mut result = ValidationResult::new();

    // Check 1: Magic bytes
    check_magic_bytes(bytes, &mut result);

    // Check 2: Minimum file size
    check_minimum_size(bytes, &mut result);

    // Check 3: Format version
    check_version(bytes, &mut result);

    // Check 4: Metadata present
    check_metadata_present(bytes, &mut result);

    // Check 5: Tensors non-empty
    check_tensors_nonempty(bytes, &mut result);

    // Check 6: No NaN values in payload
    check_no_nan(bytes, &mut result);

    // Check 7: No Inf values in payload
    check_no_inf(bytes, &mut result);

    // Check 8: Compression validity
    check_compression_valid(bytes, &mut result);

    // Check 9: Size consistency
    check_size_consistent(bytes, &mut result);

    // Check 10: Checksum
    check_checksum(bytes, &mut result);

    result
}

fn check_magic_bytes(bytes: &[u8], result: &mut ValidationResult) {
    if bytes.len() >= 4 && &bytes[0..4] == b"APR2" {
        result.add("magic_bytes", CheckStatus::Pass, "Valid APR2 magic bytes");
    } else if bytes.len() >= 4 {
        result.add(
            "magic_bytes",
            CheckStatus::Fail,
            &format!("Invalid magic: {:?}", &bytes[0..4]),
        );
    } else {
        result.add(
            "magic_bytes",
            CheckStatus::Fail,
            "File too small for magic bytes",
        );
    }
}

fn check_minimum_size(bytes: &[u8], result: &mut ValidationResult) {
    if bytes.len() >= 64 {
        result.add(
            "minimum_size",
            CheckStatus::Pass,
            &format!("{} bytes (>= 64 byte minimum)", bytes.len()),
        );
    } else {
        result.add(
            "minimum_size",
            CheckStatus::Fail,
            &format!("{} bytes (< 64 byte minimum)", bytes.len()),
        );
    }
}

fn check_version(bytes: &[u8], result: &mut ValidationResult) {
    if bytes.len() > 4 {
        let version = bytes[4];
        if version == 2 || version == 0 {
            result.add(
                "format_version",
                CheckStatus::Pass,
                &format!("Version byte: {version}"),
            );
        } else if version == 1 {
            result.add(
                "format_version",
                CheckStatus::Warn,
                "Legacy v1 format detected; consider upgrading",
            );
        } else {
            result.add(
                "format_version",
                CheckStatus::Fail,
                &format!("Unknown version: {version}"),
            );
        }
    } else {
        result.add(
            "format_version",
            CheckStatus::Fail,
            "File too small for version byte",
        );
    }
}

fn check_metadata_present(bytes: &[u8], result: &mut ValidationResult) {
    // Look for metadata markers (name=, description=)
    let has_name = bytes.windows(5).any(|w| w == b"name=");
    let has_desc = bytes.windows(12).any(|w| w == b"description=");
    if has_name {
        result.add(
            "metadata_present",
            CheckStatus::Pass,
            "Model name found in metadata",
        );
    } else if has_desc {
        result.add(
            "metadata_present",
            CheckStatus::Warn,
            "Name missing but description present",
        );
    } else {
        result.add(
            "metadata_present",
            CheckStatus::Warn,
            "No standard metadata fields found",
        );
    }
}

fn check_tensors_nonempty(bytes: &[u8], result: &mut ValidationResult) {
    // Payload must contain actual data beyond the header
    let header_size = 64;
    if bytes.len() > header_size + 16 {
        let payload_size = bytes.len() - header_size;
        result.add(
            "tensors_nonempty",
            CheckStatus::Pass,
            &format!("Payload size: {payload_size} bytes"),
        );
    } else if bytes.len() > header_size {
        result.add(
            "tensors_nonempty",
            CheckStatus::Warn,
            "Payload very small; model may be trivial",
        );
    } else {
        result.add(
            "tensors_nonempty",
            CheckStatus::Fail,
            "No tensor payload detected",
        );
    }
}

fn check_no_nan(bytes: &[u8], result: &mut ValidationResult) {
    let nan_count = count_nan_values(bytes);
    if nan_count == 0 {
        result.add("no_nan", CheckStatus::Pass, "No NaN values detected");
    } else {
        result.add(
            "no_nan",
            CheckStatus::Fail,
            &format!("{nan_count} NaN value(s) detected in tensor data"),
        );
    }
}

fn check_no_inf(bytes: &[u8], result: &mut ValidationResult) {
    let inf_count = count_inf_values(bytes);
    if inf_count == 0 {
        result.add("no_inf", CheckStatus::Pass, "No Inf values detected");
    } else {
        result.add(
            "no_inf",
            CheckStatus::Warn,
            &format!("{inf_count} Inf value(s) detected in tensor data"),
        );
    }
}

fn check_compression_valid(bytes: &[u8], result: &mut ValidationResult) {
    // If compression markers exist, verify they are recognized
    let lz4_magic: [u8; 4] = [0x04, 0x22, 0x4D, 0x18];
    let zstd_magic: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];
    let has_lz4 = bytes.windows(4).any(|w| w == lz4_magic);
    let has_zstd = bytes.windows(4).any(|w| w == zstd_magic);

    if has_lz4 && has_zstd {
        result.add(
            "compression_valid",
            CheckStatus::Warn,
            "Multiple compression formats detected",
        );
    } else if has_lz4 || has_zstd || bytes.len() < 256 {
        let method = if has_lz4 {
            "LZ4"
        } else if has_zstd {
            "Zstd"
        } else {
            "None/uncompressed"
        };
        result.add(
            "compression_valid",
            CheckStatus::Pass,
            &format!("Compression: {method}"),
        );
    } else {
        result.add(
            "compression_valid",
            CheckStatus::Pass,
            "No compression detected (raw payload)",
        );
    }
}

fn check_size_consistent(bytes: &[u8], result: &mut ValidationResult) {
    // Check that file size is reasonable (not truncated)
    if bytes.len() >= 64 && bytes.len() % 4 == 0 {
        result.add(
            "size_consistent",
            CheckStatus::Pass,
            "File size is 4-byte aligned",
        );
    } else if bytes.len() >= 64 {
        result.add(
            "size_consistent",
            CheckStatus::Warn,
            "File size not 4-byte aligned; possible truncation",
        );
    } else {
        result.add(
            "size_consistent",
            CheckStatus::Fail,
            "File smaller than minimum header size",
        );
    }
}

fn check_checksum(bytes: &[u8], result: &mut ValidationResult) {
    // Compute a simple checksum over the payload
    let checksum = compute_checksum(bytes);
    if checksum > 0 {
        result.add(
            "checksum",
            CheckStatus::Pass,
            &format!("Checksum: 0x{checksum:08X}"),
        );
    } else {
        result.add(
            "checksum",
            CheckStatus::Warn,
            "Zero checksum; payload may be all zeros",
        );
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn count_nan_values(bytes: &[u8]) -> usize {
    let payload_start = 64.min(bytes.len());
    let mut count = 0;
    let payload = &bytes[payload_start..];
    // Check each 4-byte float
    for chunk in payload.chunks_exact(4) {
        let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let f = f32::from_bits(bits);
        if f.is_nan() {
            count += 1;
        }
    }
    count
}

fn count_inf_values(bytes: &[u8]) -> usize {
    let payload_start = 64.min(bytes.len());
    let mut count = 0;
    let payload = &bytes[payload_start..];
    for chunk in payload.chunks_exact(4) {
        let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let f = f32::from_bits(bits);
        if f.is_infinite() {
            count += 1;
        }
    }
    count
}

fn compute_checksum(bytes: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5; // FNV-1a offset basis
    for &byte in bytes {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193); // FNV prime
    }
    hash
}

fn inject_nan_at(bytes: &mut [u8], offset: usize) {
    // IEEE 754 NaN: exponent all 1s, non-zero mantissa
    let nan_bits: u32 = 0x7FC0_0000;
    let nan_bytes = nan_bits.to_le_bytes();
    if offset + 4 <= bytes.len() {
        bytes[offset..offset + 4].copy_from_slice(&nan_bytes);
    }
}

#[cfg(test)]
fn inject_inf_at(bytes: &mut [u8], offset: usize) {
    let inf_bits: u32 = 0x7F80_0000;
    let inf_bytes = inf_bits.to_le_bytes();
    if offset + 4 <= bytes.len() {
        bytes[offset..offset + 4].copy_from_slice(&inf_bytes);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        let bundle = make_valid_bundle();
        let result = validate_model(&bundle);
        assert!(
            result.all_passed(),
            "Valid model should pass all checks, but failed: {:?}",
            result
                .checks
                .iter()
                .filter(|c| c.status == CheckStatus::Fail)
                .collect::<Vec<_>>()
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

    #[test]
    fn test_all_passed_true() {
        let mut r = ValidationResult::new();
        r.add("a", CheckStatus::Pass, "ok");
        r.add("b", CheckStatus::Warn, "ok");
        assert!(r.all_passed());
    }

    #[test]
    fn test_all_passed_false() {
        let mut r = ValidationResult::new();
        r.add("a", CheckStatus::Pass, "ok");
        r.add("b", CheckStatus::Fail, "bad");
        assert!(!r.all_passed());
    }

    #[test]
    fn test_checksum_nonzero_for_real_data() {
        let bundle = make_valid_bundle();
        let cs = compute_checksum(&bundle);
        assert_ne!(cs, 0);
    }

    #[test]
    fn test_count_nan_clean_data() {
        let bundle = make_valid_bundle();
        assert_eq!(count_nan_values(&bundle), 0);
    }

    #[test]
    fn test_count_nan_with_injected() {
        let mut bundle = make_valid_bundle();
        inject_nan_at(&mut bundle, 64);
        assert!(count_nan_values(&bundle) >= 1);
    }
}
