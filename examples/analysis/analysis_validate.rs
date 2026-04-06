//! # APR Model Validation
//!
//! CLI equivalent: `apr validate model.apr`
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

fn validate_model(bytes: &[u8]) -> ValidationResult {
    let mut result = ValidationResult::new();
    check_magic_bytes(bytes, &mut result);
    check_minimum_size(bytes, &mut result);
    check_version(bytes, &mut result);
    check_metadata_present(bytes, &mut result);
    check_tensors_nonempty(bytes, &mut result);
    check_no_nan(bytes, &mut result);
    check_no_inf(bytes, &mut result);
    check_compression_valid(bytes, &mut result);
    check_size_consistent(bytes, &mut result);
    check_checksum(bytes, &mut result);
    result
}

fn check_magic_bytes(bytes: &[u8], result: &mut ValidationResult) {
    let (status, detail) = if bytes.len() >= 4 && &bytes[0..4] == b"APR2" {
        (CheckStatus::Pass, "Valid APR2 magic bytes".into())
    } else if bytes.len() >= 4 {
        (
            CheckStatus::Fail,
            format!("Invalid magic: {:?}", &bytes[0..4]),
        )
    } else {
        (CheckStatus::Fail, "File too small for magic bytes".into())
    };
    result.add("magic_bytes", status, &detail);
}

fn check_minimum_size(bytes: &[u8], result: &mut ValidationResult) {
    let (status, op) = if bytes.len() >= 64 {
        (CheckStatus::Pass, ">=")
    } else {
        (CheckStatus::Fail, "<")
    };
    result.add(
        "minimum_size",
        status,
        &format!("{} bytes ({op} 64 byte minimum)", bytes.len()),
    );
}

fn check_version(bytes: &[u8], result: &mut ValidationResult) {
    if bytes.len() <= 4 {
        result.add(
            "format_version",
            CheckStatus::Fail,
            "File too small for version byte",
        );
        return;
    }
    let version = bytes[4];
    let (status, detail) = match version {
        0 | 2 => (CheckStatus::Pass, format!("Version byte: {version}")),
        1 => (
            CheckStatus::Warn,
            "Legacy v1 format detected; consider upgrading".into(),
        ),
        _ => (CheckStatus::Fail, format!("Unknown version: {version}")),
    };
    result.add("format_version", status, &detail);
}

fn check_metadata_present(bytes: &[u8], result: &mut ValidationResult) {
    let has_name = bytes.windows(5).any(|w| w == b"name=");
    let has_desc = bytes.windows(12).any(|w| w == b"description=");
    let (status, detail) = if has_name {
        (CheckStatus::Pass, "Model name found in metadata")
    } else if has_desc {
        (CheckStatus::Warn, "Name missing but description present")
    } else {
        (CheckStatus::Warn, "No standard metadata fields found")
    };
    result.add("metadata_present", status, detail);
}

fn check_tensors_nonempty(bytes: &[u8], result: &mut ValidationResult) {
    let header_size = 64;
    let (status, detail) = if bytes.len() > header_size + 16 {
        let payload_size = bytes.len() - header_size;
        (
            CheckStatus::Pass,
            format!("Payload size: {payload_size} bytes"),
        )
    } else if bytes.len() > header_size {
        (
            CheckStatus::Warn,
            "Payload very small; model may be trivial".into(),
        )
    } else {
        (CheckStatus::Fail, "No tensor payload detected".into())
    };
    result.add("tensors_nonempty", status, &detail);
}

fn check_no_nan(bytes: &[u8], result: &mut ValidationResult) {
    let n = count_special_values(bytes, f32::is_nan);
    let (s, d) = if n == 0 {
        (CheckStatus::Pass, "No NaN values detected".into())
    } else {
        (
            CheckStatus::Fail,
            format!("{n} NaN value(s) detected in tensor data"),
        )
    };
    result.add("no_nan", s, &d);
}

fn check_no_inf(bytes: &[u8], result: &mut ValidationResult) {
    let n = count_special_values(bytes, f32::is_infinite);
    let (s, d) = if n == 0 {
        (CheckStatus::Pass, "No Inf values detected".into())
    } else {
        (
            CheckStatus::Warn,
            format!("{n} Inf value(s) detected in tensor data"),
        )
    };
    result.add("no_inf", s, &d);
}

fn check_compression_valid(bytes: &[u8], result: &mut ValidationResult) {
    let lz4_magic: [u8; 4] = [0x04, 0x22, 0x4D, 0x18];
    let zstd_magic: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];
    let has_lz4 = bytes.windows(4).any(|w| w == lz4_magic);
    let has_zstd = bytes.windows(4).any(|w| w == zstd_magic);
    let (status, detail) = if has_lz4 && has_zstd {
        (
            CheckStatus::Warn,
            "Multiple compression formats detected".into(),
        )
    } else if has_lz4 || has_zstd || bytes.len() < 256 {
        let m = if has_lz4 {
            "LZ4"
        } else if has_zstd {
            "Zstd"
        } else {
            "None/uncompressed"
        };
        (CheckStatus::Pass, format!("Compression: {m}"))
    } else {
        (
            CheckStatus::Pass,
            "No compression detected (raw payload)".into(),
        )
    };
    result.add("compression_valid", status, &detail);
}

fn check_size_consistent(bytes: &[u8], result: &mut ValidationResult) {
    let (status, detail) = if bytes.len() >= 64 && bytes.len() % 4 == 0 {
        (CheckStatus::Pass, "File size is 4-byte aligned")
    } else if bytes.len() >= 64 {
        (
            CheckStatus::Warn,
            "File size not 4-byte aligned; possible truncation",
        )
    } else {
        (CheckStatus::Fail, "File smaller than minimum header size")
    };
    result.add("size_consistent", status, detail);
}

fn check_checksum(bytes: &[u8], result: &mut ValidationResult) {
    let cs = compute_checksum(bytes);
    let (s, d) = if cs > 0 {
        (CheckStatus::Pass, format!("Checksum: 0x{cs:08X}"))
    } else {
        (
            CheckStatus::Warn,
            "Zero checksum; payload may be all zeros".into(),
        )
    };
    result.add("checksum", s, &d);
}

fn count_special_values(bytes: &[u8], pred: fn(f32) -> bool) -> usize {
    let start = 64.min(bytes.len());
    bytes[start..]
        .chunks_exact(4)
        .filter(|c| pred(f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))))
        .count()
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
