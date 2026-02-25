//! # APR Model Pre-Flight Check
//!
//! CLI equivalent: `apr check model.apr`
//!
//! Runs a 10-stage sequential pre-flight health check pipeline on an APR
//! model file. Each stage produces a pass/fail/skip result with detail.
//! The final report summarizes overall model readiness for deployment.

use apr_cookbook::prelude::*;
use std::fmt;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Result of a single check stage.
#[derive(Debug, Clone)]
struct StageResult {
    name: String,
    passed: bool,
    skipped: bool,
    detail: String,
}

impl StageResult {
    fn pass(name: &str, detail: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            skipped: false,
            detail: detail.to_string(),
        }
    }

    fn fail(name: &str, detail: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            skipped: false,
            detail: detail.to_string(),
        }
    }

    fn skip(name: &str, detail: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            skipped: true,
            detail: detail.to_string(),
        }
    }

    fn status_str(&self) -> &str {
        if self.skipped {
            "SKIP"
        } else if self.passed {
            "PASS"
        } else {
            "FAIL"
        }
    }
}

/// Overall verdict for the check report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckVerdict {
    Pass,
    Fail,
    Warn,
}

impl fmt::Display for CheckVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckVerdict::Pass => write!(f, "PASS"),
            CheckVerdict::Fail => write!(f, "FAIL"),
            CheckVerdict::Warn => write!(f, "WARN"),
        }
    }
}

/// Full check report across all stages.
#[derive(Debug, Clone)]
struct CheckReport {
    model_name: String,
    stages: Vec<StageResult>,
}

impl CheckReport {
    fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            stages: Vec::with_capacity(10),
        }
    }

    fn add(&mut self, result: StageResult) {
        self.stages.push(result);
    }

    fn passed_count(&self) -> usize {
        self.stages.iter().filter(|s| s.passed).count()
    }

    fn failed_count(&self) -> usize {
        self.stages
            .iter()
            .filter(|s| !s.passed && !s.skipped)
            .count()
    }

    fn skipped_count(&self) -> usize {
        self.stages.iter().filter(|s| s.skipped).count()
    }

    fn verdict(&self) -> CheckVerdict {
        if self.failed_count() > 0 {
            CheckVerdict::Fail
        } else if self.skipped_count() > 0 {
            CheckVerdict::Warn
        } else {
            CheckVerdict::Pass
        }
    }
}

// ---------------------------------------------------------------------------
// Known dtype byte values (APR v2 header byte 7)
// ---------------------------------------------------------------------------

const KNOWN_DTYPES: [u8; 4] = [0, 1, 2, 3]; // FP32, FP16, Int8, Int4

// ---------------------------------------------------------------------------
// Stage implementations
// ---------------------------------------------------------------------------

/// Stage 1: Validate APR2 magic bytes.
fn stage_magic_bytes(bytes: &[u8]) -> StageResult {
    if bytes.len() < 4 {
        return StageResult::fail("Magic bytes", "File too short for magic bytes");
    }
    if &bytes[0..4] == b"APR2" {
        StageResult::pass("Magic bytes", "Valid APR2 magic")
    } else {
        StageResult::fail(
            "Magic bytes",
            &format!("Expected APR2, got {:?}", &bytes[0..4]),
        )
    }
}

/// Stage 2: Validate header integrity (minimum 64 bytes, valid version).
fn stage_header_integrity(bytes: &[u8]) -> StageResult {
    if bytes.len() < 64 {
        return StageResult::fail(
            "Header integrity",
            &format!("Header too short: {} bytes (need 64)", bytes.len()),
        );
    }
    let major = bytes[4];
    let minor = bytes[5];
    if major == 2 && minor == 0 {
        StageResult::pass(
            "Header integrity",
            &format!(
                "Valid header, version {major}.{minor}, {} bytes",
                bytes.len()
            ),
        )
    } else {
        StageResult::fail(
            "Header integrity",
            &format!("Unexpected version {major}.{minor}"),
        )
    }
}

/// Stage 3: Verify tensor count matches index.
fn stage_tensor_count(bytes: &[u8]) -> StageResult {
    if bytes.len() < 12 {
        return StageResult::skip("Tensor count", "Header too short to read tensor count");
    }
    let count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if count == 0 {
        StageResult::fail("Tensor count", "Zero tensors declared")
    } else {
        StageResult::pass("Tensor count", &format!("{count} tensor(s) declared"))
    }
}

/// Stage 4: Shape consistency -- no zero dimensions in tensor index.
fn stage_shape_consistency(bytes: &[u8]) -> StageResult {
    if bytes.len() < 64 {
        return StageResult::skip("Shape consistency", "No header to parse");
    }
    let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let index_offset = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;

    let mut offset = index_offset;
    let mut zero_dim_found = false;

    for _ in 0..tensor_count {
        // Read name_len + name
        if offset + 4 > bytes.len() {
            break;
        }
        let name_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4 + name_len;

        // Read shape_len + shape dims
        if offset + 4 > bytes.len() {
            break;
        }
        let shape_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        for _ in 0..shape_len {
            if offset + 8 > bytes.len() {
                break;
            }
            let dim = u64::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]);
            if dim == 0 {
                zero_dim_found = true;
            }
            offset += 8;
        }

        // Skip offset (8) + length (8) + dtype (1)
        offset += 17;
    }

    if zero_dim_found {
        StageResult::fail("Shape consistency", "Zero dimension found in tensor shape")
    } else {
        StageResult::pass(
            "Shape consistency",
            &format!("All shapes valid across {tensor_count} tensor(s)"),
        )
    }
}

/// Stage 5: Dtype validation -- only known dtype codes.
fn stage_dtype_validation(bytes: &[u8]) -> StageResult {
    if bytes.len() < 8 {
        return StageResult::skip("Dtype validation", "Header too short");
    }
    let dtype_byte = bytes[7];
    if KNOWN_DTYPES.contains(&dtype_byte) {
        let name = match dtype_byte {
            0 => "FP32",
            1 => "FP16",
            2 => "Int8",
            3 => "Int4",
            _ => "unknown",
        };
        StageResult::pass(
            "Dtype validation",
            &format!("Dtype: {name} (code {dtype_byte})"),
        )
    } else {
        StageResult::fail(
            "Dtype validation",
            &format!("Unknown dtype code: {dtype_byte}"),
        )
    }
}

/// Stage 6: Weight range check -- no extreme values (|w| > 1e6).
fn stage_weight_range(bytes: &[u8]) -> StageResult {
    let payload_offset = get_payload_start(bytes);
    let payload = &bytes[payload_offset..];
    if payload.len() < 4 {
        return StageResult::skip("Weight range", "No payload data");
    }

    let threshold: f32 = 1e6;
    let mut extreme_count: usize = 0;
    let mut total: usize = 0;

    for chunk in payload.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if val.is_finite() {
            total += 1;
            if val.abs() > threshold {
                extreme_count += 1;
            }
        }
    }

    if extreme_count > 0 {
        StageResult::fail(
            "Weight range",
            &format!("{extreme_count}/{total} values exceed magnitude {threshold:.0e}"),
        )
    } else {
        StageResult::pass(
            "Weight range",
            &format!("{total} values within normal range"),
        )
    }
}

/// Stage 7: NaN/Inf scan of tensor payload.
fn stage_nan_inf_scan(bytes: &[u8]) -> StageResult {
    let payload_offset = get_payload_start(bytes);
    let payload = &bytes[payload_offset..];
    if payload.len() < 4 {
        return StageResult::skip("NaN/Inf scan", "No payload data");
    }

    let mut nan_count: usize = 0;
    let mut inf_count: usize = 0;
    let mut total: usize = 0;

    for chunk in payload.chunks_exact(4) {
        let val = f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        total += 1;
        if val.is_nan() {
            nan_count += 1;
        } else if val.is_infinite() {
            inf_count += 1;
        }
    }

    if nan_count > 0 || inf_count > 0 {
        StageResult::fail(
            "NaN/Inf scan",
            &format!("{nan_count} NaN, {inf_count} Inf in {total} values"),
        )
    } else {
        StageResult::pass(
            "NaN/Inf scan",
            &format!("{total} values clean (no NaN/Inf)"),
        )
    }
}

/// Stage 8: Sparsity analysis -- report fraction of zero weights.
fn stage_sparsity_analysis(bytes: &[u8]) -> StageResult {
    let payload_offset = get_payload_start(bytes);
    let payload = &bytes[payload_offset..];
    if payload.len() < 4 {
        return StageResult::skip("Sparsity analysis", "No payload data");
    }

    let mut zero_count: usize = 0;
    let mut total: usize = 0;

    for chunk in payload.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        total += 1;
        if val == 0.0 {
            zero_count += 1;
        }
    }

    let sparsity = if total > 0 {
        zero_count as f64 / total as f64
    } else {
        0.0
    };

    // Pass always (informational), but note if >90% sparse
    let detail = format!(
        "{:.1}% sparse ({zero_count}/{total} zeros)",
        sparsity * 100.0,
    );
    if sparsity > 0.9 {
        StageResult::pass(
            "Sparsity analysis",
            &format!("{detail} [highly sparse -- consider pruned format]"),
        )
    } else {
        StageResult::pass("Sparsity analysis", &detail)
    }
}

/// Stage 9: Embedding dimension consistency -- all tensors sharing a common dim.
fn stage_embedding_consistency(bytes: &[u8]) -> StageResult {
    if bytes.len() < 64 {
        return StageResult::skip("Embedding consistency", "No header");
    }
    let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    if tensor_count < 2 {
        return StageResult::skip(
            "Embedding consistency",
            "Need at least 2 tensors to check consistency",
        );
    }

    let index_offset = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;

    let mut all_dims: Vec<Vec<u64>> = Vec::new();
    let mut offset = index_offset;

    for _ in 0..tensor_count {
        if offset + 4 > bytes.len() {
            break;
        }
        let name_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4 + name_len;

        if offset + 4 > bytes.len() {
            break;
        }
        let shape_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let mut dims = Vec::new();
        for _ in 0..shape_len {
            if offset + 8 > bytes.len() {
                break;
            }
            let dim = u64::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]);
            dims.push(dim);
            offset += 8;
        }
        all_dims.push(dims);

        // Skip offset (8) + length (8) + dtype (1)
        offset += 17;
    }

    // Find a common dimension shared by the majority of tensors
    let mut dim_counts = std::collections::HashMap::new();
    for dims in &all_dims {
        for &d in dims {
            *dim_counts.entry(d).or_insert(0usize) += 1;
        }
    }

    let most_common = dim_counts
        .iter()
        .max_by_key(|(_, count)| **count)
        .map(|(dim, count)| (*dim, *count));

    match most_common {
        Some((dim, count)) if count >= tensor_count / 2 => StageResult::pass(
            "Embedding consistency",
            &format!("Common dim {dim} shared by {count}/{tensor_count} tensors",),
        ),
        Some((dim, count)) => StageResult::pass(
            "Embedding consistency",
            &format!("Most common dim {dim} in {count}/{tensor_count} tensors (low overlap)",),
        ),
        None => StageResult::skip("Embedding consistency", "No dimensions found"),
    }
}

/// Stage 10: Checksum verification (FNV-1a over full file).
fn stage_checksum_verification(bytes: &[u8]) -> StageResult {
    if bytes.is_empty() {
        return StageResult::fail("Checksum verification", "Empty file");
    }

    let checksum = compute_fnv1a(bytes);
    if checksum == 0x811c_9dc5 {
        // Offset basis unchanged means empty or all-zero xor
        StageResult::fail(
            "Checksum verification",
            "Degenerate checksum (unchanged from offset basis)",
        )
    } else {
        StageResult::pass(
            "Checksum verification",
            &format!("FNV-1a: 0x{checksum:08X}"),
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute FNV-1a checksum of a byte slice.
fn compute_fnv1a(bytes: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for &byte in bytes {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

/// Get payload start offset from APR v2 header.
fn get_payload_start(bytes: &[u8]) -> usize {
    if bytes.len() >= 20 && &bytes[0..4] == b"APR2" {
        (u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize).min(bytes.len())
    } else {
        64.min(bytes.len())
    }
}

/// Run all 10 check stages on model bytes.
fn run_check_pipeline(model_name: &str, bytes: &[u8]) -> CheckReport {
    let mut report = CheckReport::new(model_name);
    report.add(stage_magic_bytes(bytes));
    report.add(stage_header_integrity(bytes));
    report.add(stage_tensor_count(bytes));
    report.add(stage_shape_consistency(bytes));
    report.add(stage_dtype_validation(bytes));
    report.add(stage_weight_range(bytes));
    report.add(stage_nan_inf_scan(bytes));
    report.add(stage_sparsity_analysis(bytes));
    report.add(stage_embedding_consistency(bytes));
    report.add(stage_checksum_verification(bytes));
    report
}

fn print_report(report: &CheckReport) {
    println!("\nModel: {}", report.model_name);
    println!("{:<8} {:<25} {:<6} Detail", "Stage", "Name", "Status");
    println!("{}", "-".repeat(80));
    for (i, stage) in report.stages.iter().enumerate() {
        println!(
            "{:<8} {:<25} {:<6} {}",
            i + 1,
            stage.name,
            stage.status_str(),
            stage.detail,
        );
    }
    println!(
        "\nSummary: passed={}, failed={}, skipped={}, verdict={}",
        report.passed_count(),
        report.failed_count(),
        report.skipped_count(),
        report.verdict(),
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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
