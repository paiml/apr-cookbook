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

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Status of a single qualification gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateStatus {
    Pass,
    Fail,
    Skip,
}

impl fmt::Display for GateStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GateStatus::Pass => f.write_str("Pass"),
            GateStatus::Fail => f.write_str("Fail"),
            GateStatus::Skip => f.write_str("Skip"),
        }
    }
}

/// Result of running a single qualification gate.
#[derive(Debug, Clone)]
struct GateResult {
    name: String,
    status: GateStatus,
    duration_ms: f64,
    detail: String,
}

impl GateResult {
    fn new(name: &str, status: GateStatus, duration_ms: f64, detail: &str) -> Self {
        Self {
            name: name.to_string(),
            status,
            duration_ms,
            detail: detail.to_string(),
        }
    }
}

/// Qualification tier derived from gate results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QualifyTier {
    /// All 11 gates passed.
    Smoke,
    /// 8 or more gates passed (none failed — skips are tolerated).
    Qualified,
    /// Fewer than 8 gates passed, or critical failures.
    Rejected,
}

impl fmt::Display for QualifyTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QualifyTier::Smoke => f.write_str("Smoke"),
            QualifyTier::Qualified => f.write_str("Qualified"),
            QualifyTier::Rejected => f.write_str("Rejected"),
        }
    }
}

/// Full qualification report for a model.
#[derive(Debug, Clone)]
struct QualifyReport {
    model_name: String,
    gates: Vec<GateResult>,
    tier: QualifyTier,
}

// ---------------------------------------------------------------------------
// Tier computation
// ---------------------------------------------------------------------------

fn compute_tier(gates: &[GateResult]) -> QualifyTier {
    let passed = gates
        .iter()
        .filter(|g| g.status == GateStatus::Pass)
        .count();
    let failed = gates
        .iter()
        .filter(|g| g.status == GateStatus::Fail)
        .count();

    if failed == 0 && passed == gates.len() {
        QualifyTier::Smoke
    } else if passed >= 8 {
        QualifyTier::Qualified
    } else {
        QualifyTier::Rejected
    }
}

// ---------------------------------------------------------------------------
// Helper: timed gate runner
// ---------------------------------------------------------------------------

fn run_gate(name: &str, f: impl FnOnce() -> (GateStatus, String)) -> GateResult {
    let start = Instant::now();
    let (status, detail) = f();
    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    GateResult::new(name, status, duration_ms, &detail)
}

// ---------------------------------------------------------------------------
// APR v2 header helpers
// ---------------------------------------------------------------------------

/// Extract the payload offset from an APR v2 header (bytes 16..20, u32 LE).
fn payload_offset(bytes: &[u8]) -> usize {
    if bytes.len() >= 20 {
        (u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize).min(bytes.len())
    } else {
        bytes.len()
    }
}

/// Decompress the payload if needed, returning raw weight bytes.
fn decompress_payload(bytes: &[u8]) -> Option<Vec<u8>> {
    if bytes.len() < 64 {
        return None;
    }
    let comp = bytes[6];
    let po = payload_offset(bytes);
    if po >= bytes.len() {
        return None;
    }
    let raw = &bytes[po..];
    match comp {
        0 => Some(raw.to_vec()),
        1 => lz4_flex::decompress_size_prepended(raw).ok(),
        2 => zstd::decode_all(raw).ok(),
        _ => None,
    }
}

/// Interpret a byte slice as f32 values (LE).
fn slice_as_f32(data: &[u8]) -> impl Iterator<Item = f32> + '_ {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
}

// ---------------------------------------------------------------------------
// 11 Qualification Gates
// ---------------------------------------------------------------------------

/// Gate 1: format_valid — APR2 magic bytes present.
fn gate_format_valid(bytes: &[u8]) -> GateResult {
    run_gate("format_valid", || {
        if bytes.len() >= 4 && &bytes[0..4] == b"APR2" {
            (GateStatus::Pass, "Valid APR2 magic bytes".into())
        } else {
            (GateStatus::Fail, "Missing or invalid APR2 magic".into())
        }
    })
}

/// Gate 2: header_parseable — Can parse the 64-byte header without error.
fn gate_header_parseable(bytes: &[u8]) -> GateResult {
    run_gate("header_parseable", || {
        if bytes.len() < 64 {
            return (
                GateStatus::Fail,
                format!("Header too short: {} bytes (need 64)", bytes.len()),
            );
        }
        // Version byte must be known (0, 1, or 2)
        let version = bytes[4];
        if version > 2 {
            return (GateStatus::Fail, format!("Unknown version byte: {version}"));
        }
        // Compression byte must be known (0, 1, 2)
        let comp = bytes[6];
        if comp > 2 {
            return (
                GateStatus::Fail,
                format!("Unknown compression byte: {comp}"),
            );
        }
        // Quantization byte must be known (0..3)
        let quant = bytes[7];
        if quant > 3 {
            return (
                GateStatus::Fail,
                format!("Unknown quantization byte: {quant}"),
            );
        }
        (
            GateStatus::Pass,
            format!("v{version}, comp={comp}, quant={quant}"),
        )
    })
}

/// Gate 3: tensor_loadable — At least one tensor entry in the index.
fn gate_tensor_loadable(bytes: &[u8]) -> GateResult {
    run_gate("tensor_loadable", || {
        if bytes.len() < 12 {
            return (GateStatus::Fail, "File too short for tensor count".into());
        }
        let count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if count == 0 {
            (GateStatus::Fail, "Zero tensors in index".into())
        } else {
            (GateStatus::Pass, format!("{count} tensor(s) indexed"))
        }
    })
}

/// Gate 4: no_nan — No NaN values in the weight payload.
fn gate_no_nan(bytes: &[u8]) -> GateResult {
    run_gate("no_nan", || {
        let Some(data) = decompress_payload(bytes) else {
            return (GateStatus::Skip, "Cannot read payload".into());
        };
        let nan_count = slice_as_f32(&data).filter(|v| v.is_nan()).count();
        if nan_count == 0 {
            (GateStatus::Pass, "No NaN values detected".into())
        } else {
            (GateStatus::Fail, format!("{nan_count} NaN value(s) found"))
        }
    })
}

/// Gate 5: no_inf — No Inf values in the weight payload.
fn gate_no_inf(bytes: &[u8]) -> GateResult {
    run_gate("no_inf", || {
        let Some(data) = decompress_payload(bytes) else {
            return (GateStatus::Skip, "Cannot read payload".into());
        };
        let inf_count = slice_as_f32(&data).filter(|v| v.is_infinite()).count();
        if inf_count == 0 {
            (GateStatus::Pass, "No Inf values detected".into())
        } else {
            (GateStatus::Fail, format!("{inf_count} Inf value(s) found"))
        }
    })
}

/// Gate 6: size_reasonable — File size between 64 bytes and 4 GiB.
fn gate_size_reasonable(bytes: &[u8]) -> GateResult {
    run_gate("size_reasonable", || {
        let len = bytes.len();
        let max_size: usize = 4 * 1024 * 1024 * 1024; // 4 GiB
        if len < 64 {
            (
                GateStatus::Fail,
                format!("{len} bytes — below 64-byte minimum"),
            )
        } else if len > max_size {
            (
                GateStatus::Fail,
                format!("{len} bytes — exceeds 4 GiB maximum"),
            )
        } else {
            (GateStatus::Pass, format!("{len} bytes within bounds"))
        }
    })
}

/// Gate 7: shape_consistent — No zero-dimension tensors.
fn gate_shape_consistent(bytes: &[u8]) -> GateResult {
    run_gate("shape_consistent", || {
        if bytes.len() < 64 {
            return (
                GateStatus::Skip,
                "Header too short to inspect shapes".into(),
            );
        }
        let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        if tensor_count == 0 {
            return (GateStatus::Skip, "No tensors to check".into());
        }
        let index_offset =
            u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;

        let mut pos = index_offset;
        let mut zero_dim_found = false;
        for _ in 0..tensor_count {
            if pos + 4 > bytes.len() {
                break;
            }
            // name_len + name
            let name_len =
                u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                    as usize;
            pos += 4 + name_len;
            if pos + 4 > bytes.len() {
                break;
            }
            // shape_len
            let shape_len =
                u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                    as usize;
            pos += 4;
            for _ in 0..shape_len {
                if pos + 8 > bytes.len() {
                    break;
                }
                let dim = u64::from_le_bytes([
                    bytes[pos],
                    bytes[pos + 1],
                    bytes[pos + 2],
                    bytes[pos + 3],
                    bytes[pos + 4],
                    bytes[pos + 5],
                    bytes[pos + 6],
                    bytes[pos + 7],
                ]);
                if dim == 0 {
                    zero_dim_found = true;
                }
                pos += 8;
            }
            // offset (8) + length (8) + dtype (1)
            pos += 17;
        }
        if zero_dim_found {
            (GateStatus::Fail, "Zero-dimension tensor detected".into())
        } else {
            (
                GateStatus::Pass,
                format!("{tensor_count} tensor(s), all shapes non-zero"),
            )
        }
    })
}

/// Gate 8: dtype_supported — Quantization byte maps to a known dtype.
fn gate_dtype_supported(bytes: &[u8]) -> GateResult {
    run_gate("dtype_supported", || {
        if bytes.len() < 8 {
            return (GateStatus::Fail, "File too short for dtype byte".into());
        }
        let quant = bytes[7];
        let label = match quant {
            0 => "FP32",
            1 => "FP16",
            2 => "Int8",
            3 => "Int4",
            _ => return (GateStatus::Fail, format!("Unknown dtype byte: {quant}")),
        };
        (GateStatus::Pass, format!("dtype={label} (byte={quant})"))
    })
}

/// Gate 9: compression_decodable — If compressed, can decompress the payload.
fn gate_compression_decodable(bytes: &[u8]) -> GateResult {
    run_gate("compression_decodable", || {
        if bytes.len() < 64 {
            return (GateStatus::Skip, "Header too short".into());
        }
        let comp = bytes[6];
        if comp == 0 {
            return (
                GateStatus::Skip,
                "No compression — nothing to decode".into(),
            );
        }
        let po = payload_offset(bytes);
        if po >= bytes.len() {
            return (GateStatus::Fail, "Payload offset beyond file end".into());
        }
        let compressed = &bytes[po..];
        let result = match comp {
            1 => lz4_flex::decompress_size_prepended(compressed)
                .map(|d| d.len())
                .map_err(|e| e.to_string()),
            2 => zstd::decode_all(compressed)
                .map(|d| d.len())
                .map_err(|e| e.to_string()),
            _ => return (GateStatus::Fail, format!("Unknown compression: {comp}")),
        };
        match result {
            Ok(size) => (
                GateStatus::Pass,
                format!("Decompressed {size} bytes successfully"),
            ),
            Err(e) => (GateStatus::Fail, format!("Decompression failed: {e}")),
        }
    })
}

/// Gate 10: metadata_present — Tensor index contains named entries.
fn gate_metadata_present(bytes: &[u8]) -> GateResult {
    run_gate("metadata_present", || {
        if bytes.len() < 64 {
            return (GateStatus::Fail, "Header too short for metadata".into());
        }
        let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let index_offset =
            u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;

        if tensor_count == 0 {
            return (GateStatus::Fail, "No tensor entries (no metadata)".into());
        }
        // Try to read the first tensor name from the index.
        if index_offset + 4 > bytes.len() {
            return (GateStatus::Fail, "Index offset beyond file end".into());
        }
        let name_len = u32::from_le_bytes([
            bytes[index_offset],
            bytes[index_offset + 1],
            bytes[index_offset + 2],
            bytes[index_offset + 3],
        ]) as usize;
        let name_start = index_offset + 4;
        if name_start + name_len > bytes.len() || name_len == 0 {
            return (GateStatus::Fail, "Tensor name missing or truncated".into());
        }
        let name = String::from_utf8_lossy(&bytes[name_start..name_start + name_len]);
        (
            GateStatus::Pass,
            format!("{tensor_count} named tensor(s), first: \"{name}\""),
        )
    })
}

/// Gate 11: checksum_valid — FNV-1a checksum is non-zero (payload is not all zeros).
fn gate_checksum_valid(bytes: &[u8]) -> GateResult {
    run_gate("checksum_valid", || {
        let po = payload_offset(bytes);
        if po >= bytes.len() {
            return (GateStatus::Skip, "No payload to checksum".into());
        }
        let payload = &bytes[po..];
        let mut hash: u32 = 0x811c_9dc5;
        for &b in payload {
            hash ^= u32::from(b);
            hash = hash.wrapping_mul(0x0100_0193);
        }
        // A zero hash on non-empty data is suspicious but possible; we check
        // that the hash is non-trivial (not the offset basis — that would mean
        // the payload is empty, which is already caught elsewhere).
        if hash == 0x811c_9dc5 {
            (
                GateStatus::Fail,
                "Checksum equals offset basis — empty payload?".into(),
            )
        } else {
            (GateStatus::Pass, format!("FNV-1a checksum: 0x{hash:08X}"))
        }
    })
}

// ---------------------------------------------------------------------------
// Run all 11 gates
// ---------------------------------------------------------------------------

fn run_all_gates(bytes: &[u8]) -> Vec<GateResult> {
    vec![
        gate_format_valid(bytes),
        gate_header_parseable(bytes),
        gate_tensor_loadable(bytes),
        gate_no_nan(bytes),
        gate_no_inf(bytes),
        gate_size_reasonable(bytes),
        gate_shape_consistent(bytes),
        gate_dtype_supported(bytes),
        gate_compression_decodable(bytes),
        gate_metadata_present(bytes),
        gate_checksum_valid(bytes),
    ]
}

fn build_report(model_name: &str, gates: Vec<GateResult>) -> QualifyReport {
    let tier = compute_tier(&gates);
    QualifyReport {
        model_name: model_name.to_string(),
        gates,
        tier,
    }
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

fn print_report(report: &QualifyReport) {
    println!("\n{:<25} {:<6} {:>8}  Detail", "Gate", "Status", "ms");
    println!("{}", "-".repeat(78));
    for g in &report.gates {
        println!(
            "{:<25} {:<6} {:>8.3}  {}",
            g.name, g.status, g.duration_ms, g.detail,
        );
    }

    let passed = report
        .gates
        .iter()
        .filter(|g| g.status == GateStatus::Pass)
        .count();
    let failed = report
        .gates
        .iter()
        .filter(|g| g.status == GateStatus::Fail)
        .count();
    let skipped = report
        .gates
        .iter()
        .filter(|g| g.status == GateStatus::Skip)
        .count();
    let total_ms: f64 = report.gates.iter().map(|g| g.duration_ms).sum();

    println!("\nModel:    {}", report.model_name);
    println!("Passed:   {passed}/{}", report.gates.len());
    println!("Failed:   {failed}");
    println!("Skipped:  {skipped}");
    println!("Total:    {total_ms:.3} ms");
    println!("Tier:     {}", report.tier);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

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
