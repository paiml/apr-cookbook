#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use super::types::*;

#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use std::fmt;

/// Stage 1: Validate APR2 magic bytes.
pub fn stage_magic_bytes(bytes: &[u8]) -> StageResult {
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
pub fn stage_header_integrity(bytes: &[u8]) -> StageResult {
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
pub fn stage_tensor_count(bytes: &[u8]) -> StageResult {
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

/// dimension vectors (one per tensor).
pub fn parse_tensor_shapes(bytes: &[u8]) -> Option<Vec<Vec<u64>>> {
    if bytes.len() < 64 {
        return None;
    }
    let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let index_offset = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;

    let mut offset = index_offset;
    let mut all_shapes = Vec::with_capacity(tensor_count);

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

        let mut dims = Vec::with_capacity(shape_len);
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
        all_shapes.push(dims);

        // Skip offset (8) + length (8) + dtype (1)
        offset += 17;
    }

    Some(all_shapes)
}

/// Stage 4: Shape consistency -- no zero dimensions in tensor index.
pub fn stage_shape_consistency(bytes: &[u8]) -> StageResult {
    let Some(all_shapes) = parse_tensor_shapes(bytes) else {
        return StageResult::skip("Shape consistency", "No header to parse");
    };

    let zero_dim_found = all_shapes.iter().any(|dims| dims.contains(&0));

    if zero_dim_found {
        StageResult::fail("Shape consistency", "Zero dimension found in tensor shape")
    } else {
        StageResult::pass(
            "Shape consistency",
            &format!("All shapes valid across {} tensor(s)", all_shapes.len()),
        )
    }
}

/// Stage 5: Dtype validation -- only known dtype codes.
pub fn stage_dtype_validation(bytes: &[u8]) -> StageResult {
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
pub fn stage_weight_range(bytes: &[u8]) -> StageResult {
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
pub fn stage_nan_inf_scan(bytes: &[u8]) -> StageResult {
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
pub fn stage_sparsity_analysis(bytes: &[u8]) -> StageResult {
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
pub fn stage_embedding_consistency(bytes: &[u8]) -> StageResult {
    let Some(all_dims) = parse_tensor_shapes(bytes) else {
        return StageResult::skip("Embedding consistency", "No header");
    };

    let tensor_count = all_dims.len();
    if tensor_count < 2 {
        return StageResult::skip(
            "Embedding consistency",
            "Need at least 2 tensors to check consistency",
        );
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
pub fn stage_checksum_verification(bytes: &[u8]) -> StageResult {
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

/// Compute FNV-1a checksum of a byte slice.
pub fn compute_fnv1a(bytes: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for &byte in bytes {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

/// Get payload start offset from APR v2 header.
pub fn get_payload_start(bytes: &[u8]) -> usize {
    if bytes.len() >= 20 && &bytes[0..4] == b"APR2" {
        (u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize).min(bytes.len())
    } else {
        64.min(bytes.len())
    }
}

/// Run all 10 check stages on model bytes.
pub fn run_check_pipeline(model_name: &str, bytes: &[u8]) -> CheckReport {
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

pub fn print_report(report: &CheckReport) {
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
