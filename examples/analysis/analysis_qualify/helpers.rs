//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
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
use std::time::Instant;

pub fn compute_tier(gates: &[GateResult]) -> QualifyTier {
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

pub fn run_gate(name: &str, f: impl FnOnce() -> (GateStatus, String)) -> GateResult {
    let start = Instant::now();
    let (status, detail) = f();
    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    GateResult::new(name, status, duration_ms, &detail)
}

/// Extract the payload offset from an APR v2 header (bytes 16..20, u32 LE).
pub fn payload_offset(bytes: &[u8]) -> usize {
    if bytes.len() >= 20 {
        (u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize).min(bytes.len())
    } else {
        bytes.len()
    }
}

/// Decompress the payload if needed, returning raw weight bytes.
pub fn decompress_payload(bytes: &[u8]) -> Option<Vec<u8>> {
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
pub fn slice_as_f32(data: &[u8]) -> impl Iterator<Item = f32> + '_ {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
}

/// Gate 1: format_valid — APR2 magic bytes present.
pub fn gate_format_valid(bytes: &[u8]) -> GateResult {
    run_gate("format_valid", || {
        if bytes.len() >= 4 && &bytes[0..4] == b"APR2" {
            (GateStatus::Pass, "Valid APR2 magic bytes".into())
        } else {
            (GateStatus::Fail, "Missing or invalid APR2 magic".into())
        }
    })
}

/// Gate 2: header_parseable — Can parse the 64-byte header without error.
pub fn gate_header_parseable(bytes: &[u8]) -> GateResult {
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
pub fn gate_tensor_loadable(bytes: &[u8]) -> GateResult {
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
pub fn gate_no_nan(bytes: &[u8]) -> GateResult {
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
pub fn gate_no_inf(bytes: &[u8]) -> GateResult {
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
pub fn gate_size_reasonable(bytes: &[u8]) -> GateResult {
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
pub fn gate_shape_consistent(bytes: &[u8]) -> GateResult {
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
pub fn gate_dtype_supported(bytes: &[u8]) -> GateResult {
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
pub fn gate_compression_decodable(bytes: &[u8]) -> GateResult {
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
pub fn gate_metadata_present(bytes: &[u8]) -> GateResult {
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
pub fn gate_checksum_valid(bytes: &[u8]) -> GateResult {
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

pub fn run_all_gates(bytes: &[u8]) -> Vec<GateResult> {
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

pub fn build_report(model_name: &str, gates: Vec<GateResult>) -> QualifyReport {
    let tier = compute_tier(&gates);
    QualifyReport {
        model_name: model_name.to_string(),
        gates,
        tier,
    }
}

pub fn print_report(report: &QualifyReport) {
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
