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
use apr_cookbook::prelude::*;
use rand::Rng;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// A contiguous slice extracted from tensor data.
#[derive(Debug, Clone)]
pub struct TensorSlice {
    pub tensor_name: String,
    pub offset: usize,
    pub length: usize,
    pub values: Vec<f32>,
    pub raw_bytes: Vec<u8>,
}

/// Describes how to slice a tensor.
#[derive(Debug, Clone, PartialEq)]
pub enum SliceOp {
    // Elements in `[start, end)`.
    Range(usize, usize),
    // All elements in row `r` of a 2-D tensor.
    Row(usize),
    // All elements in column `c` of a 2-D tensor.
    Column(usize),
    // Every `n`-th element from the flat buffer.
    Stride(usize),
}

/// Descriptive statistics for a slice.
#[derive(Debug, Clone)]
pub struct SliceStats {
    pub mean: f64,
    pub min: f32,
    pub max: f32,
    pub sum: f64,
}

/// A completed slice operation with its result and statistics.
#[derive(Debug, Clone)]
pub struct SliceResult {
    pub op: SliceOp,
    pub slice: TensorSlice,
    pub stats: SliceStats,
}

// ---------------------------------------------------------------------------
// Slice helpers
// ---------------------------------------------------------------------------

impl std::fmt::Display for SliceOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Range(a, b) => write!(f, "Range({a}..{b})"),
            Self::Row(r) => write!(f, "Row({r})"),
            Self::Column(c) => write!(f, "Column({c})"),
            Self::Stride(s) => write!(f, "Stride(every {s})"),
        }
    }
}

/// Compute descriptive statistics for a value slice.
pub fn compute_stats(values: &[f32]) -> SliceStats {
    if values.is_empty() {
        return SliceStats {
            mean: 0.0,
            min: 0.0,
            max: 0.0,
            sum: 0.0,
        };
    }
    let sum: f64 = values.iter().map(|v| f64::from(*v)).sum();
    let mean = sum / values.len() as f64;
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    SliceStats {
        mean,
        min,
        max,
        sum,
    }
}

/// Build a `TensorSlice` from selected indices into the flat data.
pub fn build_slice(name: &str, data: &[f32], indices: &[usize]) -> TensorSlice {
    let values: Vec<f32> = indices
        .iter()
        .filter(|&&i| i < data.len())
        .map(|&i| data[i])
        .collect();

    let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let offset = indices.first().copied().unwrap_or(0);

    TensorSlice {
        tensor_name: name.to_string(),
        offset,
        length: values.len(),
        values,
        raw_bytes,
    }
}

// Execute a single slice operation against flat tensor data.
//
// For `Row` and `Column` the caller must supply `cols` (number of columns
/// in the logical 2-D layout).  The value is ignored for `Range`/`Stride`.
pub fn execute_slice(
    name: &str,
    data: &[f32],
    op: &SliceOp,
    cols: usize,
) -> std::result::Result<SliceResult, String> {
    let indices: Vec<usize> = match op {
        SliceOp::Range(start, end) => {
            if *start >= *end {
                return Err(format!("invalid range: start ({start}) >= end ({end})"));
            }
            if *start >= data.len() {
                return Err(format!(
                    "range start ({start}) out of bounds (len={})",
                    data.len()
                ));
            }
            (*start..(*end).min(data.len())).collect()
        }
        SliceOp::Row(r) => {
            if cols == 0 {
                return Err("column count must be > 0 for Row slice".to_string());
            }
            let row_start = r * cols;
            if row_start >= data.len() {
                return Err(format!(
                    "row {r} out of bounds (data len={}, cols={cols})",
                    data.len()
                ));
            }
            let row_end = (row_start + cols).min(data.len());
            (row_start..row_end).collect()
        }
        SliceOp::Column(c) => {
            if cols == 0 {
                return Err("column count must be > 0 for Column slice".to_string());
            }
            if *c >= cols {
                return Err(format!("column {c} out of bounds (cols={cols})"));
            }
            (*c..data.len()).step_by(cols).collect()
        }
        SliceOp::Stride(s) => {
            if *s == 0 {
                return Err("stride must be > 0".to_string());
            }
            (0..data.len()).step_by(*s).collect()
        }
    };

    let slice = build_slice(name, data, &indices);
    let stats = compute_stats(&slice.values);

    Ok(SliceResult {
        op: op.clone(),
        slice,
        stats,
    })
}

// ---------------------------------------------------------------------------
// Hex and dtype helpers
// ---------------------------------------------------------------------------

/// Render bytes as a space-separated hex string (max `limit` bytes shown).
pub fn bytes_to_hex(data: &[u8], limit: usize) -> String {
    let show = data.len().min(limit);
    let hex: String = data[..show]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ");
    if show < data.len() {
        format!("{hex} ... ({} more bytes)", data.len() - show)
    } else {
        hex
    }
}

/// Simulate f32 -> f16 -> f32 round-trip and return the converted values.
pub fn f32_to_f16_roundtrip(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&v| {
            let bits = f16_from_f32(v);
            f16_to_f32(bits)
        })
        .collect()
}

/// Convert an f32 to its IEEE 754 half-precision bit pattern.
pub fn f16_from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 0xFF {
        // Inf / NaN
        return sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 };
    }

    let new_exp = exponent - 127 + 15;

    if new_exp >= 0x1F {
        // Overflow -> Inf
        return sign | 0x7C00;
    }
    if new_exp <= 0 {
        // Underflow -> zero (flush to zero)
        return sign;
    }

    sign | ((new_exp as u16) << 10) | ((mantissa >> 13) as u16)
}

/// Convert an IEEE 754 half-precision bit pattern back to f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1F);
    let mantissa = u32::from(bits & 0x03FF);

    if exponent == 0x1F {
        // Inf / NaN
        let f_bits = sign | 0x7F80_0000 | (mantissa << 13);
        return f32::from_bits(f_bits);
    }
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal
        let f_bits = sign | (mantissa << 13);
        return f32::from_bits(f_bits);
    }

    let f_bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    f32::from_bits(f_bits)
}

/// Compute max absolute error between two equally-sized slices.
pub fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Compute mean absolute error between two equally-sized slices.
pub fn mean_abs_error(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let total: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| f64::from((x - y).abs()))
        .sum();
    total / a.len() as f64
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

pub fn print_slice_summary(result: &SliceResult) {
    let s = &result.slice;
    let byte_start = s.offset * 4;
    let byte_end = byte_start + s.length * 4;

    println!("  Operation:   {}", result.op);
    println!("  Tensor:      {}", s.tensor_name);
    println!("  Offset:      {} (element index)", s.offset);
    println!("  Length:      {} elements", s.length);
    println!(
        "  Byte range:  0x{byte_start:04x}..0x{byte_end:04x} ({} bytes)",
        s.raw_bytes.len()
    );

    let preview: Vec<String> = s.values.iter().take(8).map(|v| format!("{v:.4}")).collect();
    let ellipsis = if s.values.len() > 8 { " ..." } else { "" };
    println!("  Values:      [{}{}]", preview.join(", "), ellipsis);

    println!("  Hex:         {}", bytes_to_hex(&s.raw_bytes, 32));

    println!(
        "  Stats:       mean={:.6}, min={:.6}, max={:.6}, sum={:.4}",
        result.stats.mean, result.stats.min, result.stats.max, result.stats.sum,
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
