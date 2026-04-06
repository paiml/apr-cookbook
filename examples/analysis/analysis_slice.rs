//! # Tensor Slice Extraction and Decoding
//!
//! CLI equivalent: `apr tensors model.apr --slice weights --range 10..20`
//!
//! Extract and decode a range of elements from tensor data. Demonstrates
//! index-range slicing, row/column extraction, strided access, hex dumping,
//! dtype conversion with precision-loss analysis, and per-slice statistics.
//!
//! ## What this demonstrates
//! - Index-range, row, column, and strided tensor slicing
//! - Raw byte extraction and hex representation
//! - f32 to f16 conversion with precision loss measurement
//! - Per-slice descriptive statistics (mean, min, max, sum)

use apr_cookbook::prelude::*;
use rand::Rng;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// A contiguous slice extracted from tensor data.
#[derive(Debug, Clone)]
struct TensorSlice {
    tensor_name: String,
    offset: usize,
    length: usize,
    values: Vec<f32>,
    raw_bytes: Vec<u8>,
}

/// Describes how to slice a tensor.
#[derive(Debug, Clone, PartialEq)]
enum SliceOp {
    /// Elements in `[start, end)`.
    Range(usize, usize),
    /// All elements in row `r` of a 2-D tensor.
    Row(usize),
    /// All elements in column `c` of a 2-D tensor.
    Column(usize),
    /// Every `n`-th element from the flat buffer.
    Stride(usize),
}

/// Descriptive statistics for a slice.
#[derive(Debug, Clone)]
struct SliceStats {
    mean: f64,
    min: f32,
    max: f32,
    sum: f64,
}

/// A completed slice operation with its result and statistics.
#[derive(Debug, Clone)]
struct SliceResult {
    op: SliceOp,
    slice: TensorSlice,
    stats: SliceStats,
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
fn compute_stats(values: &[f32]) -> SliceStats {
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
fn build_slice(name: &str, data: &[f32], indices: &[usize]) -> TensorSlice {
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

/// Execute a single slice operation against flat tensor data.
///
/// For `Row` and `Column` the caller must supply `cols` (number of columns
/// in the logical 2-D layout).  The value is ignored for `Range`/`Stride`.
fn execute_slice(
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
fn bytes_to_hex(data: &[u8], limit: usize) -> String {
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
fn f32_to_f16_roundtrip(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&v| {
            let bits = f16_from_f32(v);
            f16_to_f32(bits)
        })
        .collect()
}

/// Convert an f32 to its IEEE 754 half-precision bit pattern.
fn f16_from_f32(value: f32) -> u16 {
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
fn f16_to_f32(bits: u16) -> f32 {
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
fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Compute mean absolute error between two equally-sized slices.
fn mean_abs_error(a: &[f32], b: &[f32]) -> f64 {
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

fn print_slice_summary(result: &SliceResult) {
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

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("analysis_slice")?;

    // -- Section 1: Create synthetic tensor with known values ----------------
    println!("=== Tensor Slice Extraction ===\n");

    let n = 100;
    let known_values: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

    // Extend with RNG-generated data to demonstrate ctx.rng() usage
    let extra: Vec<f32> = (0..156)
        .map(|_| ctx.rng().gen_range(-1.0_f32..1.0))
        .collect();
    let mut data = known_values.clone();
    data.extend_from_slice(&extra);

    println!(
        "Tensor: {} elements ({} known + {} random)\n",
        data.len(),
        n,
        extra.len()
    );

    // -- Section 2: Slice by index range ------------------------------------
    println!("--- Slice by Index Range: elements 10..20 ---");
    let range_result = execute_slice("weights", &data, &SliceOp::Range(10, 20), 0)
        .map_err(CookbookError::invalid_format)?;
    print_slice_summary(&range_result);
    println!();

    // -- Section 3: Slice by row for 2-D tensor [10, 10]: row 3 -------------
    println!("--- Slice by Row (shape [10,10], row 3) ---");
    let cols = 10;
    let row_result = execute_slice("weights", &data, &SliceOp::Row(3), cols)
        .map_err(CookbookError::invalid_format)?;
    print_slice_summary(&row_result);
    println!();

    // -- Section 4: Slice by column for 2-D tensor: column 5 ----------------
    println!("--- Slice by Column (shape [10,10], column 5) ---");
    let col_result = execute_slice("weights", &data, &SliceOp::Column(5), cols)
        .map_err(CookbookError::invalid_format)?;
    print_slice_summary(&col_result);
    println!();

    // -- Section 5: Slice with stride: every 3rd element --------------------
    println!("--- Slice with Stride (every 3rd element) ---");
    let stride_result = execute_slice("weights", &data, &SliceOp::Stride(3), 0)
        .map_err(CookbookError::invalid_format)?;
    print_slice_summary(&stride_result);
    println!();

    // -- Section 6: Dtype conversion f32 -> f16 round-trip ------------------
    println!("--- Dtype Conversion: f32 -> f16 -> f32 ---");
    let sample = &range_result.slice.values;
    let converted = f32_to_f16_roundtrip(sample);

    println!("  Original f32:  {:?}", &sample[..5.min(sample.len())]);
    println!(
        "  After f16 RT:  {:?}",
        &converted[..5.min(converted.len())]
    );

    let mae = mean_abs_error(sample, &converted);
    let max_err = max_abs_error(sample, &converted);
    println!("  Mean abs error: {mae:.8}");
    println!("  Max  abs error: {max_err:.8}");
    println!();

    // -- Section 7: Slice summary table -------------------------------------
    println!("--- Slice Summary Table ---");
    println!(
        "{:<22} {:>8} {:>14} {:>14} {:>10}",
        "Operation", "Length", "Byte Start", "Byte End", "Mean"
    );
    println!("{}", "-".repeat(70));

    let all_results = [&range_result, &row_result, &col_result, &stride_result];
    for r in &all_results {
        let byte_start = r.slice.offset * 4;
        let byte_end = byte_start + r.slice.length * 4;
        println!(
            "{:<22} {:>8} {:>14} {:>14} {:>10.6}",
            format!("{}", r.op),
            r.slice.length,
            format!("0x{byte_start:04x}"),
            format!("0x{byte_end:04x}"),
            r.stats.mean,
        );
    }

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build deterministic test data: [0.00, 0.01, ..., 0.99].
    fn make_known_data(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 * 0.01).collect()
    }

    #[test]
    fn test_range_slice_values() {
        let data = make_known_data(100);
        let result = execute_slice("t", &data, &SliceOp::Range(10, 20), 0);
        assert!(result.is_ok());
        let r = result.expect("range slice should succeed");
        assert_eq!(r.slice.length, 10);
        for (i, v) in r.slice.values.iter().enumerate() {
            let expected = (10 + i) as f32 * 0.01;
            assert!((v - expected).abs() < 1e-6, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_range_slice_clamped_to_data_length() {
        let data = make_known_data(15);
        let result = execute_slice("t", &data, &SliceOp::Range(10, 100), 0);
        assert!(result.is_ok());
        let r = result.expect("clamped range should succeed");
        assert_eq!(r.slice.length, 5);
    }

    #[test]
    fn test_range_invalid_empty() {
        let data = make_known_data(10);
        let result = execute_slice("t", &data, &SliceOp::Range(5, 5), 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_row_slice() {
        let data = make_known_data(100);
        let result = execute_slice("t", &data, &SliceOp::Row(3), 10);
        assert!(result.is_ok());
        let r = result.expect("row slice should succeed");
        assert_eq!(r.slice.length, 10);
        // Row 3 starts at index 30
        let expected_first = 0.30_f32;
        assert!((r.slice.values[0] - expected_first).abs() < 1e-6);
    }

    #[test]
    fn test_row_out_of_bounds() {
        let data = make_known_data(20);
        let result = execute_slice("t", &data, &SliceOp::Row(10), 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_column_slice() {
        let data = make_known_data(100);
        let result = execute_slice("t", &data, &SliceOp::Column(5), 10);
        assert!(result.is_ok());
        let r = result.expect("column slice should succeed");
        // Column 5 with 100 elements and 10 cols -> rows 0..10
        assert_eq!(r.slice.length, 10);
        // Values: data[5], data[15], data[25], ...
        let expected: Vec<f32> = (0..10).map(|row| (row * 10 + 5) as f32 * 0.01).collect();
        for (i, (got, want)) in r.slice.values.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6, "column mismatch at row {i}");
        }
    }

    #[test]
    fn test_column_out_of_bounds() {
        let data = make_known_data(20);
        let result = execute_slice("t", &data, &SliceOp::Column(10), 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_stride_slice() {
        let data = make_known_data(30);
        let result = execute_slice("t", &data, &SliceOp::Stride(3), 0);
        assert!(result.is_ok());
        let r = result.expect("stride slice should succeed");
        // Every 3rd: indices 0, 3, 6, 9, ..., 27 -> 10 elements
        assert_eq!(r.slice.length, 10);
        for (i, v) in r.slice.values.iter().enumerate() {
            let expected = (i * 3) as f32 * 0.01;
            assert!((v - expected).abs() < 1e-6, "stride mismatch at {i}");
        }
    }

    #[test]
    fn test_stride_zero_rejected() {
        let data = make_known_data(10);
        let result = execute_slice("t", &data, &SliceOp::Stride(0), 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_stats() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_stats(&values);
        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
        assert!((stats.sum - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_roundtrip_precision_loss() {
        let values: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let converted = f32_to_f16_roundtrip(&values);
        // f16 has ~3 decimal digits of precision
        for (orig, conv) in values.iter().zip(converted.iter()) {
            let err = (orig - conv).abs();
            assert!(
                err < 0.001,
                "f16 round-trip error too large: {orig} -> {conv} (err={err})"
            );
        }
        // But should NOT be perfectly equal for all values (precision loss)
        let any_diff = values
            .iter()
            .zip(converted.iter())
            .any(|(a, b)| (a - b).abs() > 0.0);
        assert!(any_diff, "expected some precision loss from f16 round-trip");
    }
}
