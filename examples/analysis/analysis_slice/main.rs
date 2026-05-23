#![allow(unused_imports)]
//! # Tensor Slice Extraction and Decoding
//!
//! CLI equivalent: `apr tensors model.apr --slice weights --range 10..20`
//! Contract: contracts/recipe-iiur-v1.yaml
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
//!
//!
//! ## Format Variants
//! ```bash
//! apr tensors model.apr          # APR native format
//! apr tensors model.gguf         # GGUF (llama.cpp compatible)
//! apr tensors model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use rand::Rng;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
