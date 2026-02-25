//! # Combined Quantization and Compression Conversion
//!
//! **CLI equivalent:** `apr convert model.apr --quantize int4 --compress lz4`
//!
//! Demonstrates re-quantizing and re-compressing an APR model. This is a
//! common workflow when optimizing models for edge deployment: reduce
//! precision from FP32 to INT4/INT8 and apply compression for smaller
//! file sizes.
//!
//! ## Sections
//! 1. Original model stats — FP32 uncompressed baseline
//! 2. Quantization — FP32 → INT4 precision reduction
//! 3. Compression — None → LZ4 byte-level compression
//! 4. Size comparison table — before/after for each transformation

use apr_cookbook::prelude::*;

// ---------------------------------------------------------------------------
// Quantization simulation
// ---------------------------------------------------------------------------

/// Quantization level with metadata.
#[derive(Debug, Clone, Copy)]
struct QuantLevel {
    name: &'static str,
    bits_per_element: usize,
    quantization: Quantization,
}

const FP32: QuantLevel = QuantLevel {
    name: "FP32",
    bits_per_element: 32,
    quantization: Quantization::FP32,
};

const FP16: QuantLevel = QuantLevel {
    name: "FP16",
    bits_per_element: 16,
    quantization: Quantization::FP16,
};

const INT8: QuantLevel = QuantLevel {
    name: "INT8",
    bits_per_element: 8,
    quantization: Quantization::Int8,
};

const INT4: QuantLevel = QuantLevel {
    name: "INT4",
    bits_per_element: 4,
    quantization: Quantization::Int4,
};

/// Simulate quantizing FP32 bytes to a lower precision.
///
/// Returns the quantized bytes and the number of elements.
fn quantize_payload(fp32_bytes: &[u8], target: QuantLevel) -> (Vec<u8>, usize) {
    let num_elements = fp32_bytes.len() / 4; // 4 bytes per FP32
    let target_bytes = (num_elements * target.bits_per_element).div_ceil(8);

    // Simulate quantization by generating deterministic data at the target size.
    // In a real implementation, this would dequantize → requantize with proper scaling.
    let seed = hash_name_to_seed(target.name);
    let mut quantized = Vec::with_capacity(target_bytes);
    for i in 0..target_bytes {
        let idx = i as u64;
        let byte = ((idx.wrapping_mul(seed)
            ^ u64::from(fp32_bytes.get(i % fp32_bytes.len()).copied().unwrap_or(0)))
            & 0xFF) as u8;
        quantized.push(byte);
    }

    (quantized, num_elements)
}

/// Simulate dequantizing back to FP32 bytes.
#[allow(dead_code)]
fn dequantize_payload(quantized_bytes: &[u8], source: QuantLevel, num_elements: usize) -> Vec<u8> {
    let fp32_bytes = num_elements * 4;
    let mut output = Vec::with_capacity(fp32_bytes);

    let seed = hash_name_to_seed(source.name);
    for i in 0..fp32_bytes {
        let idx = i as u64;
        let byte = ((idx.wrapping_mul(seed)
            ^ u64::from(
                quantized_bytes
                    .get(i % quantized_bytes.len())
                    .copied()
                    .unwrap_or(0),
            ))
            & 0xFF) as u8;
        output.push(byte);
    }

    output
}

// ---------------------------------------------------------------------------
// Compression simulation
// ---------------------------------------------------------------------------

/// Compression level metadata.
#[derive(Debug, Clone, Copy)]
struct CompressLevel {
    name: &'static str,
    compression: Compression,
    typical_ratio: f64, // e.g., 0.7 means 70% of original
}

const NO_COMPRESS: CompressLevel = CompressLevel {
    name: "None",
    compression: Compression::None,
    typical_ratio: 1.0,
};

const LZ4_COMPRESS: CompressLevel = CompressLevel {
    name: "LZ4",
    compression: Compression::Lz4,
    typical_ratio: 0.75,
};

const ZSTD_COMPRESS: CompressLevel = CompressLevel {
    name: "Zstd",
    compression: Compression::Zstd,
    typical_ratio: 0.60,
};

/// Simulate compressing data.
///
/// Returns the "compressed" data (simulated by truncating to ratio + adding header).
fn compress_payload(data: &[u8], level: CompressLevel) -> Vec<u8> {
    if matches!(level.compression, Compression::None) {
        return data.to_vec();
    }

    let compressed_size = (data.len() as f64 * level.typical_ratio) as usize;
    let mut output = Vec::with_capacity(compressed_size + 8);

    // 4-byte compression marker + 4-byte original size
    let marker: &[u8] = match level.compression {
        Compression::Lz4 => b"LZ4B",
        Compression::Zstd => b"ZSTD",
        Compression::None => b"\0\0\0\0",
    };
    output.extend_from_slice(marker);
    output.extend_from_slice(&(data.len() as u32).to_le_bytes());

    // Simulated compressed data
    output.extend_from_slice(&data[..compressed_size.min(data.len())]);
    output
}

// ---------------------------------------------------------------------------
// Conversion pipeline
// ---------------------------------------------------------------------------

/// Full conversion result with metrics.
#[allow(dead_code)]
struct ConversionResult {
    original_size: usize,
    after_quantize_size: usize,
    after_compress_size: usize,
    apr_bundle: Vec<u8>,
    num_elements: usize,
    quant: QuantLevel,
    compress: CompressLevel,
}

/// Execute the full conversion pipeline:
/// 1. Create FP32 model
/// 2. Quantize to target precision
/// 3. Compress with target algorithm
/// 4. Bundle into APR v2
fn convert_pipeline(
    fp32_payload: &[u8],
    tensor_name: &str,
    tensor_shape: Vec<usize>,
    target_quant: QuantLevel,
    target_compress: CompressLevel,
) -> ConversionResult {
    let original_size = fp32_payload.len();

    // Step 1: Quantize
    let (quantized, num_elements) = quantize_payload(fp32_payload, target_quant);
    let after_quantize_size = quantized.len();

    // Step 2: Compress
    let compressed = compress_payload(&quantized, target_compress);
    let after_compress_size = compressed.len();

    // Step 3: Bundle into APR v2
    let apr_bundle = ModelBundleV2::new()
        .with_name("converted-model")
        .with_compression(target_compress.compression)
        .with_quantization(target_quant.quantization)
        .add_tensor(tensor_name, tensor_shape, compressed.clone())
        .build();

    ConversionResult {
        original_size,
        after_quantize_size,
        after_compress_size,
        apr_bundle,
        num_elements,
        quant: target_quant,
        compress: target_compress,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("format_convert_quantize")?;

    // Section 1: Original model stats
    let dim: usize = 256;
    let num_elements = dim * dim;
    let fp32_payload = generate_model_payload(42, num_elements); // 256x256 FP32 → num_elements * 4 bytes

    println!("=== Original Model ===");
    println!("Tensor:       weight [{dim}x{dim}]");
    println!("Elements:     {num_elements}");
    println!("Precision:    FP32 (32 bits/element)");
    println!("Compression:  None");
    println!(
        "Size:         {} bytes ({:.1} KB)",
        fp32_payload.len(),
        fp32_payload.len() as f64 / 1024.0
    );
    println!();

    // Section 2: Quantization sweep
    println!("=== Quantization Comparison ===");
    println!(
        "{:<8} {:<12} {:<12} {:<10} {:<10}",
        "Level", "Bits/Elem", "Size (B)", "Ratio", "Savings"
    );
    println!("{}", "-".repeat(52));

    for quant in [FP32, FP16, INT8, INT4] {
        let (quantized, _) = quantize_payload(&fp32_payload, quant);
        let ratio = quantized.len() as f64 / fp32_payload.len() as f64;
        let savings = (1.0 - ratio) * 100.0;
        println!(
            "{:<8} {:<12} {:<12} {:<10.2} {:<10.1}%",
            quant.name,
            quant.bits_per_element,
            quantized.len(),
            ratio,
            savings,
        );
    }
    println!();

    // Section 3: Compression comparison
    println!("=== Compression Comparison (on INT8 data) ===");
    let (int8_data, _) = quantize_payload(&fp32_payload, INT8);
    println!(
        "{:<8} {:<12} {:<10} {:<10}",
        "Method", "Size (B)", "Ratio", "Savings"
    );
    println!("{}", "-".repeat(40));

    for compress in [NO_COMPRESS, LZ4_COMPRESS, ZSTD_COMPRESS] {
        let compressed = compress_payload(&int8_data, compress);
        let ratio = compressed.len() as f64 / int8_data.len() as f64;
        let savings = (1.0 - ratio) * 100.0;
        println!(
            "{:<8} {:<12} {:<10.2} {:<10.1}%",
            compress.name,
            compressed.len(),
            ratio,
            savings,
        );
    }
    println!();

    // Section 4: Full pipeline — FP32/None → INT4/LZ4
    println!("=== Full Pipeline: FP32/None → INT4/LZ4 ===");
    let result = convert_pipeline(&fp32_payload, "weight", vec![dim, dim], INT4, LZ4_COMPRESS);

    println!("{:<25} {:<12} {:<10}", "Stage", "Size (B)", "Ratio");
    println!("{}", "-".repeat(47));
    println!(
        "{:<25} {:<12} {:<10}",
        "Original (FP32, None)", result.original_size, "1.00",
    );
    println!(
        "{:<25} {:<12} {:<10.2}",
        format!("Quantized ({})", result.quant.name),
        result.after_quantize_size,
        result.after_quantize_size as f64 / result.original_size as f64,
    );
    println!(
        "{:<25} {:<12} {:<10.2}",
        format!("Compressed ({})", result.compress.name),
        result.after_compress_size,
        result.after_compress_size as f64 / result.original_size as f64,
    );
    println!(
        "{:<25} {:<12} {:<10.2}",
        "APR v2 Bundle",
        result.apr_bundle.len(),
        result.apr_bundle.len() as f64 / result.original_size as f64,
    );
    println!();
    println!(
        "Total reduction: {:.1}%",
        (1.0 - result.apr_bundle.len() as f64 / result.original_size as f64) * 100.0
    );

    assert_eq!(&result.apr_bundle[0..4], b"APR2");
    assert!(result.after_quantize_size < result.original_size);

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fp32(n: usize) -> Vec<u8> {
        generate_model_payload(42, n)
    }

    #[test]
    fn test_quantize_reduces_size() {
        let fp32 = sample_fp32(1024);
        let (int8, _) = quantize_payload(&fp32, INT8);
        assert!(int8.len() < fp32.len());
    }

    #[test]
    fn test_int4_smaller_than_int8() {
        let fp32 = sample_fp32(1024);
        let (int8, _) = quantize_payload(&fp32, INT8);
        let (int4, _) = quantize_payload(&fp32, INT4);
        assert!(int4.len() < int8.len());
    }

    #[test]
    fn test_fp32_quantize_same_size() {
        let fp32 = sample_fp32(1024);
        let (requant, _) = quantize_payload(&fp32, FP32);
        assert_eq!(requant.len(), fp32.len());
    }

    #[test]
    fn test_quantize_changes_content() {
        let fp32 = sample_fp32(1024);
        let (int8, _) = quantize_payload(&fp32, INT8);
        // Content differs because of precision reduction simulation
        assert_ne!(
            &int8[..int8.len().min(fp32.len())],
            &fp32[..int8.len().min(fp32.len())]
        );
    }

    #[test]
    fn test_compression_reduces_size() {
        let data = sample_fp32(1024);
        let compressed = compress_payload(&data, LZ4_COMPRESS);
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn test_no_compression_identity() {
        let data = sample_fp32(256);
        let result = compress_payload(&data, NO_COMPRESS);
        assert_eq!(result, data);
    }

    #[test]
    fn test_zstd_smaller_than_lz4() {
        let data = sample_fp32(4096);
        let lz4 = compress_payload(&data, LZ4_COMPRESS);
        let zstd = compress_payload(&data, ZSTD_COMPRESS);
        assert!(zstd.len() < lz4.len());
    }

    #[test]
    fn test_pipeline_output_smaller_than_input() {
        let fp32 = sample_fp32(2048);
        let result = convert_pipeline(&fp32, "w", vec![2048], INT4, LZ4_COMPRESS);
        assert!(result.after_compress_size < result.original_size);
    }

    #[test]
    fn test_pipeline_produces_valid_apr() {
        let fp32 = sample_fp32(512);
        let result = convert_pipeline(&fp32, "w", vec![512], INT8, LZ4_COMPRESS);
        assert_eq!(&result.apr_bundle[0..4], b"APR2");
    }

    #[test]
    fn test_num_elements_correct() {
        let fp32 = sample_fp32(1024);
        let (_, num) = quantize_payload(&fp32, INT8);
        assert_eq!(num, 1024);
    }

    #[test]
    fn test_dequantize_produces_fp32_size() {
        let fp32 = sample_fp32(256);
        let (quantized, num) = quantize_payload(&fp32, INT8);
        let restored = dequantize_payload(&quantized, INT8, num);
        assert_eq!(restored.len(), 256 * 4);
    }

    #[test]
    fn test_compress_header_lz4() {
        let data = sample_fp32(256);
        let compressed = compress_payload(&data, LZ4_COMPRESS);
        assert_eq!(&compressed[0..4], b"LZ4B");
    }

    #[test]
    fn test_compress_header_zstd() {
        let data = sample_fp32(256);
        let compressed = compress_payload(&data, ZSTD_COMPRESS);
        assert_eq!(&compressed[0..4], b"ZSTD");
    }
}
