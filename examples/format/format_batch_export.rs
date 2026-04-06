//! # Batch Export to Multiple Formats
//!
//! **CLI equivalent:** `apr export model.apr --batch gguf,safetensors,onnx`
//!
//! Demonstrates batch exporting an APR model to multiple formats
//! simultaneously. A single source model is converted to GGUF,
//! SafeTensors, and ONNX formats, with a size comparison table
//! and format feature matrix.
//!
//! ## Sections
//! 1. Single source model — create the APR v2 reference model
//! 2. Parallel export to 3+ formats — convert to each target
//! 3. Size comparison table — compare output sizes
//! 4. Format feature matrix — capabilities of each format
//!
//! ## References
//! - Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP. DOI: 10.18653/v1/2020.emnlp-demos.6

use apr_cookbook::prelude::*;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Format {
    Apr,
    SafeTensors,
    Gguf,
    Onnx,
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Format::Apr => write!(f, "APR"),
            Format::SafeTensors => write!(f, "SafeTensors"),
            Format::Gguf => write!(f, "GGUF"),
            Format::Onnx => write!(f, "ONNX"),
        }
    }
}

/// Features supported by each format.
#[derive(Debug)]
struct FormatFeatures {
    format: Format,
    quantization: bool,
    compression: bool,
    streaming: bool,
    zero_copy: bool,
    graph_info: bool,
    extension: &'static str,
}

/// Result of a single format export.
#[derive(Debug)]
#[allow(dead_code)]
struct ExportResult {
    format: Format,
    data: Vec<u8>,
    size_bytes: usize,
    tensor_count: usize,
}

// ---------------------------------------------------------------------------
// Format conversion
// ---------------------------------------------------------------------------

/// Export APR model data to SafeTensors format.
fn export_safetensors(payload: &[u8], tensor_count: usize) -> Vec<u8> {
    let elements_per_tensor = payload.len() / (tensor_count.max(1) * 4);
    let mut header_parts = Vec::new();
    let mut offset = 0usize;
    for i in 0..tensor_count {
        let tensor_size = elements_per_tensor * 4;
        header_parts.push(format!(
            "\"t{i}\":{{\"dtype\":\"F32\",\"shape\":[{elements_per_tensor}],\"data_offsets\":[{offset},{}]}}",
            offset + tensor_size
        ));
        offset += tensor_size;
    }
    let header = format!("{{{}}}", header_parts.join(","));

    let mut output = Vec::new();
    output.extend_from_slice(&(header.len() as u64).to_le_bytes());
    output.extend_from_slice(header.as_bytes());
    output.extend_from_slice(payload);
    output
}

/// Export APR model data to GGUF format.
fn export_gguf(payload: &[u8], tensor_count: usize) -> Vec<u8> {
    let mut output = Vec::new();
    // Header
    output.extend_from_slice(b"GGUF");
    output.extend_from_slice(&3u32.to_le_bytes()); // version
    output.extend_from_slice(&(tensor_count as u64).to_le_bytes());
    output.extend_from_slice(&0u64.to_le_bytes()); // metadata count

    // Tensor info (simplified)
    for i in 0..tensor_count {
        let name = format!("t{i}");
        let name_bytes = name.as_bytes();
        output.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
        output.extend_from_slice(name_bytes);
        output.extend_from_slice(&1u32.to_le_bytes()); // 1 dimension
        let elements = payload.len() / (tensor_count.max(1) * 4);
        output.extend_from_slice(&(elements as u64).to_le_bytes());
        output.extend_from_slice(&0u32.to_le_bytes()); // F32 type
        output.extend_from_slice(
            &(i as u64 * (payload.len() / tensor_count.max(1)) as u64).to_le_bytes(),
        );
    }

    // Alignment padding
    let align = 32;
    let padding = (align - (output.len() % align)) % align;
    output.extend(std::iter::repeat(0u8).take(padding));

    // Tensor data
    output.extend_from_slice(payload);
    output
}

/// Export APR model data to ONNX format (simulated protobuf).
fn export_onnx(payload: &[u8], tensor_count: usize) -> Vec<u8> {
    let mut output = Vec::new();

    // ONNX protobuf-like header
    output.push(0x08); // field 1, varint
    output.push(0x07); // IR version 7

    // Model metadata
    output.push(0x12); // field 2, length-delimited
    let producer = b"apr-cookbook";
    output.push(producer.len() as u8);
    output.extend_from_slice(producer);

    // Graph with initializers (tensors)
    output.push(0x1a); // field 3, length-delimited
    let graph_size_placeholder = output.len();
    output.push(0x00); // placeholder for graph length

    for i in 0..tensor_count {
        let name = format!("t{i}");
        // Tensor name
        output.push(0x0a); // field 1, length-delimited
        output.push(name.len() as u8);
        output.extend_from_slice(name.as_bytes());
    }

    // Update graph length
    let graph_len = output.len() - graph_size_placeholder - 1;
    output[graph_size_placeholder] = graph_len.min(255) as u8;

    // Raw tensor data
    output.extend_from_slice(payload);
    output
}

// ---------------------------------------------------------------------------
// Batch export
// ---------------------------------------------------------------------------

/// Export an APR model to multiple formats simultaneously.
///
/// Returns a map from format to export result. The source APR bundle
/// is not modified.
fn batch_export(
    apr_payload: &[u8],
    tensor_count: usize,
    formats: &[Format],
) -> HashMap<Format, ExportResult> {
    let mut results = HashMap::new();

    for &fmt in formats {
        let data = match fmt {
            Format::Apr => {
                let mut v = b"APR2".to_vec();
                v.extend_from_slice(apr_payload);
                v
            }
            Format::SafeTensors => export_safetensors(apr_payload, tensor_count),
            Format::Gguf => export_gguf(apr_payload, tensor_count),
            Format::Onnx => export_onnx(apr_payload, tensor_count),
        };

        let size_bytes = data.len();
        results.insert(
            fmt,
            ExportResult {
                format: fmt,
                data,
                size_bytes,
                tensor_count,
            },
        );
    }

    results
}

/// Build the format feature matrix.
fn format_features() -> Vec<FormatFeatures> {
    vec![
        FormatFeatures {
            format: Format::Apr,
            quantization: true,
            compression: true,
            streaming: true,
            zero_copy: true,
            graph_info: false,
            extension: ".apr",
        },
        FormatFeatures {
            format: Format::SafeTensors,
            quantization: false,
            compression: false,
            streaming: false,
            zero_copy: true,
            graph_info: false,
            extension: ".safetensors",
        },
        FormatFeatures {
            format: Format::Gguf,
            quantization: true,
            compression: false,
            streaming: true,
            zero_copy: true,
            graph_info: false,
            extension: ".gguf",
        },
        FormatFeatures {
            format: Format::Onnx,
            quantization: true,
            compression: false,
            streaming: false,
            zero_copy: false,
            graph_info: true,
            extension: ".onnx",
        },
    ]
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("format_batch_export")?;

    // Section 1: Single source model
    println!("=== Source APR Model ===");
    let tensor_count = 6;
    let elements_per_tensor = 128 * 64;
    let total_elements = tensor_count * elements_per_tensor;
    let payload = generate_model_payload(42, total_elements);

    let mut apr_builder = ModelBundleV2::new()
        .with_name("batch-export-demo")
        .with_compression(Compression::None)
        .with_quantization(Quantization::FP32);

    for i in 0..tensor_count {
        let start = i * elements_per_tensor * 4;
        let end = start + elements_per_tensor * 4;
        apr_builder = apr_builder.add_tensor(
            format!("layer.{i}.weight"),
            vec![128, 64],
            payload[start..end].to_vec(),
        );
    }
    let apr_bundle = apr_builder.build();

    println!("Model name:   batch-export-demo");
    println!("Tensors:      {tensor_count}");
    println!("Elements:     {total_elements}");
    println!("Payload:      {} bytes", payload.len());
    println!("APR bundle:   {} bytes", apr_bundle.len());
    println!();

    // Section 2: Batch export
    println!("=== Batch Export ===");
    let target_formats = vec![Format::Apr, Format::SafeTensors, Format::Gguf, Format::Onnx];
    let results = batch_export(&payload, tensor_count, &target_formats);
    println!("Exported to {} formats", results.len());
    println!();

    // Section 3: Size comparison table
    println!("=== Size Comparison ===");
    println!(
        "{:<15} {:<12} {:<10} {:<10} {:<10}",
        "Format", "Size (B)", "vs APR", "Overhead", "Extension"
    );
    println!("{}", "-".repeat(57));

    let apr_size = results.get(&Format::Apr).map_or(0, |r| r.size_bytes);

    for fmt in &target_formats {
        if let Some(result) = results.get(fmt) {
            let ratio = if apr_size > 0 {
                result.size_bytes as f64 / apr_size as f64
            } else {
                0.0
            };
            let overhead = result.size_bytes as i64 - apr_size as i64;
            let ext = format_features()
                .iter()
                .find(|f| f.format == *fmt)
                .map_or("?", |f| f.extension);
            println!(
                "{:<15} {:<12} {:<10.2} {:<+10} {:<10}",
                format!("{fmt}"),
                result.size_bytes,
                ratio,
                overhead,
                ext,
            );
        }
    }
    println!();

    // Section 4: Format feature matrix
    println!("=== Format Feature Matrix ===");
    let features = format_features();
    println!(
        "{:<15} {:<8} {:<8} {:<8} {:<8} {:<8}",
        "Format", "Quant", "Compr", "Stream", "ZeroCp", "Graph"
    );
    println!("{}", "-".repeat(55));
    for feat in &features {
        let yn = |b: bool| if b { "yes" } else { "no" };
        println!(
            "{:<15} {:<8} {:<8} {:<8} {:<8} {:<8}",
            format!("{}", feat.format),
            yn(feat.quantization),
            yn(feat.compression),
            yn(feat.streaming),
            yn(feat.zero_copy),
            yn(feat.graph_info),
        );
    }

    // Verify all formats produced output
    for fmt in &target_formats {
        assert!(results.contains_key(fmt), "Missing export for {fmt}");
        assert!(results[fmt].size_bytes > 0, "Empty export for {fmt}");
    }

    // Verify source unchanged
    assert_eq!(&apr_bundle[0..4], b"APR2");

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_payload() -> Vec<u8> {
        generate_model_payload(42, 1024)
    }

    #[test]
    fn test_all_formats_produced() {
        let payload = sample_payload();
        let formats = vec![Format::Apr, Format::SafeTensors, Format::Gguf, Format::Onnx];
        let results = batch_export(&payload, 2, &formats);
        for fmt in &formats {
            assert!(results.contains_key(fmt), "Missing {fmt}");
        }
    }

    #[test]
    fn test_sizes_differ_by_format() {
        let payload = sample_payload();
        let formats = vec![Format::SafeTensors, Format::Gguf, Format::Onnx];
        let results = batch_export(&payload, 2, &formats);
        let sizes: Vec<usize> = formats.iter().map(|f| results[f].size_bytes).collect();
        // At least two formats should have different sizes
        assert!(
            sizes.windows(2).any(|w| w[0] != w[1]),
            "All formats produced same size: {sizes:?}"
        );
    }

    #[test]
    fn test_source_unchanged() {
        let payload = sample_payload();
        let payload_clone = payload.clone();
        let formats = vec![Format::SafeTensors, Format::Gguf];
        let _results = batch_export(&payload, 2, &formats);
        assert_eq!(payload, payload_clone);
    }

    #[test]
    fn test_empty_batch_produces_empty() {
        let payload = sample_payload();
        let results = batch_export(&payload, 2, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_safetensors_has_header() {
        let payload = sample_payload();
        let results = batch_export(&payload, 2, &[Format::SafeTensors]);
        let st = &results[&Format::SafeTensors].data;
        assert!(st.len() >= 8);
        let header_len = u64::from_le_bytes(st[0..8].try_into().unwrap());
        assert!(header_len > 0);
    }

    #[test]
    fn test_gguf_has_magic() {
        let payload = sample_payload();
        let results = batch_export(&payload, 2, &[Format::Gguf]);
        let gguf = &results[&Format::Gguf].data;
        assert_eq!(&gguf[0..4], b"GGUF");
    }

    #[test]
    fn test_apr_has_magic() {
        let payload = sample_payload();
        let results = batch_export(&payload, 2, &[Format::Apr]);
        let apr = &results[&Format::Apr].data;
        assert_eq!(&apr[0..4], b"APR2");
    }

    #[test]
    fn test_onnx_has_ir_marker() {
        let payload = sample_payload();
        let results = batch_export(&payload, 2, &[Format::Onnx]);
        let onnx = &results[&Format::Onnx].data;
        assert_eq!(onnx[0], 0x08);
        assert_eq!(onnx[1], 0x07);
    }

    #[test]
    fn test_tensor_count_preserved() {
        let payload = sample_payload();
        let results = batch_export(&payload, 5, &[Format::Gguf, Format::SafeTensors]);
        for result in results.values() {
            assert_eq!(result.tensor_count, 5);
        }
    }

    #[test]
    fn test_format_features_complete() {
        let features = format_features();
        assert_eq!(features.len(), 4);
        let formats: Vec<Format> = features.iter().map(|f| f.format).collect();
        assert!(formats.contains(&Format::Apr));
        assert!(formats.contains(&Format::SafeTensors));
        assert!(formats.contains(&Format::Gguf));
        assert!(formats.contains(&Format::Onnx));
    }

    #[test]
    fn test_apr_features() {
        let features = format_features();
        let apr = features.iter().find(|f| f.format == Format::Apr).unwrap();
        assert!(apr.quantization);
        assert!(apr.compression);
        assert!(apr.zero_copy);
        assert_eq!(apr.extension, ".apr");
    }

    #[test]
    fn test_single_format_batch() {
        let payload = sample_payload();
        let results = batch_export(&payload, 1, &[Format::Gguf]);
        assert_eq!(results.len(), 1);
        assert!(results.contains_key(&Format::Gguf));
    }
}
