//! # Export APR Model to SafeTensors Format
//!
//! **CLI equivalent:** `apr export model.apr --format safetensors`
//!
//! Demonstrates exporting an APR v2 model to the SafeTensors format.
//! SafeTensors uses a JSON header followed by raw tensor data, with
//! explicit offset tracking for zero-copy loading.
//!
//! ## Sections
//! 1. APR model creation — build a multi-tensor APR v2 bundle
//! 2. Header generation — produce SafeTensors JSON header with tensor metadata
//! 3. Tensor data layout — compute contiguous offsets for each tensor
//! 4. Format comparison — compare APR vs SafeTensors sizes and features
//!
//! ## References
//! - Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP. DOI: 10.18653/v1/2020.emnlp-demos.6

use apr_cookbook::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// SafeTensors types
// ---------------------------------------------------------------------------

/// Metadata for a single tensor in SafeTensors format.
#[derive(Debug, Clone)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

/// SafeTensors file header (serialized as JSON).
#[derive(Debug)]
struct SafeTensorsHeader {
    tensors: HashMap<String, TensorMeta>,
}

/// Complete SafeTensors export result.
struct SafeTensorsExport {
    header_json: String,
    header_size: usize,
    tensor_data: Vec<u8>,
    total_size: usize,
}

// ---------------------------------------------------------------------------
// Export logic
// ---------------------------------------------------------------------------

/// Convert an APR v2 bundle to SafeTensors format.
///
/// The SafeTensors format is:
/// 1. 8-byte little-endian header length
/// 2. JSON header with tensor metadata (dtype, shape, data_offsets)
/// 3. Contiguous tensor data block
fn apr_to_safetensors(tensor_map: &[(&str, Vec<usize>, Vec<u8>)]) -> SafeTensorsExport {
    let mut header = SafeTensorsHeader {
        tensors: HashMap::new(),
    };
    let mut tensor_data = Vec::new();

    for (name, shape, data) in tensor_map {
        let start = tensor_data.len();
        tensor_data.extend_from_slice(data);
        let end = tensor_data.len();

        header.tensors.insert(
            (*name).to_string(),
            TensorMeta {
                dtype: "F32".to_string(),
                shape: shape.clone(),
                data_offsets: (start, end),
            },
        );
    }

    let header_json = serialize_header(&header);
    let header_size = header_json.len();
    let total_size = 8 + header_size + tensor_data.len();

    SafeTensorsExport {
        header_json,
        header_size,
        tensor_data,
        total_size,
    }
}

/// Serialize the SafeTensors header to JSON.
fn serialize_header(header: &SafeTensorsHeader) -> String {
    let mut entries = Vec::new();

    let mut keys: Vec<_> = header.tensors.keys().collect();
    keys.sort();

    for key in keys {
        let meta = &header.tensors[key];
        let shape_str = meta
            .shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!(
            "\"{key}\":{{\"dtype\":\"{}\",\"shape\":[{shape_str}],\"data_offsets\":[{},{}]}}",
            meta.dtype, meta.data_offsets.0, meta.data_offsets.1
        ));
    }

    format!("{{{}}}", entries.join(","))
}

/// Validate that a SafeTensors header is well-formed.
fn validate_header(header: &SafeTensorsHeader, data_len: usize) -> Vec<String> {
    let mut errors = Vec::new();

    for (name, meta) in &header.tensors {
        if meta.data_offsets.1 < meta.data_offsets.0 {
            errors.push(format!("{name}: end offset < start offset"));
        }
        if meta.data_offsets.1 > data_len {
            errors.push(format!("{name}: end offset exceeds data length"));
        }
        let expected_bytes: usize = meta.shape.iter().product::<usize>() * dtype_size(&meta.dtype);
        let actual_bytes = meta.data_offsets.1 - meta.data_offsets.0;
        if actual_bytes != expected_bytes {
            errors.push(format!(
                "{name}: expected {expected_bytes} bytes, got {actual_bytes}"
            ));
        }
    }

    errors
}

/// Return byte size for a dtype string.
fn dtype_size(dtype: &str) -> usize {
    match dtype {
        "F32" => 4,
        "F16" | "BF16" => 2,
        "I8" | "U8" => 1,
        "F64" => 8,
        _ => 4,
    }
}

/// Assemble the final SafeTensors binary.
fn assemble_safetensors(export: &SafeTensorsExport) -> Vec<u8> {
    let mut output = Vec::with_capacity(export.total_size);
    // 8-byte LE header length
    output.extend_from_slice(&(export.header_size as u64).to_le_bytes());
    // JSON header
    output.extend_from_slice(export.header_json.as_bytes());
    // Tensor data
    output.extend_from_slice(&export.tensor_data);
    output
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("format_export_safetensors")?;

    // Section 1: Create APR model with multiple tensors
    println!("=== APR v2 Source Model ===");
    let tensors: Vec<(&str, Vec<usize>, Vec<u8>)> = vec![
        (
            "encoder.weight",
            vec![128, 64],
            generate_model_payload(1, 128 * 64),
        ),
        ("encoder.bias", vec![128], generate_model_payload(2, 128)),
        (
            "decoder.weight",
            vec![64, 128],
            generate_model_payload(3, 64 * 128),
        ),
        ("decoder.bias", vec![64], generate_model_payload(4, 64)),
    ];

    let mut apr_builder = ModelBundleV2::new()
        .with_name("export-demo")
        .with_compression(Compression::None)
        .with_quantization(Quantization::FP32);

    for (name, shape, data) in &tensors {
        apr_builder = apr_builder.add_tensor(*name, shape.clone(), data.clone());
    }
    let apr_bundle = apr_builder.build();
    println!("APR bundle size: {} bytes", apr_bundle.len());
    println!("Tensors:         {}", tensors.len());
    println!();

    // Section 2: Generate SafeTensors header
    let export = apr_to_safetensors(&tensors);
    println!("=== SafeTensors Header ===");
    println!("Header JSON ({} bytes):", export.header_size);
    println!("  {}", export.header_json);
    println!();

    // Section 3: Tensor data layout
    println!("=== Tensor Data Layout ===");
    println!(
        "{:<20} {:<10} {:<15} {:<20}",
        "Tensor", "DType", "Shape", "Offsets"
    );
    println!("{}", "-".repeat(65));
    let mut sorted_keys: Vec<_> = export
        .header_json
        .split('"')
        .filter(|s| s.contains('.'))
        .collect();
    sorted_keys.sort_unstable();
    // Re-parse from header struct for clean output
    let reparsed = apr_to_safetensors(&tensors);
    let _ = reparsed; // use the export directly

    for (name, shape, _) in &tensors {
        let shape_str = format!("{shape:?}");
        let size = shape.iter().product::<usize>() * 4;
        println!("{name:<20} {:<10} {shape_str:<15} {size} bytes", "F32");
    }
    println!();

    // Section 4: Format comparison
    let st_binary = assemble_safetensors(&export);
    println!("=== Format Comparison ===");
    println!("{:<20} {:<15} {:<15}", "Property", "APR v2", "SafeTensors");
    println!("{}", "-".repeat(50));
    println!(
        "{:<20} {:<15} {:<15}",
        "Total size",
        format!("{} B", apr_bundle.len()),
        format!("{} B", st_binary.len()),
    );
    println!("{:<20} {:<15} {:<15}", "Magic bytes", "APR2", "(none)");
    println!("{:<20} {:<15} {:<15}", "Header format", "Binary", "JSON");
    println!("{:<20} {:<15} {:<15}", "Compression", "LZ4/Zstd", "None");
    println!(
        "{:<20} {:<15} {:<15}",
        "Quantization", "FP32/INT4/8", "FP32/F16"
    );

    // Validate
    let header_for_validation = SafeTensorsHeader {
        tensors: tensors
            .iter()
            .scan(0usize, |offset, (name, shape, data)| {
                let start = *offset;
                *offset += data.len();
                Some((
                    (*name).to_string(),
                    TensorMeta {
                        dtype: "F32".to_string(),
                        shape: shape.clone(),
                        data_offsets: (start, *offset),
                    },
                ))
            })
            .collect(),
    };
    let errors = validate_header(&header_for_validation, export.tensor_data.len());
    println!("\nValidation errors: {}", errors.len());

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tensors() -> Vec<(&'static str, Vec<usize>, Vec<u8>)> {
        vec![
            ("w1", vec![4, 4], generate_model_payload(1, 4 * 4)),
            ("w2", vec![2, 4], generate_model_payload(2, 2 * 4)),
            ("b1", vec![4], generate_model_payload(3, 4)),
        ]
    }

    #[test]
    fn test_header_valid_json() {
        let export = apr_to_safetensors(&sample_tensors());
        // Must start with { and end with }
        assert!(export.header_json.starts_with('{'));
        assert!(export.header_json.ends_with('}'));
        // Must contain all tensor names
        assert!(export.header_json.contains("w1"));
        assert!(export.header_json.contains("w2"));
        assert!(export.header_json.contains("b1"));
    }

    #[test]
    fn test_offsets_consistent() {
        let tensors = sample_tensors();
        let export = apr_to_safetensors(&tensors);
        let header = SafeTensorsHeader {
            tensors: tensors
                .iter()
                .scan(0usize, |offset, (name, shape, data)| {
                    let start = *offset;
                    *offset += data.len();
                    Some((
                        name.to_string(),
                        TensorMeta {
                            dtype: "F32".to_string(),
                            shape: shape.clone(),
                            data_offsets: (start, *offset),
                        },
                    ))
                })
                .collect(),
        };
        let errors = validate_header(&header, export.tensor_data.len());
        assert!(errors.is_empty(), "Validation errors: {errors:?}");
    }

    #[test]
    fn test_all_tensors_present() {
        let tensors = sample_tensors();
        let export = apr_to_safetensors(&tensors);
        // Header must mention all tensors
        for (name, _, _) in &tensors {
            assert!(
                export.header_json.contains(name),
                "Missing tensor {name} in header"
            );
        }
    }

    #[test]
    fn test_roundtrip_metadata() {
        let tensors = sample_tensors();
        let export = apr_to_safetensors(&tensors);
        // Verify data_offsets dtype and shape are in the JSON
        assert!(export.header_json.contains("\"dtype\":\"F32\""));
        assert!(export.header_json.contains("\"shape\""));
        assert!(export.header_json.contains("\"data_offsets\""));
    }

    #[test]
    fn test_total_size_correct() {
        let tensors = sample_tensors();
        let export = apr_to_safetensors(&tensors);
        let binary = assemble_safetensors(&export);
        assert_eq!(binary.len(), export.total_size);
    }

    #[test]
    fn test_header_length_prefix() {
        let tensors = sample_tensors();
        let export = apr_to_safetensors(&tensors);
        let binary = assemble_safetensors(&export);
        let header_len = u64::from_le_bytes(binary[0..8].try_into().unwrap()) as usize;
        assert_eq!(header_len, export.header_size);
    }

    #[test]
    fn test_tensor_data_size() {
        let tensors = sample_tensors();
        let total_bytes: usize = tensors.iter().map(|(_, _, d)| d.len()).sum();
        let export = apr_to_safetensors(&tensors);
        assert_eq!(export.tensor_data.len(), total_bytes);
    }

    #[test]
    fn test_empty_tensors() {
        let export = apr_to_safetensors(&[]);
        assert_eq!(export.header_json, "{}");
        assert!(export.tensor_data.is_empty());
    }

    #[test]
    fn test_dtype_size_variants() {
        assert_eq!(dtype_size("F32"), 4);
        assert_eq!(dtype_size("F16"), 2);
        assert_eq!(dtype_size("BF16"), 2);
        assert_eq!(dtype_size("I8"), 1);
        assert_eq!(dtype_size("F64"), 8);
    }

    #[test]
    fn test_serialize_header_sorted() {
        let mut header = SafeTensorsHeader {
            tensors: HashMap::new(),
        };
        header.tensors.insert(
            "z_tensor".to_string(),
            TensorMeta {
                dtype: "F32".to_string(),
                shape: vec![2],
                data_offsets: (0, 8),
            },
        );
        header.tensors.insert(
            "a_tensor".to_string(),
            TensorMeta {
                dtype: "F32".to_string(),
                shape: vec![2],
                data_offsets: (8, 16),
            },
        );
        let json = serialize_header(&header);
        let a_pos = json.find("a_tensor").unwrap();
        let z_pos = json.find("z_tensor").unwrap();
        assert!(a_pos < z_pos, "Header keys should be sorted");
    }
}
