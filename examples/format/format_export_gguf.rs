//! # Export APR Model to GGUF Format
//!
//! **CLI equivalent:** `apr export model.apr --format gguf`
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/apr-format-roundtrip-v1.yaml
//!
//! Demonstrates exporting an APR v2 model to the GGUF (GPT-Generated
//! Unified Format) format. GGUF is used by llama.cpp and friends for
//! efficient quantized inference on CPUs.
//!
//! ## Sections
//! 1. GGUF format overview — magic bytes, version, header structure
//! 2. Metadata mapping — APR metadata to GGUF key-value pairs
//! 3. Tensor data conversion — APR tensors to GGUF tensor descriptors
//! 4. Quantization type mapping — APR quantization levels to GGUF types
//!
//!
//! ## Format Variants
//! ```bash
//! apr export model.apr          # APR native format
//! apr export model.gguf         # GGUF (llama.cpp compatible)
//! apr export model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP. DOI: 10.18653/v1/2020.emnlp-demos.6

use apr_cookbook::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// GGUF types
// ---------------------------------------------------------------------------

/// GGUF file magic bytes.
const GGUF_MAGIC: [u8; 4] = *b"GGUF";
/// Current GGUF version.
const GGUF_VERSION: u32 = 3;

/// GGUF file header.
#[derive(Debug)]
struct GgufHeader {
    magic: [u8; 4],
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
}

/// GGUF metadata value types.
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum GgufValue {
    Uint32(u32),
    Float32(f32),
    String(String),
    Uint64(u64),
}

/// GGUF tensor descriptor.
#[derive(Debug)]
struct GgufTensorInfo {
    name: String,
    n_dimensions: u32,
    dimensions: Vec<u64>,
    gguf_type: GgufQuantType,
    offset: u64,
}

/// GGUF quantization type codes (subset).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
#[allow(dead_code)]
enum GgufQuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 8,
}

// ---------------------------------------------------------------------------
// APR → GGUF conversion
// ---------------------------------------------------------------------------

/// Map APR Quantization level to GGUF quantization type.
fn apr_quant_to_gguf(quant: Quantization) -> GgufQuantType {
    match quant {
        Quantization::FP32 => GgufQuantType::F32,
        Quantization::FP16 => GgufQuantType::F16,
        Quantization::Int8 => GgufQuantType::Q8_0,
        Quantization::Int4 => GgufQuantType::Q4_0,
    }
}

/// Build GGUF metadata key-value pairs from APR model metadata.
fn build_gguf_metadata(
    model_name: &str,
    tensor_count: usize,
    quant: Quantization,
) -> HashMap<String, GgufValue> {
    let mut kv = HashMap::new();
    kv.insert(
        "general.architecture".to_string(),
        GgufValue::String("transformer".to_string()),
    );
    kv.insert(
        "general.name".to_string(),
        GgufValue::String(model_name.to_string()),
    );
    kv.insert(
        "general.file_type".to_string(),
        GgufValue::Uint32(apr_quant_to_gguf(quant) as u32),
    );
    kv.insert(
        "general.quantization_version".to_string(),
        GgufValue::Uint32(2),
    );
    kv.insert(
        "apr.source_format".to_string(),
        GgufValue::String("APR2".to_string()),
    );
    kv.insert(
        "apr.tensor_count".to_string(),
        GgufValue::Uint64(tensor_count as u64),
    );
    kv
}

/// Convert APR tensors to GGUF tensor descriptors.
fn build_gguf_tensors(
    tensors: &[(&str, Vec<usize>, Vec<u8>)],
    quant: Quantization,
) -> Vec<GgufTensorInfo> {
    let mut offset: u64 = 0;
    let gguf_type = apr_quant_to_gguf(quant);

    tensors
        .iter()
        .map(|(name, shape, data)| {
            let info = GgufTensorInfo {
                name: (*name).to_string(),
                n_dimensions: shape.len() as u32,
                dimensions: shape.iter().map(|&s| s as u64).collect(),
                gguf_type,
                offset,
            };
            offset += data.len() as u64;
            info
        })
        .collect()
}

/// Serialize a complete GGUF file from components.
fn serialize_gguf(
    header: &GgufHeader,
    metadata: &HashMap<String, GgufValue>,
    tensor_infos: &[GgufTensorInfo],
    tensor_data: &[u8],
) -> Vec<u8> {
    let mut output = Vec::new();

    // Magic
    output.extend_from_slice(&header.magic);
    // Version
    output.extend_from_slice(&header.version.to_le_bytes());
    // Tensor count
    output.extend_from_slice(&header.tensor_count.to_le_bytes());
    // Metadata KV count
    output.extend_from_slice(&header.metadata_kv_count.to_le_bytes());

    // Metadata KV pairs (simplified serialization)
    let mut sorted_keys: Vec<_> = metadata.keys().collect();
    sorted_keys.sort();
    for key in sorted_keys {
        let key_bytes = key.as_bytes();
        output.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
        output.extend_from_slice(key_bytes);
        match &metadata[key] {
            GgufValue::Uint32(v) => {
                output.extend_from_slice(&4u32.to_le_bytes()); // type tag
                output.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::Float32(v) => {
                output.extend_from_slice(&6u32.to_le_bytes());
                output.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::String(v) => {
                output.extend_from_slice(&8u32.to_le_bytes());
                let s_bytes = v.as_bytes();
                output.extend_from_slice(&(s_bytes.len() as u64).to_le_bytes());
                output.extend_from_slice(s_bytes);
            }
            GgufValue::Uint64(v) => {
                output.extend_from_slice(&5u32.to_le_bytes()); // type tag for u64
                output.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    // Tensor info section
    for info in tensor_infos {
        let name_bytes = info.name.as_bytes();
        output.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
        output.extend_from_slice(name_bytes);
        output.extend_from_slice(&info.n_dimensions.to_le_bytes());
        for &dim in &info.dimensions {
            output.extend_from_slice(&dim.to_le_bytes());
        }
        output.extend_from_slice(&(info.gguf_type as u32).to_le_bytes());
        output.extend_from_slice(&info.offset.to_le_bytes());
    }

    // Tensor data (aligned)
    let alignment = 32;
    let padding = (alignment - (output.len() % alignment)) % alignment;
    output.extend(std::iter::repeat(0u8).take(padding));
    output.extend_from_slice(tensor_data);

    output
}

/// Perform the full APR → GGUF conversion pipeline.
fn apr_to_gguf(
    model_name: &str,
    tensors: &[(&str, Vec<usize>, Vec<u8>)],
    quant: Quantization,
) -> Vec<u8> {
    let metadata = build_gguf_metadata(model_name, tensors.len(), quant);
    let tensor_infos = build_gguf_tensors(tensors, quant);
    let tensor_data: Vec<u8> = tensors
        .iter()
        .flat_map(|(_, _, d)| d.iter().copied())
        .collect();

    let header = GgufHeader {
        magic: GGUF_MAGIC,
        version: GGUF_VERSION,
        tensor_count: tensors.len() as u64,
        metadata_kv_count: metadata.len() as u64,
    };

    serialize_gguf(&header, &metadata, &tensor_infos, &tensor_data)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("format_export_gguf")?;

    // Section 1: GGUF format overview
    println!("=== GGUF Format Overview ===");
    println!(
        "Magic:   {:?} ({:?})",
        GGUF_MAGIC,
        std::str::from_utf8(&GGUF_MAGIC).unwrap()
    );
    println!("Version: {GGUF_VERSION}");
    println!("Used by: llama.cpp, whisper.cpp, ggml ecosystem");
    println!();

    // Create source APR model
    let tensors: Vec<(&str, Vec<usize>, Vec<u8>)> = vec![
        (
            "blk.0.attn.weight",
            vec![64, 64],
            generate_model_payload(1, 64 * 64),
        ),
        ("blk.0.attn.bias", vec![64], generate_model_payload(2, 64)),
        (
            "blk.0.ffn.weight",
            vec![256, 64],
            generate_model_payload(3, 256 * 64),
        ),
        ("blk.0.ffn.bias", vec![256], generate_model_payload(4, 256)),
        (
            "blk.1.attn.weight",
            vec![64, 64],
            generate_model_payload(5, 64 * 64),
        ),
        (
            "output.weight",
            vec![32, 64],
            generate_model_payload(6, 32 * 64),
        ),
    ];

    let apr_bundle = ModelBundleV2::new()
        .with_name("gguf-export-demo")
        .with_compression(Compression::None)
        .with_quantization(Quantization::FP32)
        .add_tensor(tensors[0].0, tensors[0].1.clone(), tensors[0].2.clone())
        .add_tensor(tensors[1].0, tensors[1].1.clone(), tensors[1].2.clone())
        .add_tensor(tensors[2].0, tensors[2].1.clone(), tensors[2].2.clone())
        .add_tensor(tensors[3].0, tensors[3].1.clone(), tensors[3].2.clone())
        .add_tensor(tensors[4].0, tensors[4].1.clone(), tensors[4].2.clone())
        .add_tensor(tensors[5].0, tensors[5].1.clone(), tensors[5].2.clone())
        .build();

    println!("=== Source APR Model ===");
    println!("APR bundle size: {} bytes", apr_bundle.len());
    println!("Tensors:         {}", tensors.len());
    println!();

    // Section 2: Metadata mapping
    let quant = Quantization::FP32;
    let metadata = build_gguf_metadata("gguf-export-demo", tensors.len(), quant);
    println!("=== GGUF Metadata ({} entries) ===", metadata.len());
    let mut keys: Vec<_> = metadata.keys().collect();
    keys.sort();
    for key in &keys {
        println!("  {key}: {:?}", metadata[*key]);
    }
    println!();

    // Section 3: Tensor data conversion
    let tensor_infos = build_gguf_tensors(&tensors, quant);
    println!("=== GGUF Tensor Descriptors ===");
    println!(
        "{:<25} {:<6} {:<15} {:<10} {:<10}",
        "Name", "Dims", "Shape", "Type", "Offset"
    );
    println!("{}", "-".repeat(66));
    for info in &tensor_infos {
        let shape_str = format!("{:?}", info.dimensions);
        println!(
            "{:<25} {:<6} {shape_str:<15} {:<10} {:<10}",
            info.name,
            info.n_dimensions,
            format!("{:?}", info.gguf_type),
            info.offset,
        );
    }
    println!();

    // Section 4: Quantization type mapping
    println!("=== Quantization Type Mapping ===");
    println!("{:<15} {:<15}", "APR Type", "GGUF Type");
    println!("{}", "-".repeat(30));
    for (apr_q, label) in [
        (Quantization::FP32, "FP32"),
        (Quantization::FP16, "FP16"),
        (Quantization::Int8, "INT8"),
        (Quantization::Int4, "INT4"),
    ] {
        let gguf_q = apr_quant_to_gguf(apr_q);
        println!("{label:<15} {:?}", gguf_q);
    }
    println!();

    // Final conversion
    let gguf_bytes = apr_to_gguf("gguf-export-demo", &tensors, quant);
    println!("=== Export Result ===");
    println!("GGUF file size:  {} bytes", gguf_bytes.len());
    println!(
        "Magic bytes:     {:?}",
        std::str::from_utf8(&gguf_bytes[0..4]).unwrap()
    );
    let version = u32::from_le_bytes(gguf_bytes[4..8].try_into().unwrap());
    println!("Version:         {version}");
    let tc = u64::from_le_bytes(gguf_bytes[8..16].try_into().unwrap());
    println!("Tensor count:    {tc}");

    assert_eq!(&gguf_bytes[0..4], b"GGUF");
    assert_eq!(version, GGUF_VERSION);
    assert_eq!(tc, tensors.len() as u64);

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
            ("w2", vec![8, 4], generate_model_payload(2, 8 * 4)),
            ("b1", vec![4], generate_model_payload(3, 4)),
        ]
    }

    #[test]
    fn test_magic_bytes_correct() {
        let gguf = apr_to_gguf("test", &sample_tensors(), Quantization::FP32);
        assert_eq!(&gguf[0..4], b"GGUF");
    }

    #[test]
    fn test_version_valid() {
        let gguf = apr_to_gguf("test", &sample_tensors(), Quantization::FP32);
        let version = u32::from_le_bytes(gguf[4..8].try_into().unwrap());
        assert_eq!(version, GGUF_VERSION);
    }

    #[test]
    fn test_tensor_count_matches() {
        let tensors = sample_tensors();
        let gguf = apr_to_gguf("test", &tensors, Quantization::FP32);
        let tc = u64::from_le_bytes(gguf[8..16].try_into().unwrap());
        assert_eq!(tc, tensors.len() as u64);
    }

    #[test]
    fn test_metadata_kv_count() {
        let gguf = apr_to_gguf("test", &sample_tensors(), Quantization::FP32);
        let kv_count = u64::from_le_bytes(gguf[16..24].try_into().unwrap());
        assert_eq!(kv_count, 6); // 6 metadata entries
    }

    #[test]
    fn test_quant_mapping_fp32() {
        assert_eq!(apr_quant_to_gguf(Quantization::FP32), GgufQuantType::F32);
    }

    #[test]
    fn test_quant_mapping_int4() {
        assert_eq!(apr_quant_to_gguf(Quantization::Int4), GgufQuantType::Q4_0);
    }

    #[test]
    fn test_quant_mapping_int8() {
        assert_eq!(apr_quant_to_gguf(Quantization::Int8), GgufQuantType::Q8_0);
    }

    #[test]
    fn test_quant_mapping_fp16() {
        assert_eq!(apr_quant_to_gguf(Quantization::FP16), GgufQuantType::F16);
    }

    #[test]
    fn test_metadata_contains_name() {
        let meta = build_gguf_metadata("my-model", 3, Quantization::FP32);
        match &meta["general.name"] {
            GgufValue::String(s) => assert_eq!(s, "my-model"),
            other => panic!("Expected String, got {other:?}"),
        }
    }

    #[test]
    fn test_tensor_infos_offsets_sequential() {
        let tensors = sample_tensors();
        let infos = build_gguf_tensors(&tensors, Quantization::FP32);
        let mut expected_offset: u64 = 0;
        for (i, info) in infos.iter().enumerate() {
            assert_eq!(info.offset, expected_offset, "Tensor {i} offset mismatch");
            expected_offset += tensors[i].2.len() as u64;
        }
    }

    #[test]
    fn test_empty_tensors() {
        let gguf = apr_to_gguf("empty", &[], Quantization::FP32);
        assert_eq!(&gguf[0..4], b"GGUF");
        let tc = u64::from_le_bytes(gguf[8..16].try_into().unwrap());
        assert_eq!(tc, 0);
    }

    #[test]
    fn test_gguf_output_non_empty() {
        let gguf = apr_to_gguf("test", &sample_tensors(), Quantization::FP32);
        assert!(gguf.len() > 24); // header alone is 24 bytes
    }
}
