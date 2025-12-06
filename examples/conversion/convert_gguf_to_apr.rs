//! GGUF to APR format conversion.
//!
//! This example demonstrates converting GGUF models (llama.cpp format)
//! to native APR format for use with the Sovereign AI Stack.
//!
//! # Run
//!
//! ```bash
//! cargo run --example convert_gguf_to_apr
//! ```
//!
//! # Why Import from GGUF?
//!
//! GGUF is the de-facto standard for quantized LLMs:
//! - Thousands of models on Hugging Face
//! - Ollama model library
//! - TheBloke quantizations
//!
//! Converting to APR enables:
//! - Native Rust inference (no C++ deps)
//! - WASM deployment
//! - Integration with trueno SIMD
//! - Encryption and signing

use apr_cookbook::convert::{
    AprConverter, ConversionFormat, ConversionMetadata, DataType, TensorData,
};
use apr_cookbook::Result;
use std::io::Cursor;

/// GGUF magic number: "GGUF" in little-endian
const GGUF_MAGIC: u32 = 0x4655_4747;

/// GGUF format version
const GGUF_VERSION: u32 = 3;

/// GGML tensor type to APR DataType mapping
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
#[allow(dead_code)]
enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 8,
    I8 = 24,
    I16 = 25,
    I32 = 26,
}

impl GgmlType {
    #[allow(dead_code)]
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            8 => Some(Self::Q8_0),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            _ => None,
        }
    }

    fn to_apr_dtype(self) -> DataType {
        match self {
            Self::F32 | Self::I32 => DataType::F32,
            Self::F16 | Self::I16 => DataType::F16,
            Self::Q4_0 | Self::Q4_1 => DataType::Q4_0,
            Self::Q8_0 | Self::I8 => DataType::Q8_0,
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q8_0 => "Q8_0",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
        }
    }
}

/// Simulated GGUF reader for demonstration.
///
/// In production, you would use a proper GGUF parser or implement
/// the full GGUF specification reading.
struct GgufReader {
    magic: u32,
    version: u32,
    tensor_count: u64,
    metadata_count: u64,
    metadata: Vec<(String, String)>,
    tensors: Vec<GgufTensorInfo>,
}

/// Tensor metadata from GGUF
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct GgufTensorInfo {
    name: String,
    n_dims: u32,
    dims: Vec<u64>,
    dtype: GgmlType,
    offset: u64,
}

impl GgufReader {
    /// Create a GGUF reader from mock data (demonstration purposes)
    #[allow(dead_code)]
    fn from_mock_bytes(data: &[u8]) -> Result<Self> {
        use std::io::Read;

        let mut cursor = Cursor::new(data);
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        // Read magic
        cursor.read_exact(&mut buf4).map_err(|e| {
            apr_cookbook::CookbookError::invalid_format(format!("Failed to read magic: {}", e))
        })?;
        let magic = u32::from_le_bytes(buf4);

        if magic != GGUF_MAGIC {
            return Err(apr_cookbook::CookbookError::invalid_format(format!(
                "Invalid GGUF magic: 0x{:08X}, expected 0x{:08X}",
                magic, GGUF_MAGIC
            )));
        }

        // Read version
        cursor.read_exact(&mut buf4).map_err(|e| {
            apr_cookbook::CookbookError::invalid_format(format!("Failed to read version: {}", e))
        })?;
        let version = u32::from_le_bytes(buf4);

        // Read tensor count
        cursor.read_exact(&mut buf8).map_err(|e| {
            apr_cookbook::CookbookError::invalid_format(format!(
                "Failed to read tensor count: {}",
                e
            ))
        })?;
        let tensor_count = u64::from_le_bytes(buf8);

        // Read metadata count
        cursor.read_exact(&mut buf8).map_err(|e| {
            apr_cookbook::CookbookError::invalid_format(format!(
                "Failed to read metadata count: {}",
                e
            ))
        })?;
        let metadata_count = u64::from_le_bytes(buf8);

        Ok(Self {
            magic,
            version,
            tensor_count,
            metadata_count,
            metadata: Vec::new(),
            tensors: Vec::new(),
        })
    }

    /// Create a populated mock reader for demonstration
    fn mock_llama_model() -> Self {
        let tensors = vec![
            GgufTensorInfo {
                name: "token_embd.weight".to_string(),
                n_dims: 2,
                dims: vec![32000, 4096],
                dtype: GgmlType::Q8_0,
                offset: 0,
            },
            GgufTensorInfo {
                name: "blk.0.attn_q.weight".to_string(),
                n_dims: 2,
                dims: vec![4096, 4096],
                dtype: GgmlType::Q4_0,
                offset: 0,
            },
            GgufTensorInfo {
                name: "blk.0.attn_k.weight".to_string(),
                n_dims: 2,
                dims: vec![4096, 1024],
                dtype: GgmlType::Q4_0,
                offset: 0,
            },
            GgufTensorInfo {
                name: "blk.0.attn_v.weight".to_string(),
                n_dims: 2,
                dims: vec![4096, 1024],
                dtype: GgmlType::Q4_0,
                offset: 0,
            },
            GgufTensorInfo {
                name: "blk.0.attn_output.weight".to_string(),
                n_dims: 2,
                dims: vec![4096, 4096],
                dtype: GgmlType::Q4_0,
                offset: 0,
            },
            GgufTensorInfo {
                name: "output_norm.weight".to_string(),
                n_dims: 1,
                dims: vec![4096],
                dtype: GgmlType::F32,
                offset: 0,
            },
            GgufTensorInfo {
                name: "output.weight".to_string(),
                n_dims: 2,
                dims: vec![32000, 4096],
                dtype: GgmlType::Q8_0,
                offset: 0,
            },
        ];

        let metadata = vec![
            ("general.architecture".to_string(), "llama".to_string()),
            ("general.name".to_string(), "llama-7b-q4_0".to_string()),
            ("llama.context_length".to_string(), "4096".to_string()),
            ("llama.embedding_length".to_string(), "4096".to_string()),
            ("llama.block_count".to_string(), "32".to_string()),
            ("llama.attention.head_count".to_string(), "32".to_string()),
            ("llama.attention.head_count_kv".to_string(), "8".to_string()),
            ("general.quantization_version".to_string(), "2".to_string()),
        ];

        Self {
            magic: GGUF_MAGIC,
            version: GGUF_VERSION,
            tensor_count: tensors.len() as u64,
            metadata_count: metadata.len() as u64,
            metadata,
            tensors,
        }
    }

    /// Get the model architecture
    fn architecture(&self) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == "general.architecture")
            .map(|(_, v)| v.as_str())
    }

    /// Get the model name
    fn model_name(&self) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == "general.name")
            .map(|(_, v)| v.as_str())
    }

    /// Calculate total parameters
    fn total_params(&self) -> u64 {
        self.tensors
            .iter()
            .map(|t| t.dims.iter().product::<u64>())
            .sum()
    }
}

fn main() -> Result<()> {
    println!("=== APR Cookbook: GGUF → APR Conversion ===\n");

    // Check conversion is supported
    let supported =
        AprConverter::is_conversion_supported(ConversionFormat::Gguf, ConversionFormat::Apr);
    println!("Conversion supported: {}\n", supported);

    // Create mock GGUF data (simulating reading a file)
    println!("Loading mock GGUF model (simulating file read)...");
    let reader = GgufReader::mock_llama_model();

    println!("\nGGUF File Info:");
    println!("  Magic: 0x{:08X}", reader.magic);
    println!("  Version: {}", reader.version);
    println!("  Tensors: {}", reader.tensor_count);
    println!("  Metadata entries: {}", reader.metadata_count);
    println!("  Architecture: {:?}", reader.architecture());
    println!("  Model name: {:?}", reader.model_name());
    println!("  Total parameters: {}", reader.total_params());

    // Display metadata
    println!("\nMetadata:");
    for (key, value) in &reader.metadata {
        println!("  {}: {}", key, value);
    }

    // Display tensors
    println!("\nTensors:");
    for tensor in &reader.tensors {
        let params: u64 = tensor.dims.iter().product();
        println!(
            "  {} [{:?}] {} - {} params",
            tensor.name,
            tensor.dims,
            tensor.dtype.display_name(),
            params
        );
    }

    // Create APR converter
    println!("\nConverting to APR format...");
    let mut converter = AprConverter::new();

    // Set metadata
    converter.set_metadata(ConversionMetadata {
        name: reader.model_name().map(String::from),
        architecture: reader.architecture().map(String::from),
        source_format: Some(ConversionFormat::Gguf),
        ..Default::default()
    });

    // Convert tensors
    for gguf_tensor in &reader.tensors {
        // In production, you would read the actual tensor data from the file
        let shape: Vec<usize> = gguf_tensor.dims.iter().map(|&d| d as usize).collect();
        let num_elements: usize = shape.iter().product();
        let dtype = gguf_tensor.dtype.to_apr_dtype();
        let elem_size = dtype.element_size();

        let tensor = TensorData {
            name: gguf_tensor.name.clone(),
            shape,
            dtype,
            data: vec![0u8; num_elements * elem_size], // Placeholder data
        };

        converter.add_tensor(tensor);
    }

    // Generate APR output
    let apr_bytes = converter.to_apr()?;

    println!("\nConversion Summary:");
    println!("  Input: GGUF ({} tensors)", reader.tensor_count);
    println!("  Output: APR ({} bytes)", apr_bytes.len());
    println!("  Tensors converted: {}", converter.tensor_count());
    println!("  Total parameters: {}", converter.total_parameters());

    // Verify APR header
    assert_eq!(&apr_bytes[0..4], b"APRN", "APR magic should be present");
    println!("\n  ✓ APR header verified");

    println!("\n[SUCCESS] GGUF → APR conversion complete!");
    println!("\n=== Benefits of APR Format ===");
    println!("  • Pure Rust (no C++ dependencies)");
    println!("  • WASM deployment ready");
    println!("  • Native trueno SIMD acceleration");
    println!("  • Optional encryption (AES-256-GCM)");
    println!("  • Optional signing (Ed25519)");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_reader_from_mock_bytes() {
        // Create minimal valid GGUF header
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(0u64).to_le_bytes()); // tensor count
        bytes.extend_from_slice(&(0u64).to_le_bytes()); // metadata count

        let reader = GgufReader::from_mock_bytes(&bytes).unwrap();
        assert_eq!(reader.magic, GGUF_MAGIC);
        assert_eq!(reader.version, GGUF_VERSION);
    }

    #[test]
    fn test_invalid_magic_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x12345678u32.to_le_bytes());
        bytes.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(0u64).to_le_bytes());
        bytes.extend_from_slice(&(0u64).to_le_bytes());

        let result = GgufReader::from_mock_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_llama_model() {
        let reader = GgufReader::mock_llama_model();
        assert_eq!(reader.architecture(), Some("llama"));
        assert!(reader.total_params() > 0);
    }

    #[test]
    fn test_ggml_type_conversion() {
        assert!(matches!(GgmlType::F32.to_apr_dtype(), DataType::F32));
        assert!(matches!(GgmlType::F16.to_apr_dtype(), DataType::F16));
        assert!(matches!(GgmlType::Q4_0.to_apr_dtype(), DataType::Q4_0));
        assert!(matches!(GgmlType::Q8_0.to_apr_dtype(), DataType::Q8_0));
    }

    #[test]
    fn test_full_conversion_pipeline() {
        let reader = GgufReader::mock_llama_model();
        let mut converter = AprConverter::new();

        for gguf_tensor in &reader.tensors {
            let shape: Vec<usize> = gguf_tensor.dims.iter().map(|&d| d as usize).collect();
            let dtype = gguf_tensor.dtype.to_apr_dtype();
            let elem_size = dtype.element_size();
            let num_elements: usize = shape.iter().product();

            let tensor = TensorData {
                name: gguf_tensor.name.clone(),
                shape,
                dtype,
                data: vec![0u8; num_elements * elem_size],
            };
            converter.add_tensor(tensor);
        }

        let apr_bytes = converter.to_apr().unwrap();
        assert_eq!(&apr_bytes[0..4], b"APRN");
    }

    #[test]
    fn test_conversion_path_supported() {
        assert!(AprConverter::is_conversion_supported(
            ConversionFormat::Gguf,
            ConversionFormat::Apr
        ));
    }
}
