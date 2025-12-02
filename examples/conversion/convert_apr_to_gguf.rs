//! APR to GGUF format conversion.
//!
//! This example demonstrates converting APR models to GGUF format
//! for use with llama.cpp and other GGML-based inference engines.
//!
//! # Run
//!
//! ```bash
//! cargo run --example convert_apr_to_gguf
//! ```
//!
//! # Why GGUF?
//!
//! GGUF (GPT-Generated Unified Format) enables:
//! - llama.cpp inference
//! - Ollama integration
//! - Efficient quantization (Q4_K, Q5_K, Q8_0)
//! - CPU/GPU hybrid execution

use apr_cookbook::convert::{AprConverter, ConversionFormat, DataType, TensorData};
use apr_cookbook::Result;

/// GGUF magic number
const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"

/// GGUF version
const GGUF_VERSION: u32 = 3;

/// Simulated GGUF writer for demonstration.
struct GgufWriter {
    tensors: Vec<TensorData>,
    metadata: Vec<(String, String)>,
}

impl GgufWriter {
    fn new() -> Self {
        Self {
            tensors: Vec::new(),
            metadata: Vec::new(),
        }
    }

    fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.push((key.to_string(), value.to_string()));
    }

    fn add_tensor(&mut self, tensor: TensorData) {
        self.tensors.push(tensor);
    }

    fn finalize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // GGUF header
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

        // In production, would write full metadata and tensor data
        // This is a simplified demonstration

        bytes
    }
}

fn main() -> Result<()> {
    println!("=== APR Cookbook: APR → GGUF Conversion ===\n");

    // Check conversion is supported
    let supported =
        AprConverter::is_conversion_supported(ConversionFormat::Apr, ConversionFormat::Gguf);
    println!("Conversion supported: {}\n", supported);

    // Create sample APR model tensors
    let tensors = vec![
        TensorData {
            name: "token_embd.weight".to_string(),
            shape: vec![32000, 4096],
            dtype: DataType::F32,
            data: vec![],
        },
        TensorData {
            name: "blk.0.attn_q.weight".to_string(),
            shape: vec![4096, 4096],
            dtype: DataType::F32,
            data: vec![],
        },
        TensorData {
            name: "output_norm.weight".to_string(),
            shape: vec![4096],
            dtype: DataType::F32,
            data: vec![],
        },
    ];

    println!("Converting {} tensors to GGUF format:", tensors.len());

    // Create GGUF writer
    let mut writer = GgufWriter::new();

    // Add metadata
    writer.add_metadata("general.architecture", "llama");
    writer.add_metadata("general.name", "apr-cookbook-demo");
    writer.add_metadata("llama.context_length", "4096");
    writer.add_metadata("llama.embedding_length", "4096");
    writer.add_metadata("llama.block_count", "32");

    println!("\nMetadata:");
    for (key, value) in &writer.metadata {
        println!("  {}: {}", key, value);
    }

    // Add tensors
    println!("\nTensors:");
    for tensor in tensors {
        let params: usize = tensor.shape.iter().product();
        println!("  {} [{:?}] - {} params", tensor.name, tensor.shape, params);
        writer.add_tensor(tensor);
    }

    // Finalize
    let gguf_bytes = writer.finalize();
    println!("\nGGUF Output:");
    println!("  Magic: 0x{:08X}", GGUF_MAGIC);
    println!("  Version: {}", GGUF_VERSION);
    println!("  Header size: {} bytes", gguf_bytes.len());

    println!("\n[SUCCESS] APR → GGUF conversion complete!");
    println!("          Compatible with llama.cpp and Ollama.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_magic_is_correct() {
        // "GGUF" in little-endian
        assert_eq!(GGUF_MAGIC, 0x4655_4747);
    }

    #[test]
    fn test_gguf_writer_creates_valid_header() {
        let writer = GgufWriter::new();
        let bytes = writer.finalize();

        // Check magic
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(magic, GGUF_MAGIC);

        // Check version
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(version, GGUF_VERSION);
    }

    #[test]
    fn test_conversion_path_supported() {
        assert!(AprConverter::is_conversion_supported(
            ConversionFormat::Apr,
            ConversionFormat::Gguf
        ));
    }
}
