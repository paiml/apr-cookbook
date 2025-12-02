//! SafeTensors to APR format conversion.
//!
//! This example demonstrates converting HuggingFace SafeTensors
//! models to the native APR format.
//!
//! # Run
//!
//! ```bash
//! cargo run --example convert_safetensors_to_apr
//! ```
//!
//! # Why Convert?
//!
//! SafeTensors is the HuggingFace standard, but APR offers:
//! - Built-in compression (zstd)
//! - Encryption (AES-256-GCM)
//! - Digital signatures (Ed25519)
//! - Quantization (Q4_0, Q8_0)

use apr_cookbook::convert::{
    AprConverter, ConversionFormat, ConversionMetadata, DataType, TensorData,
};
use apr_cookbook::Result;

/// Simulated SafeTensors loading.
///
/// In production, you would use:
/// ```ignore
/// let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
/// ```
fn load_mock_safetensors() -> Vec<TensorData> {
    vec![
        TensorData {
            name: "model.embed_tokens.weight".to_string(),
            shape: vec![32000, 4096],
            dtype: DataType::F16,
            data: vec![0u8; 32000 * 4096 * 2], // F16 = 2 bytes
        },
        TensorData {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            shape: vec![4096, 4096],
            dtype: DataType::F16,
            data: vec![0u8; 4096 * 4096 * 2],
        },
        TensorData {
            name: "model.layers.0.self_attn.k_proj.weight".to_string(),
            shape: vec![4096, 4096],
            dtype: DataType::F16,
            data: vec![0u8; 4096 * 4096 * 2],
        },
    ]
}

fn main() -> Result<()> {
    println!("=== APR Cookbook: SafeTensors → APR Conversion ===\n");

    // Check conversion is supported
    let supported =
        AprConverter::is_conversion_supported(ConversionFormat::SafeTensors, ConversionFormat::Apr);
    println!("Conversion supported: {}\n", supported);

    // Load mock SafeTensors data
    let tensors = load_mock_safetensors();
    println!("Loaded {} tensors from SafeTensors", tensors.len());

    // Create converter
    let mut converter = AprConverter::new();

    // Set metadata
    converter.set_metadata(ConversionMetadata {
        name: Some("llama-7b-converted".to_string()),
        architecture: Some("LlamaForCausalLM".to_string()),
        source_format: Some(ConversionFormat::SafeTensors),
        ..Default::default()
    });

    // Add tensors
    for tensor in tensors {
        println!(
            "  Adding: {} [{:?}] {:?}",
            tensor.name, tensor.shape, tensor.dtype
        );
        converter.add_tensor(tensor);
    }

    // Summary
    println!("\nConversion Summary:");
    println!("  Tensors: {}", converter.tensor_count());
    println!("  Total parameters: {}", converter.total_parameters());

    // Convert to APR
    let apr_bytes = converter.to_apr()?;
    println!("  APR size: {} bytes", apr_bytes.len());

    println!("\n[SUCCESS] Conversion complete!");
    println!("          Output would be saved to: model.apr");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_safetensors_loads() {
        let tensors = load_mock_safetensors();
        assert!(!tensors.is_empty());
    }

    #[test]
    fn test_conversion_produces_valid_apr() {
        let tensors = load_mock_safetensors();
        let mut converter = AprConverter::new();

        for tensor in tensors {
            converter.add_tensor(tensor);
        }

        let apr_bytes = converter.to_apr().unwrap();

        // Should start with APR magic
        assert_eq!(&apr_bytes[0..4], b"APRN");
    }
}
