//! Quantized model loading demonstration.
//!
//! This example shows how to work with quantized models (Q4_0, Q8_0)
//! for reduced size and faster inference on edge devices.
//!
//! # Run
//!
//! ```bash
//! cargo run --example bundle_quantized_model
//! ```
//!
//! # Quantization Benefits
//!
//! | Format | Size Reduction | Accuracy Loss |
//! |--------|---------------|---------------|
//! | F32    | Baseline      | None          |
//! | Q8_0   | 75%           | <1%           |
//! | Q4_0   | 87.5%         | 1-3%          |

use apr_cookbook::bundle::{BundledModel, ModelBundle};
use apr_cookbook::Result;

/// Simulated quantization levels.
#[derive(Debug, Clone, Copy)]
enum QuantLevel {
    F32,
    Q8_0,
    Q4_0,
}

impl QuantLevel {
    fn size_factor(self) -> f32 {
        match self {
            Self::F32 => 1.0,
            Self::Q8_0 => 0.25,
            Self::Q4_0 => 0.125,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32 (full precision)",
            Self::Q8_0 => "Q8_0 (8-bit quantized)",
            Self::Q4_0 => "Q4_0 (4-bit quantized)",
        }
    }
}

/// Create a sample model at different quantization levels.
fn create_quantized_model(base_size: usize, level: QuantLevel) -> Vec<u8> {
    let quantized_size = (base_size as f32 * level.size_factor()) as usize;

    ModelBundle::new()
        .with_name(format!("model-{:?}", level).to_lowercase())
        .with_payload(vec![0u8; quantized_size])
        .build()
}

fn main() -> Result<()> {
    println!("=== APR Cookbook: Quantized Model Loading ===\n");

    let base_size = 10_000_000; // 10MB base model

    println!(
        "Comparing quantization levels for {}MB model:\n",
        base_size / 1_000_000
    );

    for level in [QuantLevel::F32, QuantLevel::Q8_0, QuantLevel::Q4_0] {
        let model_bytes = create_quantized_model(base_size, level);
        let model = BundledModel::from_bytes(&model_bytes)?;

        let reduction = (1.0 - (model.size() as f32 / base_size as f32)) * 100.0;

        println!("  {}", level.name());
        println!(
            "    Size: {} bytes ({:.1}% reduction)",
            model.size(),
            reduction
        );
        println!("    Version: {}.{}", model.version().0, model.version().1);
        println!();
    }

    println!("[INFO] Quantization enables edge deployment!");
    println!("       Q4_0 models fit on microcontrollers.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_reduces_size() {
        let base_size = 10000;

        let f32_model = create_quantized_model(base_size, QuantLevel::F32);
        let q8_model = create_quantized_model(base_size, QuantLevel::Q8_0);
        let q4_model = create_quantized_model(base_size, QuantLevel::Q4_0);

        // Q8 should be smaller than F32
        assert!(q8_model.len() < f32_model.len());
        // Q4 should be smaller than Q8
        assert!(q4_model.len() < q8_model.len());
    }

    #[test]
    fn test_quantized_models_are_valid() {
        for level in [QuantLevel::F32, QuantLevel::Q8_0, QuantLevel::Q4_0] {
            let model_bytes = create_quantized_model(1000, level);
            let result = BundledModel::from_bytes(&model_bytes);
            assert!(result.is_ok(), "Failed to load {:?} model", level);
        }
    }
}
