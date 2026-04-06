#![allow(unused_imports)]
//! Mixed-Precision Training Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Demonstrates training with different numerical precisions:
//! FP32 (baseline), simulated FP16 with loss scaling, and simulated BF16.
//! Shows accuracy vs throughput tradeoffs for each precision level.
//!
//! # Precision Levels
//!
//! ```text
//! FP32:  32-bit float → baseline accuracy, slowest
//! FP16:  16-bit float → faster, needs loss scaling for stability
//! BF16:  16-bit bfloat → faster, wider range than FP16, no loss scaling needed
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example mixed_precision_training
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr finetune model.apr          # APR native format
//! apr finetune model.gguf         # GGUF (llama.cpp compatible)
//! apr finetune model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Mixed-Precision Training Example ===\n");

    let seed = 42;
    let train_data = generate_data(200, seed);
    let test_data = generate_data(50, seed + 100);

    section_precision_memory(seed);
    section_precision_casting();
    section_training_loop(seed, &train_data, &test_data);
    section_throughput_benchmark(seed, &train_data);
    section_loss_scaler_dynamics();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp32_cast_identity() {
        let values = [0.1, -0.5, 100.0, 0.0];
        for v in values {
            assert_eq!(Precision::FP32.cast(v), v);
        }
    }

    #[test]
    fn test_fp16_cast_clamps() {
        let result = Precision::FP16.cast(100_000.0);
        assert!(result <= 65504.0);
    }

    #[test]
    fn test_bf16_cast_reduces_precision() {
        let v = 0.123_456_789_f32;
        let cast = Precision::BF16.cast(v);
        // BF16 has 7-bit mantissa, so precision is reduced
        assert!((cast - v).abs() < 0.01);
        assert_ne!(cast, v);
    }

    #[test]
    fn test_model_forward_dimensions() {
        let model = MixedPrecisionModel::new(Precision::FP32, 42);
        let input = vec![0.5; INPUT_DIM];
        let output = model.forward(&input);
        assert_eq!(output.len(), OUTPUT_DIM);
    }

    #[test]
    fn test_cross_entropy_minimum_at_target() {
        let output = vec![10.0, 0.0, 0.0, 0.0];
        let loss_correct = cross_entropy(&output, 0);
        let loss_wrong = cross_entropy(&output, 1);
        assert!(loss_correct < loss_wrong);
    }

    #[test]
    fn test_loss_scaler_growth() {
        let mut scaler = LossScaler::new(1024.0);
        for _ in 0..100 {
            scaler.update(false);
        }
        assert!(scaler.scale > 1024.0);
    }

    #[test]
    fn test_loss_scaler_backoff_on_overflow() {
        let mut scaler = LossScaler::new(1024.0);
        scaler.update(true);
        assert!(scaler.scale < 1024.0);
        assert_eq!(scaler.overflow_count, 1);
    }

    #[test]
    fn test_predict_argmax() {
        assert_eq!(predict(&[0.1, 0.9, 0.3, 0.2]), 1);
        assert_eq!(predict(&[0.9, 0.1, 0.0, 0.0]), 0);
    }

    #[test]
    fn test_precision_bits() {
        assert_eq!(Precision::FP32.bits(), 32);
        assert_eq!(Precision::FP16.bits(), 16);
        assert_eq!(Precision::BF16.bits(), 16);
    }
}
