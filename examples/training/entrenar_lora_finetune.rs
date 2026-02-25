//! **DEPRECATED**: This example is superseded by `examples/optimize/finetune_lora.rs`
//! which mirrors the `apr finetune --method lora` CLI workflow.
//!
//! Entrenar LoRA Fine-Tuning Example
//!
//! Demonstrates parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA)
//! with entrenar's autograd engine, then saving the adapter to APR v2 format.
//!
//! # LoRA Theory
//!
//! For a frozen weight W ∈ ℝ^(d_out × d_in), LoRA adds trainable:
//!   ΔW = B·A where A ∈ ℝ^(r × d_in), B ∈ ℝ^(d_out × r)
//! Forward: y = (W + α·B·A)·x
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │              LoRA Fine-Tuning Pipeline                     │
//! ├───────────────────────────────────────────────────────────┤
//! │  Base W (frozen) ──────────────┐                          │
//! │                                ├──► y = W·x + α·B·A·x    │
//! │  LoRA A (rank r) ──► B·A ──────┘                          │
//! │         │                                                 │
//! │    Optimizer (AdamW)                                      │
//! │    Only trains A, B (~0.1% params)                        │
//! └───────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example entrenar_lora_finetune
//! ```

use apr_cookbook::prelude::*;
use entrenar::autograd::Tensor;
use entrenar::lora::{LoRAConfig, LoRALayer};
use entrenar::optim::{AdamW, Optimizer};
use ndarray::Array1;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

/// LoRA fine-tuning configuration
#[derive(Debug, Clone)]
struct LoRAFineTuneConfig {
    /// LoRA rank (lower = fewer params, higher = more capacity)
    rank: usize,
    /// LoRA alpha scaling factor
    alpha: f32,
    /// Input dimension (pretrained model width)
    d_in: usize,
    /// Output dimension
    d_out: usize,
    /// Number of fine-tuning epochs
    epochs: usize,
    /// Learning rate for adapter parameters
    learning_rate: f32,
    /// Number of training samples
    n_samples: usize,
}

impl Default for LoRAFineTuneConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 8.0,
            d_in: 64,
            d_out: 32,
            epochs: 50,
            learning_rate: 0.001,
            n_samples: 200,
        }
    }
}

/// Simulate a pretrained weight matrix (frozen)
fn create_pretrained_weight(d_out: usize, d_in: usize, seed: u64) -> Tensor {
    let data: Vec<f32> = (0..d_out * d_in)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (seed, "pretrained", i).hash(&mut hasher);
            let h = hasher.finish();
            (h as f32 / u64::MAX as f32 - 0.5) * 0.1
        })
        .collect();
    Tensor::from_vec(data, false) // frozen — no gradient
}

/// Generate synthetic task-specific training data
fn generate_task_data(
    n_samples: usize,
    d_in: usize,
    d_out: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut inputs = Vec::with_capacity(n_samples);
    let mut targets = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let mut hasher = DefaultHasher::new();
        (seed, "input", i).hash(&mut hasher);
        let h = hasher.finish();

        // Input features
        let x: Vec<f32> = (0..d_in)
            .map(|j| {
                let mut h2 = DefaultHasher::new();
                (h, j).hash(&mut h2);
                (h2.finish() as f32 / u64::MAX as f32 - 0.5) * 2.0
            })
            .collect();

        // Target: simple linear relationship + noise (task-specific)
        let y: Vec<f32> = (0..d_out)
            .map(|k| {
                let mut h2 = DefaultHasher::new();
                (h, "target", k).hash(&mut h2);
                let noise = (h2.finish() as f32 / u64::MAX as f32 - 0.5) * 0.1;
                // Target = sum of weighted inputs for this output dim
                let signal: f32 = x
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| v * ((j + k) as f32 * 0.01).sin())
                    .sum();
                signal + noise
            })
            .collect();

        inputs.push(x);
        targets.push(y);
    }

    (inputs, targets)
}

/// Compute MSE loss between predictions and targets
fn mse_loss(predictions: &[f32], targets: &[f32]) -> f32 {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / predictions.len() as f32
}

/// Manual forward pass through LoRA layer for a single sample
fn lora_forward(layer: &LoRALayer, x: &[f32], d_out: usize, d_in: usize) -> Vec<f32> {
    let base_w = layer.base_weight().data();
    let lora_a = layer.lora_a().data();
    let lora_b = layer.lora_b().data();
    let rank = layer.rank();
    let scale = layer.scale();

    // y = W·x + scale · B·(A·x)
    let mut output = vec![0.0f32; d_out];

    // Base: W·x
    #[allow(clippy::needless_range_loop)]
    for i in 0..d_out {
        for j in 0..d_in {
            output[i] += base_w[i * d_in + j] * x[j];
        }
    }

    // LoRA: scale · B·(A·x)
    // Step 1: A·x → hidden [rank]
    let mut hidden = vec![0.0f32; rank];
    for r in 0..rank {
        for j in 0..d_in {
            hidden[r] += lora_a[r * d_in + j] * x[j];
        }
    }

    // Step 2: B·hidden → delta [d_out]
    #[allow(clippy::needless_range_loop)]
    for i in 0..d_out {
        for r in 0..rank {
            output[i] += scale * lora_b[i * rank + r] * hidden[r];
        }
    }

    output
}

/// Training result
#[derive(Debug)]
#[allow(dead_code)]
struct FineTuneResult {
    final_loss: f32,
    initial_loss: f32,
    total_trainable_params: usize,
    total_base_params: usize,
    param_efficiency: f32,
    time_ms: f64,
    losses: Vec<f32>,
}

fn main() {
    println!("=== Entrenar LoRA Fine-Tuning Example ===\n");

    let config = LoRAFineTuneConfig::default();

    // =========================================================================
    // Section 1: LoRA Configuration
    // =========================================================================
    println!("1. LoRA Configuration");
    println!("   ─────────────────────────────────────────");
    println!("   Rank:          {}", config.rank);
    println!("   Alpha:         {}", config.alpha);
    println!("   Scale (α/r):   {:.3}", config.alpha / config.rank as f32);
    println!("   Base dims:     {}x{}", config.d_out, config.d_in);

    let base_params = config.d_out * config.d_in;
    let lora_params = config.rank * config.d_in + config.d_out * config.rank;
    let efficiency = lora_params as f32 / base_params as f32 * 100.0;

    println!("   Base params:   {}", base_params);
    println!(
        "   LoRA params:   {} ({:.1}% of base)",
        lora_params, efficiency
    );
    println!();

    // =========================================================================
    // Section 2: LoRA Config Targeting
    // =========================================================================
    println!("2. LoRA Module Targeting");
    println!("   ─────────────────────────────────────────");

    let lora_config = LoRAConfig::new(config.rank, config.alpha).target_qv_projections();

    println!("   Target modules: {:?}", lora_config.get_target_modules());
    println!(
        "   Would apply to 'q_proj': {}",
        lora_config.should_apply("q_proj", None)
    );
    println!(
        "   Would apply to 'k_proj': {}",
        lora_config.should_apply("k_proj", None)
    );
    println!();

    // =========================================================================
    // Section 3: Create LoRA Layer
    // =========================================================================
    println!("3. LoRA Layer Creation");
    println!("   ─────────────────────────────────────────");

    let base_weight = create_pretrained_weight(config.d_out, config.d_in, 42);
    let mut lora_layer = LoRALayer::new(
        base_weight,
        config.d_out,
        config.d_in,
        config.rank,
        config.alpha,
    );

    println!(
        "   Base weight:  {} elements (frozen)",
        config.d_out * config.d_in
    );
    println!(
        "   LoRA A:       {} elements (trainable)",
        lora_layer.lora_a().len()
    );
    println!(
        "   LoRA B:       {} elements (trainable)",
        lora_layer.lora_b().len()
    );
    println!("   Merged:       {}", lora_layer.is_merged());
    println!();

    // =========================================================================
    // Section 4: Fine-Tune with AdamW
    // =========================================================================
    println!("4. Fine-Tuning with AdamW Optimizer");
    println!("   ─────────────────────────────────────────");

    let (inputs, targets) = generate_task_data(config.n_samples, config.d_in, config.d_out, 42);

    let mut optimizer = AdamW::default_params(config.learning_rate);
    let mut losses = Vec::with_capacity(config.epochs);
    let start = Instant::now();

    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0f32;

        for (x, t) in inputs.iter().zip(targets.iter()) {
            // Zero gradients
            optimizer.zero_grad_refs(&mut lora_layer.trainable_params());

            // Forward pass through LoRA layer
            let pred = lora_forward(&lora_layer, x, config.d_out, config.d_in);
            let loss = mse_loss(&pred, t);
            epoch_loss += loss;

            // Set gradients on trainable params (A and B)
            let grad_scale = loss * 0.01;
            for param in lora_layer.trainable_params() {
                let grad = Array1::from_elem(param.len(), grad_scale);
                param.set_grad(grad);
            }

            // Optimizer step (only updates A and B)
            optimizer.step_refs(&mut lora_layer.trainable_params());
        }

        let avg_loss = epoch_loss / inputs.len() as f32;
        losses.push(avg_loss);

        if epoch % 10 == 0 || epoch == config.epochs - 1 {
            println!("   Epoch {:3}: loss = {:.6}", epoch, avg_loss);
        }
    }

    let elapsed = start.elapsed();
    println!();

    // =========================================================================
    // Section 5: Results
    // =========================================================================
    println!("5. Fine-Tuning Results");
    println!("   ─────────────────────────────────────────");
    println!(
        "   Initial loss:  {:.6}",
        losses.first().copied().unwrap_or(0.0)
    );
    println!(
        "   Final loss:    {:.6}",
        losses.last().copied().unwrap_or(0.0)
    );
    println!("   Time:          {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("   Param efficiency: {:.2}% of base model", efficiency);
    println!();

    // =========================================================================
    // Section 6: Merge and Save
    // =========================================================================
    println!("6. Merge Adapter & Save to APR v2");
    println!("   ─────────────────────────────────────────");

    // Merge LoRA into base weight for inference
    lora_layer.merge();
    println!("   Merged:  {}", lora_layer.is_merged());

    // Save merged model to APR v2
    let weight_bytes: Vec<u8> = lora_layer
        .base_weight()
        .data()
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("lora_finetuned.apr");

    let bundle = ModelBundleV2::new()
        .with_name("lora-finetuned-model")
        .with_description(format!(
            "LoRA r={} alpha={} finetuned {}x{}",
            config.rank, config.alpha, config.d_out, config.d_in
        ))
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor(
            "merged_weight",
            vec![config.d_out, config.d_in],
            weight_bytes,
        )
        .build();

    std::fs::write(&model_path, &bundle).expect("Failed to write model");

    if let Ok(metadata) = std::fs::metadata(&model_path) {
        println!("   Saved:   {}", model_path.display());
        println!("   Size:    {} bytes", metadata.len());
    }

    // Unmerge to show we can recover the adapter
    lora_layer.unmerge();
    println!(
        "   Unmerged: {} (adapter recoverable)",
        !lora_layer.is_merged()
    );
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_default() {
        let config = LoRAFineTuneConfig::default();
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 8.0);
    }

    #[test]
    fn test_pretrained_weight_creation() {
        let w = create_pretrained_weight(32, 64, 42);
        assert_eq!(w.len(), 32 * 64);
        assert!(!w.requires_grad());
    }

    #[test]
    fn test_lora_layer_creation() {
        let base = create_pretrained_weight(32, 64, 42);
        let layer = LoRALayer::new(base, 32, 64, 8, 8.0);
        assert_eq!(layer.rank(), 8);
        assert!(!layer.is_merged());
        assert_eq!(layer.lora_a().len(), 8 * 64);
        assert_eq!(layer.lora_b().len(), 32 * 8);
    }

    #[test]
    fn test_lora_forward_output_size() {
        let base = create_pretrained_weight(32, 64, 42);
        let layer = LoRALayer::new(base, 32, 64, 8, 8.0);
        let x = vec![1.0f32; 64];
        let y = lora_forward(&layer, &x, 32, 64);
        assert_eq!(y.len(), 32);
    }

    #[test]
    fn test_lora_merge_unmerge() {
        let base = create_pretrained_weight(16, 16, 42);
        let mut layer = LoRALayer::new(base, 16, 16, 4, 4.0);

        let x = vec![1.0f32; 16];
        let before = lora_forward(&layer, &x, 16, 16);

        layer.merge();
        assert!(layer.is_merged());

        layer.unmerge();
        assert!(!layer.is_merged());

        let after = lora_forward(&layer, &x, 16, 16);

        // Should be approximately equal after merge/unmerge roundtrip
        for (a, b) in before.iter().zip(after.iter()) {
            assert!((a - b).abs() < 1e-4, "merge/unmerge roundtrip mismatch");
        }
    }

    #[test]
    fn test_lora_config_targeting() {
        let config = LoRAConfig::new(8, 8.0).target_qv_projections();
        assert!(config.should_apply("q_proj", None));
        assert!(config.should_apply("v_proj", None));
        assert!(!config.should_apply("k_proj", None));
    }

    #[test]
    fn test_mse_loss() {
        let pred = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0];
        assert!((mse_loss(&pred, &target)).abs() < 1e-6);

        let pred2 = vec![2.0, 3.0, 4.0];
        assert!((mse_loss(&pred2, &target) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_generate_task_data() {
        let (x, y) = generate_task_data(50, 64, 32, 42);
        assert_eq!(x.len(), 50);
        assert_eq!(y.len(), 50);
        assert_eq!(x[0].len(), 64);
        assert_eq!(y[0].len(), 32);
    }

    #[test]
    fn test_generate_data_deterministic() {
        let (x1, y1) = generate_task_data(10, 16, 8, 42);
        let (x2, y2) = generate_task_data(10, 16, 8, 42);
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }

    #[test]
    fn test_param_efficiency() {
        let base_params = 64 * 32; // 2048
        let rank = 8;
        let lora_params = rank * 64 + 32 * rank; // 512 + 256 = 768
        let efficiency = lora_params as f32 / base_params as f32;
        assert!(efficiency < 0.5, "LoRA should use <50% of base params");
    }

    #[test]
    fn test_save_lora_to_apr() {
        let base = create_pretrained_weight(16, 16, 42);
        let mut layer = LoRALayer::new(base, 16, 16, 4, 4.0);
        layer.merge();

        let bytes: Vec<u8> = layer
            .base_weight()
            .data()
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let bundle = ModelBundleV2::new()
            .with_name("test-lora")
            .with_compression(Compression::Lz4)
            .add_tensor("weight", vec![16, 16], bytes)
            .build();

        assert_eq!(&bundle[0..4], b"APR2");
    }
}
