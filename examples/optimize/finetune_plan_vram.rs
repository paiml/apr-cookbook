//! # Recipe: VRAM Planning for Fine-Tuning
//!
//! **Category**: optimize
//! **CLI Equivalent**: `apr finetune --plan`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Estimates VRAM requirements for different fine-tuning methods (full,
//! LoRA, QLoRA) and recommends optimal configurations given GPU memory
//! constraints.
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Clippy clean
//! 6. [x] No `unwrap()` in logic
//!
//! ## Learning Objective
//! Understand how to estimate GPU memory requirements for fine-tuning and
//! select the right method (full/LoRA/QLoRA) based on available hardware.
//!
//! ## Run Command
//! ```bash
//! cargo run --example finetune_plan_vram
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

use apr_cookbook::prelude::*;

/// Fine-tuning method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FinetuneMethod {
    Full,
    LoRA,
    QLoRA,
}

impl FinetuneMethod {
    const fn name(self) -> &'static str {
        match self {
            Self::Full => "Full",
            Self::LoRA => "LoRA",
            Self::QLoRA => "QLoRA",
        }
    }
}

/// Model configuration for VRAM estimation
#[derive(Debug, Clone)]
struct ModelConfig {
    /// Model name for display
    name: String,
    /// Total parameter count
    params: u64,
    /// Hidden dimension (used for activation estimates)
    hidden_dim: u64,
    /// Number of transformer layers
    num_layers: u64,
    /// Bytes per parameter for base model storage
    dtype_bytes: u64,
}

/// VRAM breakdown for a fine-tuning configuration
#[derive(Debug, Clone)]
struct VramEstimate {
    /// Base model weights in bytes
    base_model: u64,
    /// Optimizer states (momentum + variance for Adam)
    optimizer_states: u64,
    /// Gradient storage
    gradients: u64,
    /// Activation memory (forward pass)
    activations: u64,
    /// LoRA adapter overhead
    lora_overhead: u64,
    /// Total estimated VRAM
    total: u64,
}

impl VramEstimate {
    fn total_gb(&self) -> f64 {
        self.total as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Estimate VRAM requirements for a given model, method, and configuration
fn estimate_vram(
    model: &ModelConfig,
    method: FinetuneMethod,
    lora_rank: u64,
    batch_size: u64,
    seq_len: u64,
) -> VramEstimate {
    // Base model storage
    let base_model = match method {
        FinetuneMethod::Full => model.params * model.dtype_bytes,
        FinetuneMethod::LoRA => model.params * model.dtype_bytes,
        // QLoRA: 4-bit quantized base
        FinetuneMethod::QLoRA => model.params / 2, // 4 bits per param
    };

    // Trainable parameters
    let trainable_params = match method {
        FinetuneMethod::Full => model.params,
        FinetuneMethod::LoRA | FinetuneMethod::QLoRA => {
            // LoRA targets Q, K, V, O projections in each layer
            // Each projection: rank * hidden_dim * 2 (A and B matrices)
            let per_layer = 4 * lora_rank * model.hidden_dim * 2;
            per_layer * model.num_layers
        }
    };

    // Optimizer states: Adam needs 2 states (momentum + variance) per trainable param
    // Each state is FP32 (4 bytes)
    let optimizer_states = trainable_params * 4 * 2;

    // Gradients: one FP32 gradient per trainable param
    let gradients = trainable_params * 4;

    // Activation memory: proportional to batch_size * seq_len * hidden_dim * num_layers
    // Rough estimate: each layer stores activations for backprop
    let activations = batch_size * seq_len * model.hidden_dim * model.num_layers * 2;

    // LoRA adapter weight storage (FP32)
    let lora_overhead = match method {
        FinetuneMethod::Full => 0,
        FinetuneMethod::LoRA | FinetuneMethod::QLoRA => trainable_params * 4,
    };

    let total = base_model + optimizer_states + gradients + activations + lora_overhead;

    VramEstimate {
        base_model,
        optimizer_states,
        gradients,
        activations,
        lora_overhead,
        total,
    }
}

/// Recommended configuration given GPU VRAM
#[derive(Debug, Clone)]
struct OptimalConfig {
    method: FinetuneMethod,
    rank: u64,
    batch_size: u64,
    estimated_vram_gb: f64,
}

/// Find optimal fine-tuning configuration for given GPU VRAM
fn find_optimal_config(model: &ModelConfig, gpu_vram_gb: f64) -> OptimalConfig {
    let gpu_vram_bytes = (gpu_vram_gb * 1024.0 * 1024.0 * 1024.0) as u64;
    let seq_len = 512;

    // Try methods in order of preference: Full > LoRA > QLoRA
    let methods = [
        FinetuneMethod::Full,
        FinetuneMethod::LoRA,
        FinetuneMethod::QLoRA,
    ];
    let ranks = [64, 32, 16, 8, 4];
    let batch_sizes = [32, 16, 8, 4, 2, 1];

    for &method in &methods {
        let rank_list: &[u64] = if method == FinetuneMethod::Full {
            &[0] // rank is irrelevant for full fine-tuning
        } else {
            &ranks
        };

        for &rank in rank_list {
            for &bs in &batch_sizes {
                let est = estimate_vram(model, method, rank, bs, seq_len);
                // Use 90% of GPU VRAM as safe threshold
                if est.total <= (gpu_vram_bytes as f64 * 0.9) as u64 {
                    return OptimalConfig {
                        method,
                        rank,
                        batch_size: bs,
                        estimated_vram_gb: est.total_gb(),
                    };
                }
            }
        }
    }

    // Fallback: minimal QLoRA config
    let est = estimate_vram(model, FinetuneMethod::QLoRA, 4, 1, seq_len);
    OptimalConfig {
        method: FinetuneMethod::QLoRA,
        rank: 4,
        batch_size: 1,
        estimated_vram_gb: est.total_gb(),
    }
}

/// Build the suite of model configurations used for VRAM planning
fn build_model_specs() -> Vec<ModelConfig> {
    vec![
        ModelConfig {
            name: "TinyLM-125M".to_string(),
            params: 125_000_000,
            hidden_dim: 768,
            num_layers: 12,
            dtype_bytes: 4,
        },
        ModelConfig {
            name: "SmallLM-1B".to_string(),
            params: 1_000_000_000,
            hidden_dim: 2048,
            num_layers: 24,
            dtype_bytes: 4,
        },
        ModelConfig {
            name: "MediumLM-7B".to_string(),
            params: 7_000_000_000,
            hidden_dim: 4096,
            num_layers: 32,
            dtype_bytes: 2, // FP16 base
        },
        ModelConfig {
            name: "LargeLM-70B".to_string(),
            params: 70_000_000_000,
            hidden_dim: 8192,
            num_layers: 80,
            dtype_bytes: 2, // FP16 base
        },
    ]
}

/// Print model specification summary table
fn section_model_specs(models: &[ModelConfig]) {
    println!("Model Specifications");
    println!("   ─────────────────────────────────────────");
    for m in models {
        let size_gb = (m.params * m.dtype_bytes) as f64 / (1024.0 * 1024.0 * 1024.0);
        println!(
            "   {:<15}: {:>5.1}B params, hidden={}, layers={}, base={:.1} GB",
            m.name,
            m.params as f64 / 1e9,
            m.hidden_dim,
            m.num_layers,
            size_gb
        );
    }
    println!();
}

/// Print detailed VRAM breakdown for the reference model across all methods
fn section_vram_breakdown(ref_model: &ModelConfig) {
    println!("VRAM Breakdown: {} (batch=4, seq=512)", ref_model.name);
    println!("   ─────────────────────────────────────────");

    for method in [
        FinetuneMethod::Full,
        FinetuneMethod::LoRA,
        FinetuneMethod::QLoRA,
    ] {
        let rank = if method == FinetuneMethod::Full {
            0
        } else {
            16
        };
        let est = estimate_vram(ref_model, method, rank, 4, 512);
        println!("   {} (rank={}):", method.name(), rank);
        println!(
            "      Base model:       {:>8.2} GB",
            est.base_model as f64 / 1e9
        );
        println!(
            "      Optimizer states: {:>8.2} GB",
            est.optimizer_states as f64 / 1e9
        );
        println!(
            "      Gradients:        {:>8.2} GB",
            est.gradients as f64 / 1e9
        );
        println!(
            "      Activations:      {:>8.2} GB",
            est.activations as f64 / 1e9
        );
        println!(
            "      LoRA overhead:    {:>8.2} GB",
            est.lora_overhead as f64 / 1e9
        );
        println!("      TOTAL:            {:>8.2} GB", est.total_gb());
        println!();
    }
}

/// Print method comparison table (Full vs LoRA vs QLoRA) for all models
fn section_method_comparison(models: &[ModelConfig]) {
    println!("Method Comparison (batch=4, seq=512, rank=16)");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>15} {:>10} {:>10} {:>10}",
        "Model", "Full (GB)", "LoRA (GB)", "QLoRA (GB)"
    );
    println!("   {}", "-".repeat(55));

    for m in models {
        let full = estimate_vram(m, FinetuneMethod::Full, 0, 4, 512);
        let lora = estimate_vram(m, FinetuneMethod::LoRA, 16, 4, 512);
        let qlora = estimate_vram(m, FinetuneMethod::QLoRA, 16, 4, 512);
        println!(
            "   {:>15} {:>10.2} {:>10.2} {:>10.2}",
            m.name,
            full.total_gb(),
            lora.total_gb(),
            qlora.total_gb()
        );
    }
    println!();
}

/// Print GPU recommendations for each model across common GPU configurations
fn section_gpu_recommendations(models: &[ModelConfig]) {
    println!("GPU Recommendations");
    println!("   ─────────────────────────────────────────");

    let gpus = [
        ("RTX 3060 12GB", 12.0),
        ("RTX 3090 24GB", 24.0),
        ("RTX 4090 24GB", 24.0),
        ("A100 40GB", 40.0),
        ("A100 80GB", 80.0),
        ("H100 80GB", 80.0),
    ];

    for m in models {
        println!("   {}:", m.name);
        for &(gpu_name, vram) in &gpus {
            let opt = find_optimal_config(m, vram);
            let rank_str = if opt.method == FinetuneMethod::Full {
                "n/a".to_string()
            } else {
                format!("r={}", opt.rank)
            };
            println!(
                "      {:<16}: {} ({}, bs={}, est={:.1} GB)",
                gpu_name,
                opt.method.name(),
                rank_str,
                opt.batch_size,
                opt.estimated_vram_gb
            );
        }
        println!();
    }
}

/// Record VRAM estimate metrics for the reference model into the recipe context
fn section_record_metrics(ctx: &mut RecipeContext, ref_model: &ModelConfig) {
    let ref_full = estimate_vram(ref_model, FinetuneMethod::Full, 0, 4, 512);
    let ref_lora = estimate_vram(ref_model, FinetuneMethod::LoRA, 16, 4, 512);
    let ref_qlora = estimate_vram(ref_model, FinetuneMethod::QLoRA, 16, 4, 512);

    ctx.record_float_metric("ref_full_vram_gb", ref_full.total_gb());
    ctx.record_float_metric("ref_lora_vram_gb", ref_lora.total_gb());
    ctx.record_float_metric("ref_qlora_vram_gb", ref_qlora.total_gb());

    if ref_full.total_gb() > 0.0 {
        ctx.record_float_metric(
            "lora_savings_pct",
            (1.0 - ref_lora.total_gb() / ref_full.total_gb()) * 100.0,
        );
        ctx.record_float_metric(
            "qlora_savings_pct",
            (1.0 - ref_qlora.total_gb() / ref_full.total_gb()) * 100.0,
        );
    }
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("finetune_plan_vram")?;

    println!("=== VRAM Planning for Fine-Tuning ===");
    println!("Mirrors: apr finetune --plan");
    println!();

    let models = build_model_specs();
    section_model_specs(&models);
    section_vram_breakdown(&models[1]); // 1B reference model
    section_method_comparison(&models);
    section_gpu_recommendations(&models);
    section_record_metrics(&mut ctx, &models[1]);

    println!();
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> ModelConfig {
        ModelConfig {
            name: "TestLM".to_string(),
            params: 100_000_000,
            hidden_dim: 512,
            num_layers: 12,
            dtype_bytes: 4,
        }
    }

    #[test]
    fn test_estimate_non_negative() {
        let model = test_model();
        for method in [
            FinetuneMethod::Full,
            FinetuneMethod::LoRA,
            FinetuneMethod::QLoRA,
        ] {
            let est = estimate_vram(&model, method, 8, 4, 512);
            assert!(est.base_model > 0, "{method:?} base_model");
            assert!(est.optimizer_states > 0, "{method:?} optimizer");
            assert!(est.gradients > 0, "{method:?} gradients");
            assert!(est.activations > 0, "{method:?} activations");
            assert!(est.total > 0, "{method:?} total");
        }
    }

    #[test]
    fn test_vram_ordering_full_gt_lora_gt_qlora() {
        let model = test_model();
        let full = estimate_vram(&model, FinetuneMethod::Full, 0, 4, 512);
        let lora = estimate_vram(&model, FinetuneMethod::LoRA, 16, 4, 512);
        let qlora = estimate_vram(&model, FinetuneMethod::QLoRA, 16, 4, 512);
        assert!(full.total > lora.total, "Full > LoRA");
        assert!(lora.total > qlora.total, "LoRA > QLoRA");
    }

    #[test]
    fn test_optimal_config_valid() {
        let model = test_model();
        let opt = find_optimal_config(&model, 24.0);
        assert!(opt.batch_size >= 1);
        assert!(opt.estimated_vram_gb > 0.0);
        assert!(opt.estimated_vram_gb <= 24.0, "must fit in GPU");
    }

    #[test]
    fn test_scaling_with_params() {
        let small = test_model(); // 100M params
        let large = ModelConfig {
            name: "Large".to_string(),
            params: 1_000_000_000,
            hidden_dim: 2048,
            num_layers: 24,
            dtype_bytes: 4,
        };
        let small_est = estimate_vram(&small, FinetuneMethod::Full, 0, 4, 512);
        let large_est = estimate_vram(&large, FinetuneMethod::Full, 0, 4, 512);
        assert!(large_est.total > small_est.total);
    }

    #[test]
    fn test_base_model_sizes_and_overhead() {
        let model = test_model();
        // Full: base = params * dtype_bytes, no LoRA overhead
        let full = estimate_vram(&model, FinetuneMethod::Full, 0, 4, 512);
        assert_eq!(full.base_model, model.params * model.dtype_bytes);
        assert_eq!(full.lora_overhead, 0);
        // QLoRA: base = params/2 (4-bit), has LoRA overhead
        let qlora = estimate_vram(&model, FinetuneMethod::QLoRA, 8, 4, 512);
        assert_eq!(qlora.base_model, model.params / 2);
        assert!(qlora.lora_overhead > 0);
    }

    #[test]
    fn test_batch_size_affects_activations() {
        let model = test_model();
        let bs1 = estimate_vram(&model, FinetuneMethod::LoRA, 8, 1, 512);
        let bs4 = estimate_vram(&model, FinetuneMethod::LoRA, 8, 4, 512);
        assert_eq!(bs4.activations, bs1.activations * 4);
    }

    #[test]
    fn test_total_equals_sum() {
        let model = test_model();
        let est = estimate_vram(&model, FinetuneMethod::LoRA, 8, 4, 512);
        let expected = est.base_model
            + est.optimizer_states
            + est.gradients
            + est.activations
            + est.lora_overhead;
        assert_eq!(est.total, expected);
    }

    #[test]
    fn test_large_model_needs_qlora() {
        let large = ModelConfig {
            name: "70B".to_string(),
            params: 70_000_000_000,
            hidden_dim: 8192,
            num_layers: 80,
            dtype_bytes: 2,
        };
        let opt = find_optimal_config(&large, 24.0);
        assert_eq!(opt.method, FinetuneMethod::QLoRA);
    }
}
