#![allow(unused_imports)]
//! # Recipe: Attention Transfer Distillation
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! **Category**: Model Distillation
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Transfer attention maps from a large teacher model to a smaller student,
//! using MSE loss between projected attention weight matrices.
//!
//! ## Run Command
//! ```bash
//! cargo run --example distill_attention_transfer
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr distill model.apr          # APR native format
//! apr distill model.gguf         # GGUF (llama.cpp compatible)
//! apr distill model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Hinton, G. et al. (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("distill_attention_transfer")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Attention transfer distillation: Teacher -> Student");
    println!();

    // --- Section: Model Specifications ---
    println!("--- Model Specifications ---");
    println!();

    let teacher = AttentionModelSpec {
        name: "teacher".to_string(),
        layers: 12,
        hidden_size: 768,
        num_heads: 12,
        head_dim: 64,
        seq_len: 128,
        params_millions: 110.0,
    };

    let student = AttentionModelSpec {
        name: "student".to_string(),
        layers: 4,
        hidden_size: 256,
        num_heads: 4,
        head_dim: 64,
        seq_len: 128,
        params_millions: 6.5,
    };

    println!("Teacher Model:");
    println!("  Layers: {}", teacher.layers);
    println!("  Hidden: {}", teacher.hidden_size);
    println!("  Attention heads: {}", teacher.num_heads);
    println!("  Head dim: {}", teacher.head_dim);
    println!("  Parameters: {:.1}M", teacher.params_millions);
    println!();

    println!("Student Model:");
    println!("  Layers: {}", student.layers);
    println!("  Hidden: {}", student.hidden_size);
    println!("  Attention heads: {}", student.num_heads);
    println!("  Head dim: {}", student.head_dim);
    println!("  Parameters: {:.1}M", student.params_millions);
    println!();

    let compression_ratio = teacher.params_millions / student.params_millions;
    ctx.record_float_metric("compression_ratio", compression_ratio);

    // --- Section: Attention Map Extraction ---
    println!("--- Attention Map Extraction ---");
    println!();

    let layer_mappings = build_layer_mappings(&teacher, &student)?;

    println!("Layer Mappings (Teacher -> Student):");
    println!("{:-<60}", "");
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>10}",
        "Name", "T.Layer", "T.Heads", "S.Layer", "S.Heads"
    );
    println!("{:-<60}", "");

    for mapping in &layer_mappings {
        println!(
            "{:<10} {:>12} {:>12} {:>12} {:>10}",
            mapping.name,
            mapping.teacher_layer,
            mapping.teacher_heads,
            mapping.student_layer,
            mapping.student_heads,
        );
    }
    println!("{:-<60}", "");
    println!();

    ctx.record_metric("layer_mappings", layer_mappings.len() as i64);

    // --- Section: Attention Projection ---
    println!("--- Attention Projection ---");
    println!();

    let projections: Vec<AttentionProjection> = layer_mappings
        .iter()
        .map(compute_projection)
        .collect::<Result<Vec<_>>>()?;

    println!("Projection Analysis:");
    println!("{:-<65}", "");
    println!(
        "{:<10} {:>14} {:>14} {:>12} {:>10}",
        "Layer", "Teacher Shape", "Student Shape", "Projection", "Params"
    );
    println!("{:-<65}", "");

    for proj in &projections {
        println!(
            "{:<10} {:>5}x{:<8} {:>5}x{:<8} {:>12} {:>10}",
            proj.layer_name,
            proj.teacher_attn_shape.0,
            proj.teacher_attn_shape.1,
            proj.student_attn_shape.0,
            proj.student_attn_shape.1,
            proj.projection_type,
            proj.projection_params,
        );
    }
    println!("{:-<65}", "");
    println!();

    // --- Section: Training Loop ---
    println!("--- Attention Transfer Training ---");
    println!();

    let config = AttentionTransferConfig {
        epochs: 10,
        learning_rate: 1e-4,
        beta: 0.5, // Weight for attention transfer loss vs task loss
    };

    println!("Config:");
    println!("  Epochs: {}", config.epochs);
    println!("  Learning rate: {:.0e}", config.learning_rate);
    println!("  Beta (attention loss weight): {}", config.beta);
    println!();

    println!("Training Progress:");
    println!("{:-<80}", "");
    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Epoch", "Task Loss", "Attn Loss", "Total Loss", "Teacher Acc", "Student Acc"
    );
    println!("{:-<80}", "");

    let mut training_log = Vec::new();
    for epoch in 1..=config.epochs {
        let result = simulate_attention_transfer_epoch(epoch, &config, &projections)?;
        training_log.push(result.clone());

        println!(
            "{:>6} {:>12.4} {:>12.4} {:>12.4} {:>11.2}% {:>11.2}%",
            epoch,
            result.task_loss,
            result.attention_transfer_loss,
            result.total_loss,
            result.teacher_accuracy * 100.0,
            result.student_accuracy * 100.0,
        );
    }
    println!("{:-<80}", "");

    // --- Section: Results ---
    println!();
    println!("--- Results ---");
    println!();

    let final_result = training_log
        .last()
        .ok_or_else(|| CookbookError::invalid_format("No training results"))?;

    ctx.record_float_metric("final_student_accuracy", final_result.student_accuracy);
    ctx.record_float_metric("final_attention_loss", final_result.attention_transfer_loss);

    // Compare with vanilla distillation (no attention transfer)
    let vanilla_accuracy = 0.85;

    println!(
        "  Teacher accuracy:           {:.2}%",
        final_result.teacher_accuracy * 100.0
    );
    println!(
        "  Student accuracy (vanilla):  {:.2}%",
        vanilla_accuracy * 100.0
    );
    println!(
        "  Student accuracy (attn xfr): {:.2}%",
        final_result.student_accuracy * 100.0
    );
    println!(
        "  Improvement over vanilla:   +{:.2}%",
        (final_result.student_accuracy - vanilla_accuracy) * 100.0
    );
    println!(
        "  Knowledge retention:         {:.1}%",
        (final_result.student_accuracy / final_result.teacher_accuracy) * 100.0
    );
    println!("  Compression: {:.1}x fewer parameters", compression_ratio);
    println!(
        "  Final attention MSE: {:.4}",
        final_result.attention_transfer_loss
    );

    // Per-layer attention alignment quality
    println!();
    println!("Per-Layer Attention Alignment:");
    println!("{:-<50}", "");
    println!(
        "{:<10} {:>12} {:>12} {:>12}",
        "Layer", "Attn MSE", "Cosine Sim", "Quality"
    );
    println!("{:-<50}", "");

    for proj in &projections {
        let layer_quality = compute_layer_attention_quality(proj)?;
        println!(
            "{:<10} {:>12.4} {:>12.3} {:>12}",
            layer_quality.layer_name,
            layer_quality.attention_mse,
            layer_quality.cosine_similarity,
            layer_quality.quality_label,
        );
    }
    println!("{:-<50}", "");

    // Save training log
    let log_path = ctx.path("attention_transfer_log.json");
    save_log(&log_path, &training_log)?;
    println!();
    println!("Log saved to: {:?}", log_path);

    ctx.report()?;

    Ok(())
}

mod helpers;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn teacher_spec() -> AttentionModelSpec {
        AttentionModelSpec {
            name: "teacher".to_string(),
            layers: 12,
            hidden_size: 768,
            num_heads: 12,
            head_dim: 64,
            seq_len: 128,
            params_millions: 110.0,
        }
    }

    fn student_spec() -> AttentionModelSpec {
        AttentionModelSpec {
            name: "student".to_string(),
            layers: 4,
            hidden_size: 256,
            num_heads: 4,
            head_dim: 64,
            seq_len: 128,
            params_millions: 6.5,
        }
    }

    fn default_config() -> AttentionTransferConfig {
        AttentionTransferConfig {
            epochs: 10,
            learning_rate: 1e-4,
            beta: 0.5,
        }
    }

    #[test]
    fn test_build_layer_mappings() {
        let teacher = teacher_spec();
        let student = student_spec();
        let mappings = build_layer_mappings(&teacher, &student).unwrap();

        assert_eq!(mappings.len(), 4);
        assert_eq!(mappings[0].teacher_layer, 0);
        assert_eq!(mappings[3].teacher_layer, 11);
    }

    #[test]
    fn test_build_layer_mappings_heads() {
        let teacher = teacher_spec();
        let student = student_spec();
        let mappings = build_layer_mappings(&teacher, &student).unwrap();

        for mapping in &mappings {
            assert_eq!(mapping.teacher_heads, 12);
            assert_eq!(mapping.student_heads, 4);
        }
    }

    #[test]
    fn test_build_layer_mappings_zero_student_layers() {
        let teacher = teacher_spec();
        let mut student = student_spec();
        student.layers = 0;

        let result = build_layer_mappings(&teacher, &student);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_projection_linear() {
        let mapping = AttentionLayerMapping {
            name: "early".to_string(),
            teacher_layer: 0,
            student_layer: 0,
            teacher_heads: 12,
            student_heads: 4,
        };

        let proj = compute_projection(&mapping).unwrap();

        assert_eq!(proj.projection_type, "Linear");
        assert_eq!(proj.projection_params, 48); // 4 * 12
    }

    #[test]
    fn test_compute_projection_identity() {
        let mapping = AttentionLayerMapping {
            name: "same".to_string(),
            teacher_layer: 0,
            student_layer: 0,
            teacher_heads: 8,
            student_heads: 8,
        };

        let proj = compute_projection(&mapping).unwrap();

        assert_eq!(proj.projection_type, "Identity");
        assert_eq!(proj.projection_params, 0);
    }

    #[test]
    fn test_epoch_result_teacher_constant() {
        let config = default_config();
        let projections = vec![AttentionProjection {
            layer_name: "test".to_string(),
            teacher_attn_shape: (12, 128),
            student_attn_shape: (4, 128),
            projection_type: "Linear".to_string(),
            projection_params: 48,
        }];

        let r1 = simulate_attention_transfer_epoch(1, &config, &projections).unwrap();
        let r5 = simulate_attention_transfer_epoch(5, &config, &projections).unwrap();

        assert_eq!(r1.teacher_accuracy, r5.teacher_accuracy);
    }

    #[test]
    fn test_student_improves_over_epochs() {
        let config = default_config();
        let projections = vec![AttentionProjection {
            layer_name: "test".to_string(),
            teacher_attn_shape: (12, 128),
            student_attn_shape: (4, 128),
            projection_type: "Linear".to_string(),
            projection_params: 48,
        }];

        let early = simulate_attention_transfer_epoch(1, &config, &projections).unwrap();
        let late = simulate_attention_transfer_epoch(10, &config, &projections).unwrap();

        assert!(late.student_accuracy > early.student_accuracy);
    }

    #[test]
    fn test_attention_loss_decreases() {
        let config = default_config();
        let projections = vec![AttentionProjection {
            layer_name: "test".to_string(),
            teacher_attn_shape: (12, 128),
            student_attn_shape: (4, 128),
            projection_type: "Linear".to_string(),
            projection_params: 48,
        }];

        let early = simulate_attention_transfer_epoch(1, &config, &projections).unwrap();
        let late = simulate_attention_transfer_epoch(10, &config, &projections).unwrap();

        assert!(late.attention_transfer_loss < early.attention_transfer_loss);
    }

    #[test]
    fn test_total_loss_combines_components() {
        let config = AttentionTransferConfig {
            epochs: 10,
            learning_rate: 1e-4,
            beta: 0.5,
        };
        let projections = vec![AttentionProjection {
            layer_name: "test".to_string(),
            teacher_attn_shape: (12, 128),
            student_attn_shape: (4, 128),
            projection_type: "Linear".to_string(),
            projection_params: 48,
        }];

        let result = simulate_attention_transfer_epoch(5, &config, &projections).unwrap();

        let expected_total =
            (1.0 - config.beta) * result.task_loss + config.beta * result.attention_transfer_loss;
        assert!((result.total_loss - expected_total).abs() < 1e-10);
    }

    #[test]
    fn test_layer_attention_quality_bounded() {
        let proj = AttentionProjection {
            layer_name: "test_layer".to_string(),
            teacher_attn_shape: (12, 128),
            student_attn_shape: (4, 128),
            projection_type: "Linear".to_string(),
            projection_params: 48,
        };

        let quality = compute_layer_attention_quality(&proj).unwrap();

        assert!(quality.attention_mse > 0.0);
        assert!(quality.cosine_similarity >= 0.0);
        assert!(quality.cosine_similarity <= 1.0);
        assert!(!quality.quality_label.is_empty());
    }

    #[test]
    fn test_deterministic_epoch() {
        let config = default_config();
        let projections = vec![AttentionProjection {
            layer_name: "test".to_string(),
            teacher_attn_shape: (12, 128),
            student_attn_shape: (4, 128),
            projection_type: "Linear".to_string(),
            projection_params: 48,
        }];

        let r1 = simulate_attention_transfer_epoch(5, &config, &projections).unwrap();
        let r2 = simulate_attention_transfer_epoch(5, &config, &projections).unwrap();

        assert_eq!(r1.student_accuracy, r2.student_accuracy);
        assert_eq!(r1.attention_transfer_loss, r2.attention_transfer_loss);
        assert_eq!(r1.total_loss, r2.total_loss);
    }

    #[test]
    fn test_save_log() {
        let ctx = RecipeContext::new("test_attn_save").unwrap();
        let path = ctx.path("log.json");

        let log = vec![EpochResult {
            epoch: 1,
            task_loss: 1.5,
            attention_transfer_loss: 0.8,
            total_loss: 1.15,
            teacher_accuracy: 0.92,
            student_accuracy: 0.45,
        }];

        save_log(&path, &log).unwrap();
        assert!(path.exists());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_student_accuracy_bounded(epoch in 1u32..100) {
            let config = AttentionTransferConfig {
                epochs: 100,
                learning_rate: 1e-4,
                beta: 0.5,
            };
            let projections = vec![AttentionProjection {
                layer_name: "test".to_string(),
                teacher_attn_shape: (12, 128),
                student_attn_shape: (4, 128),
                projection_type: "Linear".to_string(),
                projection_params: 48,
            }];

            let result = simulate_attention_transfer_epoch(epoch, &config, &projections).unwrap();

            prop_assert!(result.student_accuracy >= 0.0);
            prop_assert!(result.student_accuracy <= result.teacher_accuracy);
        }

        #[test]
        fn prop_attention_loss_positive(epoch in 1u32..50) {
            let config = AttentionTransferConfig {
                epochs: 50,
                learning_rate: 1e-4,
                beta: 0.5,
            };
            let projections = vec![AttentionProjection {
                layer_name: "test".to_string(),
                teacher_attn_shape: (12, 128),
                student_attn_shape: (4, 128),
                projection_type: "Linear".to_string(),
                projection_params: 48,
            }];

            let result = simulate_attention_transfer_epoch(epoch, &config, &projections).unwrap();

            prop_assert!(result.attention_transfer_loss > 0.0);
            prop_assert!(result.task_loss > 0.0);
            prop_assert!(result.total_loss > 0.0);
        }

        #[test]
        fn prop_total_loss_is_weighted_sum(epoch in 1u32..50, beta in 0.0f64..1.0) {
            let config = AttentionTransferConfig {
                epochs: 50,
                learning_rate: 1e-4,
                beta,
            };
            let projections = vec![AttentionProjection {
                layer_name: "test".to_string(),
                teacher_attn_shape: (12, 128),
                student_attn_shape: (4, 128),
                projection_type: "Linear".to_string(),
                projection_params: 48,
            }];

            let result = simulate_attention_transfer_epoch(epoch, &config, &projections).unwrap();

            let expected = (1.0 - beta) * result.task_loss + beta * result.attention_transfer_loss;
            prop_assert!((result.total_loss - expected).abs() < 1e-10);
        }

        #[test]
        fn prop_layer_quality_cosine_bounded(name in "[a-z]{3,10}") {
            let proj = AttentionProjection {
                layer_name: name,
                teacher_attn_shape: (12, 128),
                student_attn_shape: (4, 128),
                projection_type: "Linear".to_string(),
                projection_params: 48,
            };

            let quality = compute_layer_attention_quality(&proj).unwrap();

            prop_assert!(quality.cosine_similarity >= 0.0);
            prop_assert!(quality.cosine_similarity <= 1.0);
            prop_assert!(quality.attention_mse > 0.0);
        }
    }
}
