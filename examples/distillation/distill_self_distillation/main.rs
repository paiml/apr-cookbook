#![allow(unused_imports)]
//! # Recipe: Self-Distillation
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
//! Self-distillation: deeper layers teach shallower layers within the same model.
//! Demonstrates born-again networks with iterative self-distillation generations.
//!
//! ## Run Command
//! ```bash
//! cargo run --example distill_self_distillation
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
    let mut ctx = RecipeContext::new("distill_self_distillation")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Self-distillation: deeper layers teach shallower layers");
    println!();

    // --- Section 1: Model Architecture ---
    println!("--- Model Architecture ---");
    println!();

    let model = SelfDistillModel {
        name: "resnet-sd".to_string(),
        num_layers: 8,
        hidden_size: 512,
        num_classes: 10,
        params_millions: 25.0,
    };

    println!("Model: {}", model.name);
    println!("  Layers: {}", model.num_layers);
    println!("  Hidden size: {}", model.hidden_size);
    println!("  Classes: {}", model.num_classes);
    println!("  Parameters: {:.1}M", model.params_millions);
    println!();

    ctx.record_metric("num_layers", i64::from(model.num_layers));

    // --- Section 2: Layer Representations ---
    println!("--- Layer Intermediate Representations ---");
    println!();

    let layer_reps = compute_layer_representations(&model)?;

    println!("{:-<65}", "");
    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>12}",
        "Layer", "Dim", "Entropy", "Confidence", "Role"
    );
    println!("{:-<65}", "");

    for rep in &layer_reps {
        println!(
            "{:>6} {:>12} {:>12.4} {:>12.3} {:>12}",
            rep.layer_index, rep.hidden_dim, rep.entropy, rep.confidence, rep.role
        );
    }
    println!("{:-<65}", "");
    println!();

    // --- Section 3: Self-Distillation (deep -> shallow) ---
    println!("--- Self-Distillation: Deep Layers -> Shallow Layers ---");
    println!();

    let config = SelfDistillConfig {
        temperature: 3.0,
        alpha_kl: 0.6,
        alpha_aux: 0.2,
        alpha_task: 0.2,
        epochs: 10,
    };

    println!("Self-Distillation Config:");
    println!("  Temperature: {}", config.temperature);
    println!("  Alpha KL (deep->shallow): {}", config.alpha_kl);
    println!("  Alpha auxiliary classifiers: {}", config.alpha_aux);
    println!("  Alpha task loss: {}", config.alpha_task);
    println!("  Epochs: {}", config.epochs);
    println!();

    let sd_pairs = build_distillation_pairs(&model)?;

    println!("Distillation Pairs (Teacher Layer -> Student Layer):");
    for pair in &sd_pairs {
        println!(
            "  Layer {} ({}) teaches Layer {} ({})",
            pair.teacher_layer, pair.teacher_role, pair.student_layer, pair.student_role
        );
    }
    println!();

    println!("Training Progress:");
    println!("{:-<75}", "");
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "Epoch", "KL Loss", "Aux Loss", "Task Loss", "Total", "Accuracy"
    );
    println!("{:-<75}", "");

    let mut training_log = Vec::new();
    for epoch in 1..=config.epochs {
        let result = simulate_self_distill_epoch(epoch, &config)?;
        training_log.push(result.clone());

        println!(
            "{:>6} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>11.2}%",
            epoch,
            result.kl_loss,
            result.aux_loss,
            result.task_loss,
            result.total_loss,
            result.accuracy * 100.0
        );
    }
    println!("{:-<75}", "");
    println!();

    let final_epoch = training_log
        .last()
        .ok_or_else(|| CookbookError::invalid_format("No training results"))?;

    ctx.record_float_metric("sd_final_accuracy", final_epoch.accuracy);
    ctx.record_float_metric("sd_final_loss", final_epoch.total_loss);

    // --- Section 4: Born-Again Networks ---
    println!("--- Born-Again Networks: Iterative Self-Distillation ---");
    println!();
    println!("Each generation trains from scratch, then self-distills using");
    println!("the previous generation as a soft-target teacher.");
    println!();

    let num_generations = 5;
    let generations = simulate_born_again_generations(num_generations, &config)?;

    println!("{:-<60}", "");
    println!(
        "{:>4} {:>15} {:>15} {:>15}",
        "Gen", "Accuracy", "Improvement", "Cumulative"
    );
    println!("{:-<60}", "");

    let baseline_accuracy = generations
        .first()
        .ok_or_else(|| CookbookError::invalid_format("No generations"))?
        .accuracy;

    for gen in &generations {
        let improvement = if gen.generation == 0 {
            0.0
        } else {
            let prev = &generations[gen.generation as usize - 1];
            gen.accuracy - prev.accuracy
        };
        let cumulative = gen.accuracy - baseline_accuracy;

        println!(
            "{:>4} {:>14.2}% {:>14.2}% {:>14.2}%",
            gen.generation,
            gen.accuracy * 100.0,
            improvement * 100.0,
            cumulative * 100.0
        );
    }
    println!("{:-<60}", "");

    let final_gen = generations
        .last()
        .ok_or_else(|| CookbookError::invalid_format("No generation results"))?;

    let total_improvement = final_gen.accuracy - baseline_accuracy;
    println!();
    println!("Born-Again Results:");
    println!("  Baseline (gen 0): {:.2}%", baseline_accuracy * 100.0);
    println!(
        "  Final (gen {}): {:.2}%",
        final_gen.generation,
        final_gen.accuracy * 100.0
    );
    println!("  Total improvement: +{:.2}%", total_improvement * 100.0);
    println!("  Generations: {}", num_generations);

    ctx.record_float_metric("born_again_baseline", baseline_accuracy);
    ctx.record_float_metric("born_again_final", final_gen.accuracy);
    ctx.record_float_metric("born_again_improvement", total_improvement);
    ctx.record_metric("born_again_generations", i64::from(num_generations));

    // --- Section 5: Layer-wise Accuracy Improvement ---
    println!();
    println!("--- Layer-wise Auxiliary Classifier Accuracy ---");
    println!();

    let layer_accuracies = compute_layer_accuracies(&model, &config)?;

    println!("{:-<55}", "");
    println!(
        "{:>6} {:>15} {:>15} {:>12}",
        "Layer", "Before SD", "After SD", "Gain"
    );
    println!("{:-<55}", "");

    for la in &layer_accuracies {
        println!(
            "{:>6} {:>14.2}% {:>14.2}% {:>11.2}%",
            la.layer_index,
            la.before_sd * 100.0,
            la.after_sd * 100.0,
            (la.after_sd - la.before_sd) * 100.0
        );
    }
    println!("{:-<55}", "");
    println!();

    // Save results
    let log_path = ctx.path("self_distillation_log.json");
    save_log(&log_path, &training_log)?;

    let gen_path = ctx.path("born_again_generations.json");
    save_generations(&gen_path, &generations)?;

    println!("Training log saved to: {:?}", log_path);
    println!("Generation log saved to: {:?}", gen_path);

    println!();
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

    fn test_model() -> SelfDistillModel {
        SelfDistillModel {
            name: "test-model".to_string(),
            num_layers: 8,
            hidden_size: 512,
            num_classes: 10,
            params_millions: 25.0,
        }
    }

    fn test_config() -> SelfDistillConfig {
        SelfDistillConfig {
            temperature: 3.0,
            alpha_kl: 0.6,
            alpha_aux: 0.2,
            alpha_task: 0.2,
            epochs: 10,
        }
    }

    #[test]
    fn test_layer_representations() {
        let model = test_model();
        let reps = compute_layer_representations(&model).unwrap();

        assert_eq!(reps.len(), model.num_layers as usize);
        // First layer should be "input"
        assert_eq!(reps[0].role, "input");
        // Last layer should be "output"
        assert_eq!(reps[reps.len() - 1].role, "output");
    }

    #[test]
    fn test_deeper_layers_higher_confidence() {
        let model = test_model();
        let reps = compute_layer_representations(&model).unwrap();

        // On average, deeper layers should have higher confidence
        let shallow_avg = reps[..3].iter().map(|r| r.confidence).sum::<f64>() / 3.0;
        let deep_avg = reps[5..].iter().map(|r| r.confidence).sum::<f64>() / 3.0;

        assert!(deep_avg > shallow_avg);
    }

    #[test]
    fn test_distillation_pairs() {
        let model = test_model();
        let pairs = build_distillation_pairs(&model).unwrap();

        // Should have num_layers/2 pairs
        assert_eq!(pairs.len(), (model.num_layers / 2) as usize);

        // Teacher layers should be deeper than student layers
        for pair in &pairs {
            assert!(pair.teacher_layer > pair.student_layer);
        }
    }

    #[test]
    fn test_self_distill_epoch_loss_decreases() {
        let config = test_config();

        let early = simulate_self_distill_epoch(1, &config).unwrap();
        let late = simulate_self_distill_epoch(10, &config).unwrap();

        assert!(late.total_loss < early.total_loss);
        assert!(late.kl_loss < early.kl_loss);
    }

    #[test]
    fn test_self_distill_accuracy_improves() {
        let config = test_config();

        let early = simulate_self_distill_epoch(1, &config).unwrap();
        let late = simulate_self_distill_epoch(10, &config).unwrap();

        assert!(late.accuracy > early.accuracy);
    }

    #[test]
    fn test_born_again_improves() {
        let config = test_config();
        let generations = simulate_born_again_generations(5, &config).unwrap();

        assert_eq!(generations.len(), 5);

        // Each generation should be at least as good as the previous
        for window in generations.windows(2) {
            assert!(window[1].accuracy >= window[0].accuracy);
        }
    }

    #[test]
    fn test_born_again_diminishing_returns() {
        let config = test_config();
        let generations = simulate_born_again_generations(5, &config).unwrap();

        // First improvement should be larger than later improvements
        let first_gain = generations[1].accuracy - generations[0].accuracy;
        let last_gain = generations[4].accuracy - generations[3].accuracy;

        assert!(first_gain > last_gain);
    }

    #[test]
    fn test_layer_accuracies_shallow_gain_more() {
        let model = test_model();
        let config = test_config();
        let accuracies = compute_layer_accuracies(&model, &config).unwrap();

        assert_eq!(accuracies.len(), model.num_layers as usize);

        // Shallow layers should gain more from self-distillation
        let shallow_gain = accuracies[0].after_sd - accuracies[0].before_sd;
        let deep_gain =
            accuracies[accuracies.len() - 1].after_sd - accuracies[accuracies.len() - 1].before_sd;

        assert!(shallow_gain > deep_gain);
    }

    #[test]
    fn test_deterministic_epoch() {
        let config = test_config();

        let r1 = simulate_self_distill_epoch(5, &config).unwrap();
        let r2 = simulate_self_distill_epoch(5, &config).unwrap();

        assert_eq!(r1.kl_loss, r2.kl_loss);
        assert_eq!(r1.accuracy, r2.accuracy);
        assert_eq!(r1.total_loss, r2.total_loss);
    }

    #[test]
    fn test_save_log() {
        let ctx = RecipeContext::new("test_sd_save_log").unwrap();
        let path = ctx.path("log.json");

        let log = vec![EpochResult {
            epoch: 1,
            kl_loss: 1.0,
            aux_loss: 0.5,
            task_loss: 0.3,
            total_loss: 0.7,
            accuracy: 0.8,
        }];

        save_log(&path, &log).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_save_generations() {
        let ctx = RecipeContext::new("test_sd_save_gen").unwrap();
        let path = ctx.path("gen.json");

        let generations = vec![GenerationResult {
            generation: 0,
            accuracy: 0.9,
            final_loss: 0.3,
        }];

        save_generations(&path, &generations).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_weighted_loss_components() {
        let config = test_config();
        let result = simulate_self_distill_epoch(5, &config).unwrap();

        // Total loss should be the weighted sum
        let expected_total = config.alpha_kl * result.kl_loss
            + config.alpha_aux * result.aux_loss
            + config.alpha_task * result.task_loss;

        assert!((result.total_loss - expected_total).abs() < 1e-10);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_accuracy_bounded(epoch in 1u32..100) {
            let config = SelfDistillConfig {
                temperature: 3.0,
                alpha_kl: 0.6,
                alpha_aux: 0.2,
                alpha_task: 0.2,
                epochs: 100,
            };

            let result = simulate_self_distill_epoch(epoch, &config).unwrap();

            prop_assert!(result.accuracy >= 0.0);
            prop_assert!(result.accuracy <= 1.0);
        }

        #[test]
        fn prop_loss_positive(epoch in 1u32..50) {
            let config = SelfDistillConfig {
                temperature: 3.0,
                alpha_kl: 0.6,
                alpha_aux: 0.2,
                alpha_task: 0.2,
                epochs: 50,
            };

            let result = simulate_self_distill_epoch(epoch, &config).unwrap();

            prop_assert!(result.kl_loss > 0.0);
            prop_assert!(result.aux_loss > 0.0);
            prop_assert!(result.task_loss > 0.0);
            prop_assert!(result.total_loss > 0.0);
        }

        #[test]
        fn prop_born_again_monotonic(num_gens in 2u32..10) {
            let config = SelfDistillConfig {
                temperature: 3.0,
                alpha_kl: 0.6,
                alpha_aux: 0.2,
                alpha_task: 0.2,
                epochs: 10,
            };

            let generations = simulate_born_again_generations(num_gens, &config).unwrap();

            for window in generations.windows(2) {
                prop_assert!(window[1].accuracy >= window[0].accuracy);
            }
        }

        #[test]
        fn prop_layer_accuracy_after_ge_before(num_layers in 2u32..20) {
            let model = SelfDistillModel {
                name: "prop-test".to_string(),
                num_layers,
                hidden_size: 256,
                num_classes: 10,
                params_millions: 10.0,
            };
            let config = SelfDistillConfig {
                temperature: 3.0,
                alpha_kl: 0.6,
                alpha_aux: 0.2,
                alpha_task: 0.2,
                epochs: 10,
            };

            let accuracies = compute_layer_accuracies(&model, &config).unwrap();

            for la in &accuracies {
                prop_assert!(la.after_sd >= la.before_sd);
                prop_assert!(la.after_sd <= 1.0);
                prop_assert!(la.before_sd >= 0.0);
            }
        }
    }
}
