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
mod tests;

#[cfg(test)]
mod proptests;
