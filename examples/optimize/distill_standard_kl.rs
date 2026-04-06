//! # Recipe: Standard KL Divergence Distillation
//!
//! **Category**: Model Optimization
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: entrenar (distillation)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] Clippy clean
//! 7. [x] Rustfmt standard
//! 8. [x] No `unwrap()` in logic
//!
//! ## Learning Objective
//! Standard KL divergence distillation with temperature scaling.
//! CLI equivalent: `apr distill --strategy standard`
//!
//! ## Run Command
//! ```bash
//! cargo run --example distill_standard_kl
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
use entrenar::distill::DistillationLoss;
use ndarray::Array2;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Generate synthetic logits with controllable bias toward correct class.
fn generate_logits(
    batch_size: usize,
    num_classes: usize,
    bias: f32,
    seed: u64,
) -> (Array2<f32>, Vec<usize>) {
    let mut data = Vec::with_capacity(batch_size * num_classes);
    let mut labels = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let label = b % num_classes;
        labels.push(label);
        for c in 0..num_classes {
            let mut hasher = DefaultHasher::new();
            (seed, b, c).hash(&mut hasher);
            let base = hasher.finish() as f32 / u64::MAX as f32 - 0.5;
            data.push(if c == label { base + bias } else { base });
        }
    }
    (
        Array2::from_shape_vec((batch_size, num_classes), data).expect("valid shape"),
        labels,
    )
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("distill_standard_kl")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Standard KL Divergence Distillation with Temperature Scaling");
    println!();

    // ── Section 1: Basic Distillation ────────────────────────────────────
    println!("--- Basic distillation with default parameters ---");

    let (teacher_logits, labels) = generate_logits(32, 10, 3.0, 42);
    let (student_logits, _) = generate_logits(32, 10, 1.0, 99);

    let temperature = 4.0;
    let alpha = 0.7;
    let loss_fn = DistillationLoss::new(temperature, alpha);
    let loss = loss_fn.forward(&student_logits, &teacher_logits, &labels);

    println!("Teacher bias=3.0, Student bias=1.0, T={temperature}, alpha={alpha}");
    println!("Distillation loss: {loss:.6}");
    println!();

    ctx.record_float_metric("basic_loss", f64::from(loss));

    // ── Section 2: Temperature Sweep ─────────────────────────────────────
    println!("--- Temperature sweep: controlling softness of targets ---");

    let temperatures = [1.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
    let fixed_alpha = 0.7;

    println!("{:>6} | {:>12}", "T", "Loss");
    println!("{:-<6}-+-{:-<12}", "", "");

    let mut temp_losses = Vec::new();
    for &t in &temperatures {
        let lf = DistillationLoss::new(t, fixed_alpha);
        let l = lf.forward(&student_logits, &teacher_logits, &labels);
        println!("{t:6.1} | {l:12.6}");
        temp_losses.push(l);
    }

    println!();
    println!("Higher temperature softens distributions, exposing inter-class relationships");
    println!();

    // ── Section 3: Alpha Balance Sweep ───────────────────────────────────
    println!("--- Alpha sweep: balancing soft targets vs hard labels ---");

    let alphas = [0.0_f32, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
    let fixed_temp = 4.0;

    println!("{:>6} | {:>12}", "Alpha", "Loss");
    println!("{:-<6}-+-{:-<12}", "", "");

    let mut alpha_losses = Vec::new();
    for &a in &alphas {
        let lf = DistillationLoss::new(fixed_temp, a);
        let l = lf.forward(&student_logits, &teacher_logits, &labels);
        println!("{a:6.2} | {l:12.6}");
        alpha_losses.push(l);
    }

    println!();
    println!("alpha=0.0 uses only hard labels; alpha=1.0 uses only soft targets");
    println!();

    // ── Section 4: Training Simulation ───────────────────────────────────
    println!("--- Training simulation: student loss decreases over iterations ---");

    let num_iterations = 10;
    let learning_rate = 0.3_f32;
    let sim_loss_fn = DistillationLoss::new(4.0, 0.7);

    let mut iteration_losses = Vec::new();
    for i in 0..num_iterations {
        let blend = learning_rate * i as f32;
        let blend = blend.min(1.0);

        // Interpolate student logits toward teacher logits
        let blended = &student_logits * (1.0 - blend) + &teacher_logits * blend;
        let l = sim_loss_fn.forward(&blended, &teacher_logits, &labels);
        println!("Iteration {i:2}: loss={l:.6} (blend={blend:.2})");
        iteration_losses.push(l);
    }

    let first_loss = iteration_losses[0];
    let last_loss = iteration_losses[iteration_losses.len() - 1];
    println!();
    println!("Loss decreased from {first_loss:.6} to {last_loss:.6}");

    ctx.record_float_metric("first_loss", f64::from(first_loss));
    ctx.record_float_metric("last_loss", f64::from(last_loss));

    // ── Summary ──────────────────────────────────────────────────────────
    println!();
    println!("--- Summary ---");
    println!("Standard KL distillation transfers dark knowledge from teacher to student.");
    println!("Temperature controls distribution softness; alpha balances soft vs hard targets.");
    println!(
        "Training simulation: {:.1}% loss reduction over {num_iterations} iterations",
        (1.0 - last_loss / first_loss) * 100.0
    );
    println!();

    ctx.report()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_non_negative() {
        let (teacher, labels) = generate_logits(16, 5, 3.0, 42);
        let (student, _) = generate_logits(16, 5, 1.0, 99);
        let loss = DistillationLoss::new(4.0, 0.7).forward(&student, &teacher, &labels);
        assert!(loss >= 0.0, "distillation loss must be non-negative");
    }

    #[test]
    fn test_identical_logits_minimal_loss() {
        let (teacher, labels) = generate_logits(16, 5, 3.0, 42);
        let loss = DistillationLoss::new(4.0, 0.7).forward(&teacher, &teacher, &labels);
        assert!(
            loss < 1.0,
            "identical logits should yield minimal loss, got {loss}"
        );
    }

    #[test]
    fn test_trained_beats_random() {
        let (teacher, labels) = generate_logits(32, 10, 3.0, 42);
        let (random_student, _) = generate_logits(32, 10, 0.0, 99);
        let (trained_student, _) = generate_logits(32, 10, 2.5, 42);

        let loss_fn = DistillationLoss::new(4.0, 0.7);
        let random_loss = loss_fn.forward(&random_student, &teacher, &labels);
        let trained_loss = loss_fn.forward(&trained_student, &teacher, &labels);

        assert!(
            trained_loss < random_loss,
            "trained student ({trained_loss:.4}) should have lower loss than random ({random_loss:.4})"
        );
    }

    #[test]
    fn test_temperature_increases_affect_loss() {
        let (teacher, labels) = generate_logits(16, 5, 3.0, 42);
        let (student, _) = generate_logits(16, 5, 1.0, 99);

        let loss_t1 = DistillationLoss::new(1.0, 0.7).forward(&student, &teacher, &labels);
        let loss_t8 = DistillationLoss::new(8.0, 0.7).forward(&student, &teacher, &labels);

        assert!(
            (loss_t1 - loss_t8).abs() > 1e-6,
            "different temperatures should yield different losses"
        );
    }

    #[test]
    fn test_alpha_zero_vs_one_differ() {
        let (teacher, labels) = generate_logits(16, 5, 3.0, 42);
        let (student, _) = generate_logits(16, 5, 1.0, 99);

        let loss_a0 = DistillationLoss::new(4.0, 0.0).forward(&student, &teacher, &labels);
        let loss_a1 = DistillationLoss::new(4.0, 1.0).forward(&student, &teacher, &labels);

        assert!(
            (loss_a0 - loss_a1).abs() > 1e-6,
            "alpha=0 and alpha=1 must produce different losses"
        );
    }

    #[test]
    fn test_alpha_one_ignores_hard_labels() {
        let (teacher, labels) = generate_logits(16, 5, 3.0, 42);
        let (student, _) = generate_logits(16, 5, 1.0, 99);

        let mut alt_labels = labels.clone();
        for l in &mut alt_labels {
            *l = (*l + 1) % 5;
        }

        let loss_orig = DistillationLoss::new(4.0, 1.0).forward(&student, &teacher, &labels);
        let loss_alt = DistillationLoss::new(4.0, 1.0).forward(&student, &teacher, &alt_labels);

        assert!(
            (loss_orig - loss_alt).abs() < 1e-4,
            "alpha=1.0 should ignore hard labels: orig={loss_orig:.6}, alt={loss_alt:.6}"
        );
    }

    #[test]
    fn test_deterministic_output() {
        let (teacher, labels) = generate_logits(16, 5, 3.0, 42);
        let (student, _) = generate_logits(16, 5, 1.0, 99);
        let loss_fn = DistillationLoss::new(4.0, 0.7);

        let loss1 = loss_fn.forward(&student, &teacher, &labels);
        let loss2 = loss_fn.forward(&student, &teacher, &labels);

        assert!(
            (loss1 - loss2).abs() < f32::EPSILON,
            "same inputs must produce identical loss"
        );
    }

    #[test]
    fn test_loss_is_finite() {
        let (teacher, labels) = generate_logits(16, 5, 3.0, 42);
        let (student, _) = generate_logits(16, 5, 1.0, 99);
        let loss = DistillationLoss::new(4.0, 0.7).forward(&student, &teacher, &labels);
        assert!(loss.is_finite(), "loss must be finite");
    }

    #[test]
    fn test_training_simulation_loss_decreases() {
        let (teacher, labels) = generate_logits(32, 10, 3.0, 42);
        let (student, _) = generate_logits(32, 10, 1.0, 99);
        let loss_fn = DistillationLoss::new(4.0, 0.7);

        let initial = loss_fn.forward(&student, &teacher, &labels);
        let blended = &student * 0.5 + &teacher * 0.5;
        let after = loss_fn.forward(&blended, &teacher, &labels);

        assert!(
            after < initial,
            "blending toward teacher should reduce loss: {initial:.4} -> {after:.4}"
        );
    }

    #[test]
    fn test_generate_logits_shape() {
        let (logits, labels) = generate_logits(8, 5, 2.0, 42);
        assert_eq!(logits.shape(), &[8, 5]);
        assert_eq!(labels.len(), 8);
        assert!(labels.iter().all(|&l| l < 5));
    }

    #[test]
    fn test_large_batch() {
        let (teacher, labels) = generate_logits(256, 20, 3.0, 42);
        let (student, _) = generate_logits(256, 20, 1.0, 99);
        let loss = DistillationLoss::new(4.0, 0.7).forward(&student, &teacher, &labels);
        assert!(loss.is_finite() && loss >= 0.0);
    }
}
