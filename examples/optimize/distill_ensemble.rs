//! # Recipe: Multi-Teacher Ensemble Distillation
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
//! Multi-teacher ensemble distillation with uniform and weighted ensembles.
//! CLI equivalent: `apr distill --strategy ensemble`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Run Command
//! ```bash
//! cargo run --example distill_ensemble
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
use entrenar::distill::EnsembleDistiller;
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
    let mut ctx = RecipeContext::new("distill_ensemble")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Multi-Teacher Ensemble Distillation");
    println!();

    let batch_size = 32;
    let num_classes = 10;
    let temperature = 4.0;
    let alpha = 0.7;

    // ── Section 1: Teacher Creation ──────────────────────────────────────
    println!("--- Creating specialist teacher models ---");

    let teacher_configs: Vec<(f32, u64, &str)> = vec![
        (3.5, 42, "math-specialist"),
        (3.0, 73, "code-specialist"),
        (2.5, 101, "language-specialist"),
    ];

    let mut teacher_logits_list = Vec::new();
    let mut labels = Vec::new();

    for (bias, seed, name) in &teacher_configs {
        let (logits, lbls) = generate_logits(batch_size, num_classes, *bias, *seed);
        println!("Teacher '{name}': bias={bias}, shape={:?}", logits.shape());
        teacher_logits_list.push(logits);
        labels = lbls;
    }

    let (student_logits, _) = generate_logits(batch_size, num_classes, 1.0, 200);
    println!("Student: bias=1.0, shape={:?}", student_logits.shape());
    println!();

    // ── Section 2: Uniform Ensemble ──────────────────────────────────────
    println!("--- Uniform ensemble: equal teacher weights ---");

    let n_teachers = teacher_logits_list.len();
    let uniform_distiller = EnsembleDistiller::uniform(n_teachers, temperature);

    let combined = uniform_distiller.combine_teachers(&teacher_logits_list);
    println!("Combined logits shape: {:?}", combined.shape());

    let uniform_loss =
        uniform_distiller.distillation_loss(&student_logits, &teacher_logits_list, &labels, alpha);
    println!("Uniform ensemble loss: {uniform_loss:.6}");
    println!();

    ctx.record_float_metric("uniform_loss", f64::from(uniform_loss));

    // ── Section 3: Weighted Ensemble ─────────────────────────────────────
    println!("--- Weighted ensemble: emphasizing strongest teacher ---");

    let weights = vec![0.6, 0.25, 0.15];
    println!("Weights: {weights:?}");
    for (w, (_, _, name)) in weights.iter().zip(teacher_configs.iter()) {
        println!("  {name}: {w:.2}");
    }

    let weighted_distiller = EnsembleDistiller::new(weights, temperature);
    let weighted_loss =
        weighted_distiller.distillation_loss(&student_logits, &teacher_logits_list, &labels, alpha);
    println!("Weighted ensemble loss: {weighted_loss:.6}");
    println!(
        "Difference (weighted - uniform): {:.6}",
        weighted_loss - uniform_loss
    );
    println!();

    ctx.record_float_metric("weighted_loss", f64::from(weighted_loss));

    // ── Section 4: Ensemble vs Single Teacher ────────────────────────────
    println!("--- Ensemble vs single-teacher comparison ---");

    let single_distiller = EnsembleDistiller::uniform(1, temperature);
    let single_loss = single_distiller.distillation_loss(
        &student_logits,
        &[teacher_logits_list[0].clone()],
        &labels,
        alpha,
    );

    println!("Single best teacher loss: {single_loss:.6}");
    println!("Uniform ensemble loss:    {uniform_loss:.6}");
    println!("Weighted ensemble loss:   {weighted_loss:.6}");
    println!();

    // Varying number of teachers
    println!("--- Effect of teacher count on ensemble quality ---");

    for n in 1..=teacher_logits_list.len() {
        let subset: Vec<_> = teacher_logits_list[..n].to_vec();
        let distiller = EnsembleDistiller::uniform(n, temperature);
        let loss = distiller.distillation_loss(&student_logits, &subset, &labels, alpha);
        println!("{n} teacher(s): loss={loss:.6}");
    }
    println!();

    // ── Summary ──────────────────────────────────────────────────────────
    println!("--- Summary ---");
    println!("Ensemble distillation combines knowledge from multiple specialist teachers.");
    println!("Uniform weighting treats all teachers equally for balanced transfer.");
    println!("Weighted ensembles emphasize stronger or domain-specific teachers.");
    println!("Multiple teachers provide diverse knowledge signals for richer distillation.");
    println!();

    ctx.report()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combine_teachers_shape() {
        let teachers: Vec<_> = (0..3)
            .map(|i| generate_logits(16, 5, 3.0, i as u64 + 42).0)
            .collect();
        let distiller = EnsembleDistiller::uniform(3, 4.0);
        let combined = distiller.combine_teachers(&teachers);
        assert_eq!(combined.shape(), &[16, 5]);
    }

    #[test]
    fn test_uniform_weights_equal() {
        let teachers: Vec<_> = (0..3)
            .map(|i| generate_logits(16, 5, 3.0, i as u64 + 42).0)
            .collect();

        let uniform = EnsembleDistiller::uniform(3, 4.0);
        let manual = EnsembleDistiller::new(vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 4.0);

        let combined_u = uniform.combine_teachers(&teachers);
        let combined_m = manual.combine_teachers(&teachers);

        let diff = (&combined_u - &combined_m).mapv(f32::abs).sum();
        assert!(
            diff < 1e-4,
            "uniform and manual 1/3 weights should match, diff={diff}"
        );
    }

    #[test]
    fn test_weighted_differs_from_uniform() {
        let teachers: Vec<_> = (0..3)
            .map(|i| generate_logits(16, 5, 3.0, i as u64 + 42).0)
            .collect();
        let (_, labels) = generate_logits(16, 5, 1.0, 99);
        let (student, _) = generate_logits(16, 5, 1.0, 200);

        let uniform_loss =
            EnsembleDistiller::uniform(3, 4.0).distillation_loss(&student, &teachers, &labels, 0.7);
        let weighted_loss = EnsembleDistiller::new(vec![0.7, 0.2, 0.1], 4.0)
            .distillation_loss(&student, &teachers, &labels, 0.7);

        assert!(
            (uniform_loss - weighted_loss).abs() > 1e-6,
            "weighted should differ from uniform"
        );
    }

    #[test]
    fn test_single_teacher_ensemble() {
        let (teacher, labels) = generate_logits(16, 5, 3.0, 42);
        let (student, _) = generate_logits(16, 5, 1.0, 99);

        let distiller = EnsembleDistiller::uniform(1, 4.0);
        let loss = distiller.distillation_loss(&student, &[teacher], &labels, 0.7);
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn test_distillation_loss_non_negative() {
        let teachers: Vec<_> = (0..3)
            .map(|i| generate_logits(16, 5, 3.0, i as u64 + 42).0)
            .collect();
        let (_, labels) = generate_logits(16, 5, 1.0, 99);
        let (student, _) = generate_logits(16, 5, 1.0, 200);

        let loss =
            EnsembleDistiller::uniform(3, 4.0).distillation_loss(&student, &teachers, &labels, 0.7);
        assert!(loss >= 0.0, "ensemble loss must be non-negative");
    }

    #[test]
    fn test_distillation_loss_finite() {
        let teachers: Vec<_> = (0..3)
            .map(|i| generate_logits(16, 5, 3.0, i as u64 + 42).0)
            .collect();
        let (_, labels) = generate_logits(16, 5, 1.0, 99);
        let (student, _) = generate_logits(16, 5, 1.0, 200);

        let loss =
            EnsembleDistiller::uniform(3, 4.0).distillation_loss(&student, &teachers, &labels, 0.7);
        assert!(loss.is_finite(), "loss must be finite");
    }

    #[test]
    fn test_identical_student_teacher() {
        let (teacher, labels) = generate_logits(16, 5, 3.0, 42);
        let distiller = EnsembleDistiller::uniform(1, 4.0);
        let loss = distiller.distillation_loss(&teacher, &[teacher.clone()], &labels, 0.7);
        assert!(
            loss < 1.0,
            "identical student/teacher should yield low loss, got {loss}"
        );
    }

    #[test]
    fn test_combine_single_teacher_identity() {
        let (teacher, _) = generate_logits(16, 5, 3.0, 42);
        let distiller = EnsembleDistiller::uniform(1, 4.0);
        let combined = distiller.combine_teachers(&[teacher.clone()]);
        let diff = (&combined - &teacher).mapv(f32::abs).sum();
        assert!(diff < 1e-5, "single teacher combine should be identity");
    }

    #[test]
    fn test_deterministic() {
        let teachers: Vec<_> = (0..3)
            .map(|i| generate_logits(16, 5, 3.0, i as u64 + 42).0)
            .collect();
        let (_, labels) = generate_logits(16, 5, 1.0, 99);
        let (student, _) = generate_logits(16, 5, 1.0, 200);

        let distiller = EnsembleDistiller::uniform(3, 4.0);
        let loss1 = distiller.distillation_loss(&student, &teachers, &labels, 0.7);
        let loss2 = distiller.distillation_loss(&student, &teachers, &labels, 0.7);
        assert!((loss1 - loss2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_two_teachers_vs_three() {
        let teachers: Vec<_> = (0..3)
            .map(|i| generate_logits(16, 5, 3.0, i as u64 + 42).0)
            .collect();
        let (_, labels) = generate_logits(16, 5, 1.0, 99);
        let (student, _) = generate_logits(16, 5, 1.0, 200);

        let loss_2 = EnsembleDistiller::uniform(2, 4.0).distillation_loss(
            &student,
            &teachers[..2],
            &labels,
            0.7,
        );
        let loss_3 =
            EnsembleDistiller::uniform(3, 4.0).distillation_loss(&student, &teachers, &labels, 0.7);

        assert!(loss_2.is_finite() && loss_2 >= 0.0);
        assert!(loss_3.is_finite() && loss_3 >= 0.0);
        assert!(
            (loss_2 - loss_3).abs() > 1e-6,
            "2 vs 3 teachers should produce different losses"
        );
    }

    #[test]
    fn test_large_ensemble() {
        let teachers: Vec<_> = (0..8)
            .map(|i| generate_logits(32, 10, 3.0, i as u64 + 42).0)
            .collect();
        let (_, labels) = generate_logits(32, 10, 1.0, 99);
        let (student, _) = generate_logits(32, 10, 1.0, 200);

        let loss =
            EnsembleDistiller::uniform(8, 4.0).distillation_loss(&student, &teachers, &labels, 0.7);
        assert!(loss.is_finite() && loss >= 0.0);
    }
}
