//! Entrenar Knowledge Distillation Example
//!
//! Demonstrates knowledge distillation using entrenar's distill module:
//! soft-target distillation, multi-teacher ensembles, and progressive
//! layer-wise distillation with hidden state matching.
//!
//! # Distillation Methods
//!
//! - **DistillationLoss**: KL divergence on temperature-scaled softmax + hard CE
//! - **EnsembleDistiller**: Weighted combination of multiple teacher predictions
//! - **ProgressiveDistiller**: Layer-wise MSE/cosine matching of hidden states
//!
//! # Running
//!
//! ```bash
//! cargo run --example entrenar_distillation
//! ```

use entrenar::distill::{DistillationLoss, EnsembleDistiller, ProgressiveDistiller};
use ndarray::Array2;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Generate synthetic logits with controllable confidence
fn generate_logits(
    batch_size: usize,
    num_classes: usize,
    correct_class_bias: f32,
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
            let logit = if c == label {
                base + correct_class_bias
            } else {
                base
            };
            data.push(logit);
        }
    }

    (
        Array2::from_shape_vec((batch_size, num_classes), data).expect("valid shape"),
        labels,
    )
}

/// Generate synthetic hidden states for layer matching
fn generate_hidden_states(
    batch_size: usize,
    hidden_dim: usize,
    num_layers: usize,
    seed: u64,
) -> Vec<Array2<f32>> {
    (0..num_layers)
        .map(|layer| {
            let data: Vec<f32> = (0..batch_size * hidden_dim)
                .map(|i| {
                    let mut hasher = DefaultHasher::new();
                    (seed, "hidden", layer, i).hash(&mut hasher);
                    hasher.finish() as f32 / u64::MAX as f32 - 0.5
                })
                .collect();
            Array2::from_shape_vec((batch_size, hidden_dim), data).expect("valid shape")
        })
        .collect()
}

fn main() {
    println!("=== Entrenar Knowledge Distillation Example ===\n");

    let batch_size = 16;
    let num_classes = 5;
    let class_names = ["cat", "dog", "bird", "fish", "frog"];

    // =========================================================================
    // Section 1: Basic Distillation Loss
    // =========================================================================
    println!("1. Temperature-Scaled Knowledge Distillation");
    println!("   ─────────────────────────────────────────");

    let (teacher_logits, labels) = generate_logits(batch_size, num_classes, 8.0, 42);
    let (student_random, _) = generate_logits(batch_size, num_classes, 0.5, 99);
    let (student_trained, _) = generate_logits(batch_size, num_classes, 5.0, 42);

    let loss_fn = DistillationLoss::new(3.0, 0.7);

    let loss_random = loss_fn.forward(&student_random, &teacher_logits, &labels);
    let loss_trained = loss_fn.forward(&student_trained, &teacher_logits, &labels);

    println!("   Teacher bias: 8.0 (confident)");
    println!("   Temperature:  3.0");
    println!("   Alpha:        0.7 (70% soft targets, 30% hard labels)");
    println!("   Classes:      {:?}", class_names);
    println!();
    println!("   Random student loss:  {:.4}", loss_random);
    println!("   Trained student loss: {:.4}", loss_trained);
    println!(
        "   Improvement:          {:.1}%",
        (1.0 - loss_trained / loss_random) * 100.0
    );
    println!();

    // =========================================================================
    // Section 2: Temperature Sweep
    // =========================================================================
    println!("2. Temperature Sweep (alpha=0.7)");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>6} {:>12} {:>12} {:>10}",
        "Temp", "RandomLoss", "TrainedLoss", "Gap"
    );
    println!("   {}", "─".repeat(44));

    for temp in [1.0, 2.0, 3.0, 5.0, 8.0, 12.0] {
        let loss_fn = DistillationLoss::new(temp, 0.7);
        let l_rand = loss_fn.forward(&student_random, &teacher_logits, &labels);
        let l_train = loss_fn.forward(&student_trained, &teacher_logits, &labels);
        println!(
            "   {:>6.1} {:>12.4} {:>12.4} {:>10.4}",
            temp,
            l_rand,
            l_train,
            l_rand - l_train
        );
    }
    println!();

    // =========================================================================
    // Section 3: Alpha Balance
    // =========================================================================
    println!("3. Alpha Balance (Temperature=3.0)");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>6} {:>12} {:>12} {:>22}",
        "Alpha", "RandomLoss", "TrainedLoss", "Mix"
    );
    println!("   {}", "─".repeat(56));

    for alpha in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0] {
        let loss_fn = DistillationLoss::new(3.0, alpha);
        let l_rand = loss_fn.forward(&student_random, &teacher_logits, &labels);
        let l_train = loss_fn.forward(&student_trained, &teacher_logits, &labels);
        let mix = format!(
            "{:.0}% soft + {:.0}% hard",
            alpha * 100.0,
            (1.0 - alpha) * 100.0
        );
        println!(
            "   {:>6.1} {:>12.4} {:>12.4} {:>22}",
            alpha, l_rand, l_train, mix
        );
    }
    println!();

    // =========================================================================
    // Section 4: Multi-Teacher Ensemble Distillation
    // =========================================================================
    println!("4. Multi-Teacher Ensemble Distillation");
    println!("   ─────────────────────────────────────────");

    // Create specialized teachers
    let (teacher_a, _) = generate_logits(batch_size, num_classes, 10.0, 100); // Math expert
    let (teacher_b, _) = generate_logits(batch_size, num_classes, 7.0, 200); // Code expert
    let (teacher_c, _) = generate_logits(batch_size, num_classes, 5.0, 300); // General

    let teachers = vec![teacher_a, teacher_b, teacher_c];

    // Uniform ensemble
    let uniform_distiller = EnsembleDistiller::uniform(3, 3.0);
    let uniform_combined = uniform_distiller.combine_teachers(&teachers);
    let uniform_loss =
        uniform_distiller.distillation_loss(&student_random, &teachers, &labels, 0.7);

    // Weighted ensemble (favoring math expert)
    let weighted_distiller = EnsembleDistiller::new(vec![3.0, 1.0, 1.0], 3.0);
    let weighted_combined = weighted_distiller.combine_teachers(&teachers);
    let weighted_loss =
        weighted_distiller.distillation_loss(&student_random, &teachers, &labels, 0.7);

    println!("   Teachers: 3 specialists (math/code/general)");
    println!(
        "   Uniform ensemble shape:  [{}, {}]",
        uniform_combined.nrows(),
        uniform_combined.ncols()
    );
    println!("   Uniform ensemble loss:   {:.4}", uniform_loss);
    println!(
        "   Weighted ensemble shape: [{}, {}]",
        weighted_combined.nrows(),
        weighted_combined.ncols()
    );
    println!(
        "   Weighted ensemble loss:  {:.4} (3:1:1 weighting)",
        weighted_loss
    );
    println!();

    // =========================================================================
    // Section 5: Progressive Layer-Wise Distillation
    // =========================================================================
    println!("5. Progressive Layer-Wise Distillation");
    println!("   ─────────────────────────────────────────");

    let num_layers = 4;
    let hidden_dim = 32;

    let teacher_hiddens = generate_hidden_states(batch_size, hidden_dim, num_layers, 42);
    let student_hiddens_close = generate_hidden_states(batch_size, hidden_dim, num_layers, 43);
    let student_hiddens_far = generate_hidden_states(batch_size, hidden_dim, num_layers, 999);

    // Uniform layer weights
    let uniform_prog = ProgressiveDistiller::uniform(num_layers, 2.0);
    let mse_close = uniform_prog.layer_wise_mse_loss(&student_hiddens_close, &teacher_hiddens);
    let mse_far = uniform_prog.layer_wise_mse_loss(&student_hiddens_far, &teacher_hiddens);
    let cos_close = uniform_prog.layer_wise_cosine_loss(&student_hiddens_close, &teacher_hiddens);
    let cos_far = uniform_prog.layer_wise_cosine_loss(&student_hiddens_far, &teacher_hiddens);

    println!("   Layers: {}, Hidden dim: {}", num_layers, hidden_dim);
    println!();
    println!("   Uniform layer weights:");
    println!(
        "     Close student:  MSE={:.4}  Cosine={:.4}",
        mse_close, cos_close
    );
    println!(
        "     Far student:    MSE={:.4}  Cosine={:.4}",
        mse_far, cos_far
    );
    println!();

    // Progressive weights (more weight on deeper layers)
    let progressive = ProgressiveDistiller::new(vec![0.5, 1.0, 2.0, 4.0], 2.0);
    let mse_prog = progressive.layer_wise_mse_loss(&student_hiddens_close, &teacher_hiddens);
    let cos_prog = progressive.layer_wise_cosine_loss(&student_hiddens_close, &teacher_hiddens);

    println!("   Progressive weights [0.5, 1.0, 2.0, 4.0]:");
    println!(
        "     Close student:  MSE={:.4}  Cosine={:.4}",
        mse_prog, cos_prog
    );
    println!();

    // =========================================================================
    // Section 6: Combined Loss (Logits + Hidden States)
    // =========================================================================
    println!("6. Combined Loss (Logit Distillation + Hidden Matching)");
    println!("   ─────────────────────────────────────────");

    let progressive = ProgressiveDistiller::uniform(num_layers, 3.0);

    println!(
        "   {:>6} {:>6} {:>12} {:>26}",
        "Alpha", "Beta", "Loss", "Mix"
    );
    println!("   {}", "─".repeat(55));

    for (alpha, beta) in [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.0, 1.0)] {
        let loss = progressive.combined_loss(
            &student_random,
            &teacher_logits,
            &student_hiddens_close,
            &teacher_hiddens,
            &labels,
            alpha,
            beta,
        );
        let mix = format!(
            "{:.0}% logit + {:.0}% hidden",
            (1.0 - beta) * 100.0,
            beta * 100.0
        );
        println!("   {:>6.1} {:>6.1} {:>12.4} {:>26}", alpha, beta, loss, mix);
    }
    println!();

    // =========================================================================
    // Section 7: Distillation Summary
    // =========================================================================
    println!("7. Distillation Strategy Comparison");
    println!("   ─────────────────────────────────────────");

    let basic = DistillationLoss::new(3.0, 0.7);
    let basic_loss = basic.forward(&student_random, &teacher_logits, &labels);

    let ensemble = EnsembleDistiller::uniform(3, 3.0);
    let ensemble_loss = ensemble.distillation_loss(&student_random, &teachers, &labels, 0.7);

    let progressive = ProgressiveDistiller::uniform(num_layers, 3.0);
    let progressive_loss = progressive.combined_loss(
        &student_random,
        &teacher_logits,
        &student_hiddens_close,
        &teacher_hiddens,
        &labels,
        0.7,
        0.3,
    );

    println!("   {:>20} {:>12}", "Strategy", "Loss");
    println!("   {}", "─".repeat(35));
    println!("   {:>20} {:>12.4}", "Basic KD", basic_loss);
    println!("   {:>20} {:>12.4}", "Ensemble (3T)", ensemble_loss);
    println!("   {:>20} {:>12.4}", "Progressive+KD", progressive_loss);
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_logits_shape() {
        let (logits, labels) = generate_logits(8, 5, 3.0, 42);
        assert_eq!(logits.nrows(), 8);
        assert_eq!(logits.ncols(), 5);
        assert_eq!(labels.len(), 8);
    }

    #[test]
    fn test_generate_logits_deterministic() {
        let (l1, lab1) = generate_logits(4, 3, 2.0, 42);
        let (l2, lab2) = generate_logits(4, 3, 2.0, 42);
        assert_eq!(l1, l2);
        assert_eq!(lab1, lab2);
    }

    #[test]
    fn test_generate_hidden_states_shape() {
        let hiddens = generate_hidden_states(8, 32, 4, 42);
        assert_eq!(hiddens.len(), 4);
        for h in &hiddens {
            assert_eq!(h.nrows(), 8);
            assert_eq!(h.ncols(), 32);
        }
    }

    #[test]
    fn test_distillation_loss_nonnegative() {
        let (teacher, labels) = generate_logits(8, 5, 8.0, 42);
        let (student, _) = generate_logits(8, 5, 1.0, 99);
        let loss_fn = DistillationLoss::new(3.0, 0.7);
        let loss = loss_fn.forward(&student, &teacher, &labels);
        assert!(loss >= 0.0, "Distillation loss should be non-negative");
        assert!(loss.is_finite());
    }

    #[test]
    fn test_trained_student_lower_loss() {
        let (teacher, labels) = generate_logits(16, 5, 8.0, 42);
        let (random_student, _) = generate_logits(16, 5, 0.5, 99);
        let (trained_student, _) = generate_logits(16, 5, 5.0, 42);

        let loss_fn = DistillationLoss::new(3.0, 0.7);
        let loss_random = loss_fn.forward(&random_student, &teacher, &labels);
        let loss_trained = loss_fn.forward(&trained_student, &teacher, &labels);

        assert!(
            loss_trained < loss_random,
            "Trained student loss {} should be less than random {}",
            loss_trained,
            loss_random
        );
    }

    #[test]
    fn test_ensemble_combine_shape() {
        let (t1, _) = generate_logits(8, 5, 8.0, 100);
        let (t2, _) = generate_logits(8, 5, 6.0, 200);
        let teachers = vec![t1, t2];

        let distiller = EnsembleDistiller::uniform(2, 3.0);
        let combined = distiller.combine_teachers(&teachers);
        assert_eq!(combined.nrows(), 8);
        assert_eq!(combined.ncols(), 5);
    }

    #[test]
    fn test_progressive_mse_zero_for_identical() {
        let hiddens = generate_hidden_states(4, 16, 3, 42);
        let distiller = ProgressiveDistiller::uniform(3, 2.0);
        let mse = distiller.layer_wise_mse_loss(&hiddens, &hiddens);
        assert!(mse.abs() < 1e-5, "MSE should be ~0 for identical inputs");
    }

    #[test]
    fn test_progressive_cosine_zero_for_identical() {
        let hiddens = generate_hidden_states(4, 16, 3, 42);
        let distiller = ProgressiveDistiller::uniform(3, 2.0);
        let cos_loss = distiller.layer_wise_cosine_loss(&hiddens, &hiddens);
        assert!(
            cos_loss.abs() < 1e-4,
            "Cosine loss should be ~0 for identical"
        );
    }

    #[test]
    fn test_combined_loss_finite() {
        let (teacher_logits, labels) = generate_logits(8, 5, 8.0, 42);
        let (student_logits, _) = generate_logits(8, 5, 1.0, 99);
        let teacher_hiddens = generate_hidden_states(8, 16, 3, 42);
        let student_hiddens = generate_hidden_states(8, 16, 3, 99);

        let distiller = ProgressiveDistiller::uniform(3, 3.0);
        let loss = distiller.combined_loss(
            &student_logits,
            &teacher_logits,
            &student_hiddens,
            &teacher_hiddens,
            &labels,
            0.7,
            0.3,
        );
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }
}
