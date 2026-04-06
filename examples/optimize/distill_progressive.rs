//! # Recipe: Progressive Layer-wise Distillation
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
//! Progressive layer-wise distillation with hidden state matching.
//! CLI equivalent: `apr distill --strategy progressive`
//!
//! ## Run Command
//! ```bash
//! cargo run --example distill_progressive
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
use entrenar::distill::ProgressiveDistiller;
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

/// Generate synthetic hidden states for each layer of a model.
fn generate_hidden_states(
    batch_size: usize,
    hidden_dim: usize,
    num_layers: usize,
    seed: u64,
) -> Vec<Array2<f32>> {
    (0..num_layers)
        .map(|layer| {
            let mut data = Vec::with_capacity(batch_size * hidden_dim);
            for b in 0..batch_size {
                for h in 0..hidden_dim {
                    let mut hasher = DefaultHasher::new();
                    (seed, layer, b, h).hash(&mut hasher);
                    let val = hasher.finish() as f32 / u64::MAX as f32 - 0.5;
                    data.push(val);
                }
            }
            Array2::from_shape_vec((batch_size, hidden_dim), data).expect("valid shape")
        })
        .collect()
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("distill_progressive")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Progressive Layer-wise Distillation with Hidden State Matching");
    println!();

    let batch_size = 32;
    let hidden_dim = 64;
    let num_layers = 4;
    let num_classes = 10;

    // ── Section 1: Hidden State Generation ───────────────────────────────
    println!("--- Generating teacher and student hidden states ---");

    let teacher_hidden = generate_hidden_states(batch_size, hidden_dim, num_layers, 42);
    let student_hidden = generate_hidden_states(batch_size, hidden_dim, num_layers, 99);

    for (i, (t, s)) in teacher_hidden.iter().zip(student_hidden.iter()).enumerate() {
        println!(
            "Layer {i}: teacher shape {:?}, student shape {:?}",
            t.shape(),
            s.shape()
        );
    }
    println!();

    // ── Section 2: MSE Matching ──────────────────────────────────────────
    println!("--- Layer-wise MSE matching (magnitude alignment) ---");

    let distiller = ProgressiveDistiller::uniform(num_layers, 4.0);
    let mse_loss = distiller.layer_wise_mse_loss(&student_hidden, &teacher_hidden);

    println!("MSE loss (uniform weights): {mse_loss:.6}");

    let mse_identical = distiller.layer_wise_mse_loss(&teacher_hidden, &teacher_hidden);
    println!("MSE loss (identical states): {mse_identical:.6}");
    println!();

    ctx.record_float_metric("mse_loss", f64::from(mse_loss));

    // ── Section 3: Cosine Matching ───────────────────────────────────────
    println!("--- Layer-wise cosine matching (directional alignment) ---");

    let cosine_loss = distiller.layer_wise_cosine_loss(&student_hidden, &teacher_hidden);
    println!("Cosine loss (uniform weights): {cosine_loss:.6}");

    let cosine_identical = distiller.layer_wise_cosine_loss(&teacher_hidden, &teacher_hidden);
    println!("Cosine loss (identical states): {cosine_identical:.6}");
    println!();

    ctx.record_float_metric("cosine_loss", f64::from(cosine_loss));

    // ── Section 4: Uniform vs Progressive Weights ────────────────────────
    println!("--- Uniform vs progressive layer weights ---");

    let uniform_mse = distiller.layer_wise_mse_loss(&student_hidden, &teacher_hidden);

    // Progressive weights emphasize deeper layers
    let progressive_weights = vec![0.5, 1.0, 2.0, 4.0];
    println!("Progressive weights: {progressive_weights:?}");
    println!("Deeper layers get higher weight to preserve abstract representations");

    let progressive_distiller = ProgressiveDistiller::new(progressive_weights.clone(), 4.0);
    let progressive_mse =
        progressive_distiller.layer_wise_mse_loss(&student_hidden, &teacher_hidden);

    println!("Uniform MSE:     {uniform_mse:.6}");
    println!("Progressive MSE: {progressive_mse:.6}");

    // Show per-layer contribution
    for (i, w) in progressive_weights.iter().enumerate() {
        let layer_t = &teacher_hidden[i];
        let layer_s = &student_hidden[i];
        let diff = layer_t - layer_s;
        let raw_mse = diff.mapv(|x| x * x).mean().unwrap_or(0.0);
        println!(
            "  Layer {i}: raw_mse={raw_mse:.4}, weight={w:.1}, weighted={:.4}",
            raw_mse * w
        );
    }
    println!();

    // ── Section 5: Combined Logit + Hidden Loss ──────────────────────────
    println!("--- Combined distillation: logit KL + hidden state matching ---");

    let (teacher_logits, labels) = generate_logits(batch_size, num_classes, 3.0, 42);
    let (student_logits, _) = generate_logits(batch_size, num_classes, 1.0, 99);

    let alpha = 0.7;
    let beta = 0.3;

    let combined = distiller.combined_loss(
        &student_logits,
        &teacher_logits,
        &student_hidden,
        &teacher_hidden,
        &labels,
        alpha,
        beta,
    );

    let hidden_mse = distiller.layer_wise_mse_loss(&student_hidden, &teacher_hidden);
    let hidden_cosine = distiller.layer_wise_cosine_loss(&student_hidden, &teacher_hidden);

    println!("Hidden MSE loss:  {hidden_mse:.6}");
    println!("Hidden cos loss:  {hidden_cosine:.6}");
    println!("Combined loss (alpha={alpha}, beta={beta}): {combined:.6}");

    ctx.record_float_metric("combined_loss", f64::from(combined));

    // ── Summary ──────────────────────────────────────────────────────────
    println!();
    println!("--- Summary ---");
    println!("Progressive distillation matches hidden representations layer by layer.");
    println!("MSE preserves activation magnitude; cosine preserves direction.");
    println!("Progressive weights [0.5, 1.0, 2.0, 4.0] emphasize deeper layers.");
    println!("Combined loss joins logit distillation with hidden state alignment.");
    println!();

    ctx.report()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_zero_for_identical() {
        let hidden = generate_hidden_states(16, 32, 4, 42);
        let distiller = ProgressiveDistiller::uniform(4, 4.0);
        let loss = distiller.layer_wise_mse_loss(&hidden, &hidden);
        assert!(
            loss.abs() < 1e-5,
            "MSE of identical states should be ~0, got {loss}"
        );
    }

    #[test]
    fn test_cosine_zero_for_identical() {
        let hidden = generate_hidden_states(16, 32, 4, 42);
        let distiller = ProgressiveDistiller::uniform(4, 4.0);
        let loss = distiller.layer_wise_cosine_loss(&hidden, &hidden);
        assert!(
            loss.abs() < 1e-4,
            "Cosine loss of identical states should be ~0, got {loss}"
        );
    }

    #[test]
    fn test_mse_positive_for_different() {
        let teacher = generate_hidden_states(16, 32, 4, 42);
        let student = generate_hidden_states(16, 32, 4, 99);
        let distiller = ProgressiveDistiller::uniform(4, 4.0);
        let loss = distiller.layer_wise_mse_loss(&student, &teacher);
        assert!(loss > 0.0, "MSE of different states should be positive");
    }

    #[test]
    fn test_cosine_positive_for_different() {
        let teacher = generate_hidden_states(16, 32, 4, 42);
        let student = generate_hidden_states(16, 32, 4, 99);
        let distiller = ProgressiveDistiller::uniform(4, 4.0);
        let loss = distiller.layer_wise_cosine_loss(&student, &teacher);
        assert!(
            loss > 0.0,
            "Cosine loss of different states should be positive"
        );
    }

    #[test]
    fn test_combined_loss_finite() {
        let (teacher_logits, labels) = generate_logits(16, 5, 3.0, 42);
        let (student_logits, _) = generate_logits(16, 5, 1.0, 99);
        let teacher_hidden = generate_hidden_states(16, 32, 4, 42);
        let student_hidden = generate_hidden_states(16, 32, 4, 99);

        let distiller = ProgressiveDistiller::uniform(4, 4.0);
        let loss = distiller.combined_loss(
            &student_logits,
            &teacher_logits,
            &student_hidden,
            &teacher_hidden,
            &labels,
            0.7,
            0.3,
        );
        assert!(loss.is_finite(), "combined loss must be finite");
    }

    #[test]
    fn test_combined_greater_than_zero() {
        let (teacher_logits, labels) = generate_logits(16, 5, 3.0, 42);
        let (student_logits, _) = generate_logits(16, 5, 1.0, 99);
        let teacher_hidden = generate_hidden_states(16, 32, 4, 42);
        let student_hidden = generate_hidden_states(16, 32, 4, 99);

        let distiller = ProgressiveDistiller::uniform(4, 4.0);
        let combined = distiller.combined_loss(
            &student_logits,
            &teacher_logits,
            &student_hidden,
            &teacher_hidden,
            &labels,
            0.7,
            0.3,
        );

        assert!(
            combined > 0.0,
            "combined loss for different student/teacher should be > 0, got {combined:.4}"
        );
    }

    #[test]
    fn test_generate_hidden_states_shape() {
        let hidden = generate_hidden_states(8, 32, 4, 42);
        assert_eq!(hidden.len(), 4);
        for layer in &hidden {
            assert_eq!(layer.shape(), &[8, 32]);
        }
    }

    #[test]
    fn test_hidden_states_deterministic() {
        let h1 = generate_hidden_states(8, 16, 3, 42);
        let h2 = generate_hidden_states(8, 16, 3, 42);
        for (a, b) in h1.iter().zip(h2.iter()) {
            assert_eq!(a, b, "same seed must produce identical hidden states");
        }
    }

    #[test]
    fn test_hidden_states_different_seeds() {
        let h1 = generate_hidden_states(8, 16, 3, 42);
        let h2 = generate_hidden_states(8, 16, 3, 99);
        let differs = h1.iter().zip(h2.iter()).any(|(a, b)| a != b);
        assert!(
            differs,
            "different seeds must produce different hidden states"
        );
    }

    #[test]
    fn test_mse_symmetric() {
        let a = generate_hidden_states(16, 32, 4, 42);
        let b = generate_hidden_states(16, 32, 4, 99);
        let distiller = ProgressiveDistiller::uniform(4, 4.0);
        let loss_ab = distiller.layer_wise_mse_loss(&a, &b);
        let loss_ba = distiller.layer_wise_mse_loss(&b, &a);
        assert!(
            (loss_ab - loss_ba).abs() < 1e-5,
            "MSE should be symmetric: {loss_ab} vs {loss_ba}"
        );
    }

    #[test]
    fn test_single_layer() {
        let teacher = generate_hidden_states(8, 16, 1, 42);
        let student = generate_hidden_states(8, 16, 1, 99);
        let distiller = ProgressiveDistiller::uniform(1, 4.0);
        let loss = distiller.layer_wise_mse_loss(&student, &teacher);
        assert!(loss.is_finite() && loss > 0.0);
    }

    #[test]
    fn test_progressive_vs_uniform_weights_differ() {
        let teacher = generate_hidden_states(16, 32, 4, 42);
        let student = generate_hidden_states(16, 32, 4, 99);

        let uniform = ProgressiveDistiller::uniform(4, 4.0);
        let progressive = ProgressiveDistiller::new(vec![0.5, 1.0, 2.0, 4.0], 4.0);

        let loss_u = uniform.layer_wise_mse_loss(&student, &teacher);
        let loss_p = progressive.layer_wise_mse_loss(&student, &teacher);

        assert!(
            (loss_u - loss_p).abs() > 1e-6,
            "progressive and uniform weights should produce different losses"
        );
    }
}
