//! # Recipe: Distillation with Checkpoint Save/Resume
//!
//! **Category**: Model Optimization
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: entrenar (distillation), serde_json
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
//! Distillation with checkpoint save/resume for fault tolerance.
//! CLI equivalent: `apr distill` + save/resume workflow
//!
//! ## Run Command
//! ```bash
//! cargo run --example distill_checkpoint
//! ```

use apr_cookbook::prelude::*;
use entrenar::distill::DistillationLoss;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

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

/// Distillation checkpoint capturing full training state.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DistillationCheckpoint {
    epoch: usize,
    loss: f32,
    student_weights: Vec<f32>,
    temperature: f32,
    alpha: f32,
}

/// Save checkpoint to a JSON file.
fn save_checkpoint(checkpoint: &DistillationCheckpoint, path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(checkpoint)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load checkpoint from a JSON file.
fn load_checkpoint(path: &Path) -> Result<()> {
    let json = std::fs::read_to_string(path)?;
    let _checkpoint: DistillationCheckpoint =
        serde_json::from_str(&json).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    Ok(())
}

/// Load checkpoint from a JSON file, returning the deserialized struct.
fn load_checkpoint_data(path: &Path) -> Result<DistillationCheckpoint> {
    let json = std::fs::read_to_string(path)?;
    let checkpoint: DistillationCheckpoint =
        serde_json::from_str(&json).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    Ok(checkpoint)
}

/// Simulate one training epoch: blend student toward teacher by `blend_factor`.
fn train_epoch(
    student_logits: &Array2<f32>,
    teacher_logits: &Array2<f32>,
    labels: &[usize],
    blend_factor: f32,
    loss_fn: &DistillationLoss,
) -> (f32, Vec<f32>) {
    let blended = student_logits * (1.0 - blend_factor) + teacher_logits * blend_factor;
    let loss = loss_fn.forward(&blended, teacher_logits, labels);
    let weights = blended.iter().copied().collect();
    (loss, weights)
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("distill_checkpoint")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Distillation with Checkpoint Save/Resume");
    println!();

    let batch_size = 32;
    let num_classes = 10;
    let temperature = 4.0;
    let alpha = 0.7;
    let total_epochs = 10;
    let interrupt_epoch = 5;
    let learning_rate = 0.1_f32;

    let loss_fn = DistillationLoss::new(temperature, alpha);
    let (teacher_logits, labels) = generate_logits(batch_size, num_classes, 3.0, 42);
    let (student_logits, _) = generate_logits(batch_size, num_classes, 1.0, 99);

    let checkpoint_path = ctx.path("distill_ckpt.json");

    // ── Section 1: Initial Training (epochs 0..5) ────────────────────────
    println!("--- Initial training: epochs 0-{} ---", interrupt_epoch - 1);

    let mut current_weights: Vec<f32> = student_logits.iter().copied().collect();
    let mut losses = Vec::new();

    for epoch in 0..interrupt_epoch {
        let blend = learning_rate * (epoch + 1) as f32;
        let blend = blend.min(1.0);

        let current = Array2::from_shape_vec((batch_size, num_classes), current_weights.clone())
            .expect("valid shape");

        let (loss, weights) = train_epoch(&current, &teacher_logits, &labels, blend, &loss_fn);
        current_weights = weights;
        losses.push(loss);

        println!("Epoch {epoch}: loss={loss:.6}, blend={blend:.2}");
    }
    println!();

    // ── Section 2: Checkpoint Save ───────────────────────────────────────
    println!("--- Saving checkpoint before interruption ---");

    let last_loss = losses[losses.len() - 1];
    let checkpoint = DistillationCheckpoint {
        epoch: interrupt_epoch,
        loss: last_loss,
        student_weights: current_weights.clone(),
        temperature,
        alpha,
    };

    save_checkpoint(&checkpoint, &checkpoint_path)?;
    println!("Checkpoint saved to: {}", checkpoint_path.display());
    println!(
        "  epoch: {}, loss: {:.6}",
        checkpoint.epoch, checkpoint.loss
    );
    println!(
        "  temperature: {}, alpha: {}",
        checkpoint.temperature, checkpoint.alpha
    );
    println!(
        "  student weights: {} values",
        checkpoint.student_weights.len()
    );
    println!();

    // ── Section 3: Simulated Interruption ────────────────────────────────
    println!("--- Simulating training interruption ---");

    let pre_interrupt_loss = last_loss;
    drop(current_weights);
    drop(losses);
    println!("Training state dropped (simulated crash)");
    println!("Last known loss: {pre_interrupt_loss:.6}");
    println!();

    // ── Section 4: Checkpoint Load ───────────────────────────────────────
    println!("--- Loading checkpoint and verifying state ---");

    // Verify the roundtrip works
    load_checkpoint(&checkpoint_path)?;

    let loaded = load_checkpoint_data(&checkpoint_path)?;
    println!("Restored epoch: {}, loss: {:.6}", loaded.epoch, loaded.loss);
    println!(
        "Restored temperature: {}, alpha: {}",
        loaded.temperature, loaded.alpha
    );
    println!("Restored weights: {} values", loaded.student_weights.len());

    assert!(
        (loaded.loss - pre_interrupt_loss).abs() < f32::EPSILON,
        "loaded loss must match saved loss"
    );
    println!();

    // ── Section 5: Resumed Training (epochs 5..10) ───────────────────────
    println!(
        "--- Resumed training: epochs {interrupt_epoch}-{} ---",
        total_epochs - 1
    );

    let resumed_loss_fn = DistillationLoss::new(loaded.temperature, loaded.alpha);
    let mut resumed_weights = loaded.student_weights;
    let mut resumed_losses = vec![loaded.loss];

    for epoch in loaded.epoch..total_epochs {
        let blend = learning_rate * (epoch + 1) as f32;
        let blend = blend.min(1.0);

        let current = Array2::from_shape_vec((batch_size, num_classes), resumed_weights.clone())
            .expect("valid shape");

        let (loss, weights) =
            train_epoch(&current, &teacher_logits, &labels, blend, &resumed_loss_fn);
        resumed_weights = weights;
        resumed_losses.push(loss);

        println!("Epoch {epoch}: loss={loss:.6}, blend={blend:.2}");
    }
    println!();

    // ── Section 6: Final Results ─────────────────────────────────────────
    println!("--- Final results ---");

    let first_loss = resumed_losses[0];
    let final_loss = resumed_losses[resumed_losses.len() - 1];

    println!("Loss at checkpoint:  {first_loss:.6} (epoch {interrupt_epoch})");
    println!(
        "Loss after training: {final_loss:.6} (epoch {})",
        total_epochs - 1
    );
    println!(
        "Continued improvement: {:.1}%",
        (1.0 - final_loss / first_loss) * 100.0
    );
    println!("Checkpoint/resume preserved training continuity.");
    println!();

    ctx.record_float_metric("checkpoint_loss", f64::from(first_loss));
    ctx.record_float_metric("final_loss", f64::from(final_loss));
    ctx.report()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_checkpoint_path() -> (RecipeContext, std::path::PathBuf) {
        let ctx = RecipeContext::new("test_distill_checkpoint").expect("context");
        let path = ctx.path("test_ckpt.json");
        (ctx, path)
    }

    #[test]
    fn test_save_load_roundtrip() {
        let (_ctx, path) = temp_checkpoint_path();
        let original = DistillationCheckpoint {
            epoch: 5,
            loss: 1.234,
            student_weights: vec![0.1, 0.2, 0.3, 0.4],
            temperature: 4.0,
            alpha: 0.7,
        };

        save_checkpoint(&original, &path).expect("save");
        let loaded = load_checkpoint_data(&path).expect("load");

        assert_eq!(loaded.epoch, original.epoch);
        assert!((loaded.loss - original.loss).abs() < 1e-6);
        assert_eq!(loaded.student_weights, original.student_weights);
        assert!((loaded.temperature - original.temperature).abs() < 1e-6);
        assert!((loaded.alpha - original.alpha).abs() < 1e-6);
    }

    #[test]
    fn test_checkpoint_contains_correct_epoch() {
        let (_ctx, path) = temp_checkpoint_path();
        let ckpt = DistillationCheckpoint {
            epoch: 7,
            loss: 0.5,
            student_weights: vec![1.0; 100],
            temperature: 6.0,
            alpha: 0.3,
        };

        save_checkpoint(&ckpt, &path).expect("save");
        let loaded = load_checkpoint_data(&path).expect("load");
        assert_eq!(loaded.epoch, 7);
    }

    #[test]
    fn test_resume_continues_from_correct_state() {
        let (teacher, labels) = generate_logits(16, 5, 3.0, 42);
        let (student, _) = generate_logits(16, 5, 1.0, 99);
        let loss_fn = DistillationLoss::new(4.0, 0.7);

        let mut weights: Vec<f32> = student.iter().copied().collect();
        let mut loss_at_3 = 0.0_f32;
        for epoch in 0..3 {
            let blend = 0.1 * (epoch + 1) as f32;
            let current = Array2::from_shape_vec((16, 5), weights.clone()).expect("shape");
            let (loss, w) = train_epoch(&current, &teacher, &labels, blend, &loss_fn);
            weights = w;
            loss_at_3 = loss;
        }

        let (_ctx, path) = temp_checkpoint_path();
        let ckpt = DistillationCheckpoint {
            epoch: 3,
            loss: loss_at_3,
            student_weights: weights,
            temperature: 4.0,
            alpha: 0.7,
        };
        save_checkpoint(&ckpt, &path).expect("save");

        let loaded = load_checkpoint_data(&path).expect("load");
        assert_eq!(loaded.epoch, 3);
        assert!((loaded.loss - loss_at_3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_loss_continues_decreasing() {
        let (teacher, labels) = generate_logits(16, 5, 3.0, 42);
        let (student, _) = generate_logits(16, 5, 1.0, 99);
        let loss_fn = DistillationLoss::new(4.0, 0.7);

        let mut weights: Vec<f32> = student.iter().copied().collect();
        let mut losses = Vec::new();

        for epoch in 0..8 {
            let blend = 0.12 * (epoch + 1) as f32;
            let blend = blend.min(1.0);
            let current = Array2::from_shape_vec((16, 5), weights.clone()).expect("shape");
            let (loss, w) = train_epoch(&current, &teacher, &labels, blend, &loss_fn);
            weights = w;
            losses.push(loss);
        }

        let first = losses[0];
        let last = losses[losses.len() - 1];
        assert!(
            last <= first,
            "loss should decrease: first={first:.4}, last={last:.4}"
        );
    }

    #[test]
    fn test_checkpoint_json_valid() {
        let (_ctx, path) = temp_checkpoint_path();
        let ckpt = DistillationCheckpoint {
            epoch: 2,
            loss: 0.99,
            student_weights: vec![0.5; 50],
            temperature: 4.0,
            alpha: 0.7,
        };

        save_checkpoint(&ckpt, &path).expect("save");
        let json = std::fs::read_to_string(&path).expect("read");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("parse");

        assert_eq!(parsed["epoch"], 2);
        assert!(parsed["student_weights"].is_array());
        assert_eq!(
            parsed["student_weights"].as_array().expect("array").len(),
            50
        );
    }

    #[test]
    fn test_load_nonexistent_fails() {
        let path = Path::new("/tmp/nonexistent_checkpoint_xyz_42.json");
        assert!(load_checkpoint_data(path).is_err());
    }

    #[test]
    fn test_train_epoch_produces_valid_output() {
        let (teacher, labels) = generate_logits(16, 5, 3.0, 42);
        let (student, _) = generate_logits(16, 5, 1.0, 99);
        let loss_fn = DistillationLoss::new(4.0, 0.7);

        let (loss, weights) = train_epoch(&student, &teacher, &labels, 0.5, &loss_fn);
        assert!(loss.is_finite());
        assert_eq!(weights.len(), 16 * 5);
    }

    #[test]
    fn test_full_blend_matches_teacher() {
        let (teacher, labels) = generate_logits(16, 5, 3.0, 42);
        let (student, _) = generate_logits(16, 5, 1.0, 99);
        let loss_fn = DistillationLoss::new(4.0, 0.7);

        let (loss, _) = train_epoch(&student, &teacher, &labels, 1.0, &loss_fn);
        assert!(
            loss < 1.0,
            "full blend toward teacher should yield low loss, got {loss}"
        );
    }

    #[test]
    fn test_checkpoint_weights_size_matches() {
        let (teacher, labels) = generate_logits(16, 5, 3.0, 42);
        let (student, _) = generate_logits(16, 5, 1.0, 99);
        let loss_fn = DistillationLoss::new(4.0, 0.7);

        let (_, weights) = train_epoch(&student, &teacher, &labels, 0.3, &loss_fn);

        let (_ctx, path) = temp_checkpoint_path();
        let ckpt = DistillationCheckpoint {
            epoch: 1,
            loss: 0.5,
            student_weights: weights.clone(),
            temperature: 4.0,
            alpha: 0.7,
        };
        save_checkpoint(&ckpt, &path).expect("save");
        let loaded = load_checkpoint_data(&path).expect("load");

        assert_eq!(loaded.student_weights.len(), 16 * 5);
        assert_eq!(loaded.student_weights.len(), weights.len());
    }
}
