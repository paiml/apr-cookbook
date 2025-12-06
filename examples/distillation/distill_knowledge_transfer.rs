//! # Recipe: Knowledge Distillation
//!
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
//! Transfer knowledge from teacher to student model.
//!
//! ## Run Command
//! ```bash
//! cargo run --example distill_knowledge_transfer
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("distill_knowledge_transfer")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Knowledge distillation: Teacher -> Student");
    println!();

    // Teacher model (large)
    let teacher = ModelSpec {
        name: "teacher".to_string(),
        layers: 12,
        hidden_size: 768,
        params_millions: 110.0,
    };

    // Student model (small)
    let student = ModelSpec {
        name: "student".to_string(),
        layers: 4,
        hidden_size: 256,
        params_millions: 6.5,
    };

    println!("Teacher Model:");
    println!("  Layers: {}", teacher.layers);
    println!("  Hidden: {}", teacher.hidden_size);
    println!("  Parameters: {:.1}M", teacher.params_millions);
    println!();

    println!("Student Model:");
    println!("  Layers: {}", student.layers);
    println!("  Hidden: {}", student.hidden_size);
    println!("  Parameters: {:.1}M", student.params_millions);
    println!();

    let compression_ratio = teacher.params_millions / student.params_millions;
    ctx.record_float_metric("compression_ratio", compression_ratio);

    // Distillation config
    let config = DistillationConfig {
        temperature: 4.0,
        alpha: 0.7, // Weight for soft targets
        epochs: 10,
    };

    println!("Distillation Config:");
    println!("  Temperature: {}", config.temperature);
    println!("  Alpha (soft target weight): {}", config.alpha);
    println!("  Epochs: {}", config.epochs);
    println!();

    // Run distillation simulation
    println!("Distillation Progress:");
    println!("{:-<60}", "");
    println!(
        "{:>6} {:>15} {:>15} {:>15}",
        "Epoch", "Teacher Acc", "Student Acc", "KD Loss"
    );
    println!("{:-<60}", "");

    let mut distillation_log = Vec::new();
    for epoch in 1..=config.epochs {
        let result = simulate_distillation_epoch(epoch, &config)?;
        distillation_log.push(result.clone());

        println!(
            "{:>6} {:>14.2}% {:>14.2}% {:>15.4}",
            epoch,
            result.teacher_accuracy * 100.0,
            result.student_accuracy * 100.0,
            result.distillation_loss
        );
    }
    println!("{:-<60}", "");

    // Final results
    let final_result = distillation_log
        .last()
        .ok_or_else(|| CookbookError::invalid_format("No results"))?;

    ctx.record_float_metric("final_student_accuracy", final_result.student_accuracy);

    println!();
    println!("Results:");
    println!(
        "  Teacher accuracy: {:.2}%",
        final_result.teacher_accuracy * 100.0
    );
    println!(
        "  Student accuracy: {:.2}%",
        final_result.student_accuracy * 100.0
    );
    println!(
        "  Knowledge retention: {:.1}%",
        (final_result.student_accuracy / final_result.teacher_accuracy) * 100.0
    );
    println!("  Compression: {:.1}x fewer parameters", compression_ratio);
    println!(
        "  Speedup: {:.1}x faster inference",
        teacher.params_millions / student.params_millions
    );

    // Save distillation log
    let log_path = ctx.path("distillation_log.json");
    save_log(&log_path, &distillation_log)?;
    println!();
    println!("Log saved to: {:?}", log_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelSpec {
    name: String,
    layers: u32,
    hidden_size: u32,
    params_millions: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DistillationConfig {
    temperature: f64,
    alpha: f64,
    epochs: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EpochResult {
    epoch: u32,
    teacher_accuracy: f64,
    student_accuracy: f64,
    distillation_loss: f64,
}

fn simulate_distillation_epoch(
    epoch: u32,
    config: &DistillationConfig,
) -> Result<EpochResult> {
    // Simulated learning curve (deterministic)
    let progress = f64::from(epoch) / f64::from(config.epochs);

    // Teacher accuracy is constant (already trained)
    let teacher_accuracy = 0.92;

    // Student learns progressively with diminishing returns
    let max_student_accuracy = 0.88; // Can't quite match teacher
    let student_accuracy = max_student_accuracy * (1.0 - (-3.0 * progress).exp());

    // Distillation loss decreases
    let initial_loss = 2.5;
    let final_loss = 0.3;
    let distillation_loss = initial_loss - (initial_loss - final_loss) * progress;

    Ok(EpochResult {
        epoch,
        teacher_accuracy,
        student_accuracy,
        distillation_loss,
    })
}

fn save_log(path: &std::path::Path, log: &[EpochResult]) -> Result<()> {
    let json = serde_json::to_string_pretty(log)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distillation_epoch() {
        let config = DistillationConfig {
            temperature: 4.0,
            alpha: 0.7,
            epochs: 10,
        };

        let result = simulate_distillation_epoch(5, &config).unwrap();

        assert!(result.student_accuracy > 0.0);
        assert!(result.teacher_accuracy > 0.0);
    }

    #[test]
    fn test_student_improves() {
        let config = DistillationConfig {
            temperature: 4.0,
            alpha: 0.7,
            epochs: 10,
        };

        let early = simulate_distillation_epoch(1, &config).unwrap();
        let late = simulate_distillation_epoch(10, &config).unwrap();

        assert!(late.student_accuracy > early.student_accuracy);
    }

    #[test]
    fn test_loss_decreases() {
        let config = DistillationConfig {
            temperature: 4.0,
            alpha: 0.7,
            epochs: 10,
        };

        let early = simulate_distillation_epoch(1, &config).unwrap();
        let late = simulate_distillation_epoch(10, &config).unwrap();

        assert!(late.distillation_loss < early.distillation_loss);
    }

    #[test]
    fn test_teacher_constant() {
        let config = DistillationConfig {
            temperature: 4.0,
            alpha: 0.7,
            epochs: 10,
        };

        let r1 = simulate_distillation_epoch(1, &config).unwrap();
        let r2 = simulate_distillation_epoch(10, &config).unwrap();

        assert_eq!(r1.teacher_accuracy, r2.teacher_accuracy);
    }

    #[test]
    fn test_deterministic() {
        let config = DistillationConfig {
            temperature: 4.0,
            alpha: 0.7,
            epochs: 10,
        };

        let r1 = simulate_distillation_epoch(5, &config).unwrap();
        let r2 = simulate_distillation_epoch(5, &config).unwrap();

        assert_eq!(r1.student_accuracy, r2.student_accuracy);
    }

    #[test]
    fn test_save_log() {
        let ctx = RecipeContext::new("test_distill_save").unwrap();
        let path = ctx.path("log.json");

        let log = vec![EpochResult {
            epoch: 1,
            teacher_accuracy: 0.9,
            student_accuracy: 0.5,
            distillation_loss: 1.0,
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
        fn prop_student_improves_over_time(epoch in 1u32..100) {
            let config = DistillationConfig {
                temperature: 4.0,
                alpha: 0.7,
                epochs: 100,
            };

            let result = simulate_distillation_epoch(epoch, &config).unwrap();

            // Student accuracy should be between 0 and teacher
            prop_assert!(result.student_accuracy >= 0.0);
            prop_assert!(result.student_accuracy <= result.teacher_accuracy);
        }

        #[test]
        fn prop_accuracy_bounded(epoch in 1u32..50) {
            let config = DistillationConfig {
                temperature: 4.0,
                alpha: 0.7,
                epochs: 50,
            };

            let result = simulate_distillation_epoch(epoch, &config).unwrap();

            prop_assert!(result.student_accuracy >= 0.0);
            prop_assert!(result.student_accuracy <= 1.0);
            prop_assert!(result.teacher_accuracy >= 0.0);
            prop_assert!(result.teacher_accuracy <= 1.0);
        }
    }
}
