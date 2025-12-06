//! # Recipe: Quantization-Aware Distillation
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
//! Distill knowledge into quantized student model.
//!
//! ## Run Command
//! ```bash
//! cargo run --example distill_quantization_aware
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("distill_quantization_aware")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Quantization-aware knowledge distillation");
    println!();

    // Baseline: FP32 teacher
    let teacher = QModelSpec {
        precision: Precision::FP32,
        accuracy: 0.92,
        size_mb: 440.0,
        latency_ms: 50.0,
    };

    println!("Teacher Model (FP32):");
    println!("  Accuracy: {:.2}%", teacher.accuracy * 100.0);
    println!("  Size: {:.1}MB", teacher.size_mb);
    println!("  Latency: {:.1}ms", teacher.latency_ms);
    println!();

    // Compare different quantization levels
    let precisions = vec![Precision::FP16, Precision::INT8, Precision::INT4];

    println!("Quantization-Aware Distillation Results:");
    println!("{:-<75}", "");
    println!(
        "{:<8} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Bits", "Accuracy", "Acc. Loss", "Size", "Latency", "Compression"
    );
    println!("{:-<75}", "");

    let mut results = Vec::new();
    for precision in &precisions {
        let result = quantize_with_distillation(&teacher, *precision)?;
        results.push(result.clone());

        let acc_loss = (teacher.accuracy - result.accuracy) * 100.0;
        let compression = teacher.size_mb / result.size_mb;

        println!(
            "{:<8} {:>11.2}% {:>11.2}% {:>10.1}MB {:>10.1}ms {:>11.1}x",
            format!("{:?}", precision),
            result.accuracy * 100.0,
            acc_loss,
            result.size_mb,
            result.latency_ms,
            compression
        );
    }
    println!("{:-<75}", "");

    // Compare with post-training quantization
    println!();
    println!("vs Post-Training Quantization (PTQ):");
    println!("{:-<55}", "");
    println!(
        "{:<8} {:>15} {:>15} {:>12}",
        "Bits", "QAT Accuracy", "PTQ Accuracy", "Improvement"
    );
    println!("{:-<55}", "");

    for (result, precision) in results.iter().zip(&precisions) {
        let ptq_accuracy = simulate_ptq(&teacher, *precision)?;
        let improvement = result.accuracy - ptq_accuracy;

        println!(
            "{:<8} {:>14.2}% {:>14.2}% {:>11.2}%",
            format!("{:?}", precision),
            result.accuracy * 100.0,
            ptq_accuracy * 100.0,
            improvement * 100.0
        );
    }
    println!("{:-<55}", "");

    // Best result
    let int8_result = results.iter().find(|r| r.precision == Precision::INT8);
    if let Some(r) = int8_result {
        ctx.record_float_metric("int8_accuracy", r.accuracy);
        ctx.record_float_metric("int8_size_mb", r.size_mb);
    }

    // Quantization schedule
    println!();
    println!("Recommended QAT Training Schedule:");
    println!("  1. Train FP32 model normally (warm-up)");
    println!("  2. Insert fake quantization operators");
    println!("  3. Fine-tune with teacher distillation");
    println!("  4. Gradually reduce precision during training");
    println!("  5. Export quantized model");

    // Save results
    let results_path = ctx.path("qat_distill.json");
    save_results(&results_path, &results)?;
    println!();
    println!("Results saved to: {:?}", results_path);

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Precision {
    FP32,
    FP16,
    INT8,
    INT4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QModelSpec {
    precision: Precision,
    accuracy: f64,
    size_mb: f64,
    latency_ms: f64,
}

fn quantize_with_distillation(
    teacher: &QModelSpec,
    target_precision: Precision,
) -> Result<QModelSpec> {
    let (bits, accuracy_penalty) = match target_precision {
        Precision::FP32 => (32, 0.0),
        Precision::FP16 => (16, 0.005), // 0.5% loss
        Precision::INT8 => (8, 0.015),  // 1.5% loss
        Precision::INT4 => (4, 0.04),   // 4% loss
    };

    // Size scales with bits
    let size = teacher.size_mb * (f64::from(bits) / 32.0);

    // Latency improves with lower precision
    let latency_factor = match target_precision {
        Precision::FP32 => 1.0,
        Precision::FP16 => 0.6,
        Precision::INT8 => 0.35,
        Precision::INT4 => 0.25,
    };
    let latency = teacher.latency_ms * latency_factor;

    // Accuracy with distillation-aware training
    let accuracy = teacher.accuracy - accuracy_penalty;

    Ok(QModelSpec {
        precision: target_precision,
        accuracy,
        size_mb: size,
        latency_ms: latency,
    })
}

fn simulate_ptq(teacher: &QModelSpec, precision: Precision) -> Result<f64> {
    // PTQ has higher accuracy loss than QAT
    let accuracy_penalty = match precision {
        Precision::FP32 => 0.0,
        Precision::FP16 => 0.01, // 1% loss
        Precision::INT8 => 0.04, // 4% loss
        Precision::INT4 => 0.12, // 12% loss
    };

    Ok(teacher.accuracy - accuracy_penalty)
}

fn save_results(path: &std::path::Path, results: &[QModelSpec]) -> Result<()> {
    let json = serde_json::to_string_pretty(results)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn teacher_model() -> QModelSpec {
        QModelSpec {
            precision: Precision::FP32,
            accuracy: 0.90,
            size_mb: 400.0,
            latency_ms: 50.0,
        }
    }

    #[test]
    fn test_fp16_quantization() {
        let teacher = teacher_model();
        let result = quantize_with_distillation(&teacher, Precision::FP16).unwrap();

        assert_eq!(result.precision, Precision::FP16);
        assert!(result.size_mb < teacher.size_mb);
    }

    #[test]
    fn test_int8_quantization() {
        let teacher = teacher_model();
        let result = quantize_with_distillation(&teacher, Precision::INT8).unwrap();

        // INT8 should be ~4x smaller than FP32
        assert!(result.size_mb < teacher.size_mb / 3.0);
    }

    #[test]
    fn test_accuracy_loss_increases() {
        let teacher = teacher_model();

        let fp16 = quantize_with_distillation(&teacher, Precision::FP16).unwrap();
        let int8 = quantize_with_distillation(&teacher, Precision::INT8).unwrap();
        let int4 = quantize_with_distillation(&teacher, Precision::INT4).unwrap();

        assert!(fp16.accuracy > int8.accuracy);
        assert!(int8.accuracy > int4.accuracy);
    }

    #[test]
    fn test_latency_improves() {
        let teacher = teacher_model();
        let result = quantize_with_distillation(&teacher, Precision::INT8).unwrap();

        assert!(result.latency_ms < teacher.latency_ms);
    }

    #[test]
    fn test_qat_better_than_ptq() {
        let teacher = teacher_model();

        let qat = quantize_with_distillation(&teacher, Precision::INT8).unwrap();
        let ptq = simulate_ptq(&teacher, Precision::INT8).unwrap();

        assert!(qat.accuracy > ptq);
    }

    #[test]
    fn test_deterministic() {
        let teacher = teacher_model();

        let r1 = quantize_with_distillation(&teacher, Precision::INT8).unwrap();
        let r2 = quantize_with_distillation(&teacher, Precision::INT8).unwrap();

        assert_eq!(r1.accuracy, r2.accuracy);
        assert_eq!(r1.size_mb, r2.size_mb);
    }

    #[test]
    fn test_save_results() {
        let ctx = RecipeContext::new("test_qat_save").unwrap();
        let path = ctx.path("results.json");

        let results = vec![teacher_model()];
        save_results(&path, &results).unwrap();

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
        fn prop_size_decreases_with_precision(
            teacher_size in 100.0f64..1000.0,
            precision_idx in 1usize..4
        ) {
            let teacher = QModelSpec {
                precision: Precision::FP32,
                accuracy: 0.90,
                size_mb: teacher_size,
                latency_ms: 50.0,
            };

            let precisions = [Precision::FP16, Precision::INT8, Precision::INT4];
            let result = quantize_with_distillation(&teacher, precisions[precision_idx - 1]).unwrap();

            prop_assert!(result.size_mb < teacher.size_mb);
        }

        #[test]
        fn prop_accuracy_bounded(teacher_acc in 0.7f64..0.99) {
            let teacher = QModelSpec {
                precision: Precision::FP32,
                accuracy: teacher_acc,
                size_mb: 400.0,
                latency_ms: 50.0,
            };

            let result = quantize_with_distillation(&teacher, Precision::INT8).unwrap();

            prop_assert!(result.accuracy >= 0.0);
            prop_assert!(result.accuracy <= 1.0);
            prop_assert!(result.accuracy <= teacher_acc);
        }
    }
}
