//! # Recipe: Pruning-Aware Distillation
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
//! Combine pruning with distillation for extreme compression.
//!
//! ## Run Command
//! ```bash
//! cargo run --example distill_pruning_aware
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("distill_pruning_aware")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Pruning-aware knowledge distillation");
    println!();

    // Original model
    let original = ModelStats {
        params_millions: 110.0,
        accuracy: 0.92,
        size_mb: 440.0,
        latency_ms: 50.0,
    };

    println!("Original Model:");
    println!("  Parameters: {:.1}M", original.params_millions);
    println!("  Accuracy: {:.2}%", original.accuracy * 100.0);
    println!("  Size: {:.1}MB", original.size_mb);
    println!("  Latency: {:.1}ms", original.latency_ms);
    println!();

    // Pruning schedules to compare
    let sparsities = vec![0.0, 0.5, 0.7, 0.9];

    println!("Pruning + Distillation Results:");
    println!("{:-<70}", "");
    println!(
        "{:>10} {:>12} {:>12} {:>12} {:>12}",
        "Sparsity", "Params", "Accuracy", "Size", "Speedup"
    );
    println!("{:-<70}", "");

    let mut results = Vec::new();
    for sparsity in &sparsities {
        let result = apply_pruning_distillation(&original, *sparsity)?;
        results.push(result.clone());

        let speedup = original.latency_ms / result.latency_ms;
        println!(
            "{:>9.0}% {:>10.1}M {:>11.2}% {:>10.1}MB {:>11.2}x",
            sparsity * 100.0,
            result.params_millions,
            result.accuracy * 100.0,
            result.size_mb,
            speedup
        );
    }
    println!("{:-<70}", "");

    // Find best tradeoff
    let best = find_best_tradeoff(&results, &original)?;
    ctx.record_float_metric("best_sparsity", best.sparsity);
    ctx.record_float_metric("best_efficiency", best.efficiency);

    println!();
    println!("Best Efficiency Tradeoff:");
    println!("  Sparsity: {:.0}%", best.sparsity * 100.0);
    println!("  Efficiency score: {:.3}", best.efficiency);
    println!(
        "  Accuracy retention: {:.1}%",
        best.accuracy_retention * 100.0
    );
    println!("  Size reduction: {:.1}x", best.size_reduction);

    // Gradual pruning schedule
    println!();
    println!("Recommended Gradual Pruning Schedule:");
    let schedule = generate_pruning_schedule(best.sparsity, 10)?;
    for (epoch, sparsity) in schedule.iter().enumerate() {
        let bar_len = (sparsity * 30.0) as usize;
        let bar = "█".repeat(bar_len);
        println!(
            "  Epoch {:>2}: {:>5.1}% {}",
            epoch + 1,
            sparsity * 100.0,
            bar
        );
    }

    // Save results
    let results_path = ctx.path("pruning_distill.json");
    save_results(&results_path, &results)?;
    println!();
    println!("Results saved to: {:?}", results_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelStats {
    params_millions: f64,
    accuracy: f64,
    size_mb: f64,
    latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TradeoffResult {
    sparsity: f64,
    efficiency: f64,
    accuracy_retention: f64,
    size_reduction: f64,
}

fn apply_pruning_distillation(original: &ModelStats, sparsity: f64) -> Result<ModelStats> {
    // Effective parameters after pruning
    let remaining_ratio = 1.0 - sparsity;
    let params = original.params_millions * remaining_ratio;

    // Accuracy loss from pruning (mitigated by distillation)
    // Without distillation, accuracy would drop more
    let accuracy_drop = sparsity * 0.08; // 8% max drop at 100% sparsity
    let distillation_recovery = sparsity * 0.04; // Distillation recovers half
    let accuracy = (original.accuracy - accuracy_drop + distillation_recovery).max(0.5);

    // Size reduction (sparse representation has overhead)
    let size = original.size_mb * remaining_ratio * 1.1; // 10% overhead for sparse format

    // Latency improvement (depends on sparsity and hardware)
    let speedup = 1.0 + sparsity * 1.5; // Up to 2.5x speedup at 100% sparsity
    let latency = original.latency_ms / speedup;

    Ok(ModelStats {
        params_millions: params,
        accuracy,
        size_mb: size,
        latency_ms: latency,
    })
}

fn find_best_tradeoff(results: &[ModelStats], original: &ModelStats) -> Result<TradeoffResult> {
    let mut best_idx = 0;
    let mut best_efficiency = 0.0f64;

    for (i, result) in results.iter().enumerate() {
        let accuracy_retention = result.accuracy / original.accuracy;
        let size_reduction = original.size_mb / result.size_mb;

        // Efficiency = accuracy_retention * size_reduction
        let efficiency = accuracy_retention * size_reduction.sqrt();

        if efficiency > best_efficiency {
            best_efficiency = efficiency;
            best_idx = i;
        }
    }

    let best = &results[best_idx];
    let sparsity = 1.0 - (best.params_millions / original.params_millions);

    Ok(TradeoffResult {
        sparsity,
        efficiency: best_efficiency,
        accuracy_retention: best.accuracy / original.accuracy,
        size_reduction: original.size_mb / best.size_mb,
    })
}

fn generate_pruning_schedule(target_sparsity: f64, epochs: usize) -> Result<Vec<f64>> {
    // Gradual cubic pruning schedule
    let schedule: Vec<f64> = (1..=epochs)
        .map(|e| {
            let progress = e as f64 / epochs as f64;
            target_sparsity * progress.powi(3) // Cubic schedule
        })
        .collect();

    Ok(schedule)
}

fn save_results(path: &std::path::Path, results: &[ModelStats]) -> Result<()> {
    let json = serde_json::to_string_pretty(results)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn original_model() -> ModelStats {
        ModelStats {
            params_millions: 100.0,
            accuracy: 0.90,
            size_mb: 400.0,
            latency_ms: 50.0,
        }
    }

    #[test]
    fn test_no_pruning() {
        let original = original_model();
        let result = apply_pruning_distillation(&original, 0.0).unwrap();

        assert_eq!(result.params_millions, original.params_millions);
        assert_eq!(result.accuracy, original.accuracy);
    }

    #[test]
    fn test_pruning_reduces_params() {
        let original = original_model();
        let result = apply_pruning_distillation(&original, 0.5).unwrap();

        assert!(result.params_millions < original.params_millions);
    }

    #[test]
    fn test_pruning_reduces_accuracy() {
        let original = original_model();
        let result = apply_pruning_distillation(&original, 0.9).unwrap();

        assert!(result.accuracy < original.accuracy);
    }

    #[test]
    fn test_pruning_improves_latency() {
        let original = original_model();
        let result = apply_pruning_distillation(&original, 0.7).unwrap();

        assert!(result.latency_ms < original.latency_ms);
    }

    #[test]
    fn test_find_best_tradeoff() {
        let original = original_model();
        let results: Vec<_> = vec![0.0, 0.5, 0.7, 0.9]
            .iter()
            .map(|s| apply_pruning_distillation(&original, *s).unwrap())
            .collect();

        let best = find_best_tradeoff(&results, &original).unwrap();

        assert!(best.efficiency > 0.0);
        assert!(best.sparsity >= 0.0 && best.sparsity <= 1.0);
    }

    #[test]
    fn test_pruning_schedule() {
        let schedule = generate_pruning_schedule(0.9, 10).unwrap();

        assert_eq!(schedule.len(), 10);
        assert!(schedule[0] < schedule[9]); // Increasing
        assert!(schedule[9] <= 0.9 + 0.001); // Reaches target
    }

    #[test]
    fn test_save_results() {
        let ctx = RecipeContext::new("test_pruning_save").unwrap();
        let path = ctx.path("results.json");

        let results = vec![original_model()];
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
        fn prop_params_decrease(sparsity in 0.0f64..0.99) {
            let original = ModelStats {
                params_millions: 100.0,
                accuracy: 0.90,
                size_mb: 400.0,
                latency_ms: 50.0,
            };

            let result = apply_pruning_distillation(&original, sparsity).unwrap();
            prop_assert!(result.params_millions <= original.params_millions);
        }

        #[test]
        fn prop_accuracy_bounded(sparsity in 0.0f64..0.99) {
            let original = ModelStats {
                params_millions: 100.0,
                accuracy: 0.90,
                size_mb: 400.0,
                latency_ms: 50.0,
            };

            let result = apply_pruning_distillation(&original, sparsity).unwrap();
            prop_assert!(result.accuracy >= 0.0);
            prop_assert!(result.accuracy <= 1.0);
        }

        #[test]
        fn prop_schedule_monotonic(target in 0.1f64..0.95, epochs in 3usize..20) {
            let schedule = generate_pruning_schedule(target, epochs).unwrap();

            for i in 1..schedule.len() {
                prop_assert!(schedule[i] >= schedule[i-1]);
            }
        }
    }
}
