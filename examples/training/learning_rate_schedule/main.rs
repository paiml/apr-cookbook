#![allow(unused_imports)]
//! Learning Rate Schedule Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Demonstrates common learning rate scheduling strategies for ML training:
//! constant, step decay, exponential decay, cosine annealing, warmup+cosine,
//! one-cycle, and polynomial decay. Compares convergence behavior across
//! schedules on a simple training task.
//!
//! # Schedule Strategies
//!
//! ```text
//! LR
//! |                           ╭── one-cycle peak
//! |  constant ────────────    │
//! |  step     ──┐             ╭╮
//! |             └──┐         ╱  ╲
//! |                └──      ╱    ╲
//! |  cosine   ╲            ╱      ╲
//! |            ╲__╱       ╱        ╲___
//! |  exp       ╲         warmup+cosine
//! |             ╲____
//! └──────────────────────── epoch →
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example learning_rate_schedule
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr finetune model.apr          # APR native format
//! apr finetune model.gguf         # GGUF (llama.cpp compatible)
//! apr finetune model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Learning Rate Schedule Example ===\n");

    let seed = 42;
    let config = ScheduleConfig::default_config();
    let train_data = generate_data(N_TRAIN, seed);
    let val_data = generate_data(N_VAL, seed + 1000);

    // =========================================================================
    // Section 1: Schedule Definitions
    // =========================================================================
    println!("1. Schedule Definitions");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>15} {:>10} {:>10} {:>10}",
        "Schedule", "LR(0)", "LR(50)", "LR(99)"
    );
    println!("   {}", "─".repeat(48));

    for &sched in &ALL_SCHEDULES {
        let lr_0 = get_lr(sched, 0, &config);
        let lr_50 = get_lr(sched, 50, &config);
        let lr_99 = get_lr(sched, 99, &config);
        println!(
            "   {:>15} {:>10.6} {:>10.6} {:>10.6}",
            sched.name(),
            lr_0,
            lr_50,
            lr_99
        );
    }
    println!();

    // =========================================================================
    // Section 2: LR Curve Visualization
    // =========================================================================
    println!("2. LR Curve Visualization (sparkline)");
    println!("   ─────────────────────────────────────────");

    for &sched in &ALL_SCHEDULES {
        let lr_values: Vec<f64> = (0..config.total_epochs)
            .map(|e| get_lr(sched, e, &config))
            .collect();
        let spark = sparkline(&lr_values, SPARKLINE_WIDTH);
        println!("   {:>8} |{}|", sched.short_name(), spark);
    }
    println!("   {:>8}  epoch 0 ────────────────────────── 99", "");
    println!();

    // =========================================================================
    // Section 3: Warmup Phase Comparison
    // =========================================================================
    println!("3. Warmup Phase Comparison");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Epoch", "Const", "Cosine", "W+Cos", "1Cycle", "Poly"
    );
    println!("   {}", "─".repeat(52));

    let warmup_schedules = [
        Schedule::Constant,
        Schedule::CosineAnnealing,
        Schedule::WarmupCosine,
        Schedule::OneCycle,
        Schedule::Polynomial,
    ];
    let warmup_sample_epochs = [0, 2, 5, 8, 10, 15, 20];

    for &epoch in &warmup_sample_epochs {
        let lrs: Vec<f64> = warmup_schedules
            .iter()
            .map(|&s| get_lr(s, epoch, &config))
            .collect();
        println!(
            "   {:>8} {:>8.5} {:>8.5} {:>8.5} {:>8.5} {:>8.5}",
            epoch, lrs[0], lrs[1], lrs[2], lrs[3], lrs[4]
        );
    }
    println!();

    // =========================================================================
    // Section 4: Training Convergence Simulation
    // =========================================================================
    println!("4. Training Convergence Simulation");
    println!("   ─────────────────────────────────────────");

    let mut results: Vec<TrainResult> = Vec::new();
    for &sched in &ALL_SCHEDULES {
        let result = train_with_schedule(sched, &config, &train_data, &val_data, seed);
        results.push(result);
    }

    println!(
        "   {:>15} {:>10} {:>10} {:>10} {:>8}",
        "Schedule", "FinalTrn", "FinalVal", "BestVal", "BestEp"
    );
    println!("   {}", "─".repeat(56));

    for result in &results {
        println!(
            "   {:>15} {:>10.6} {:>10.6} {:>10.6} {:>8}",
            result.schedule.name(),
            result.final_train_loss,
            result.final_val_loss,
            result.best_val_loss,
            result.best_epoch
        );
    }
    println!();

    // Loss curve sparklines
    println!("   Loss curves (val loss over epochs):");
    for result in &results {
        let spark = sparkline(&result.loss_curve, SPARKLINE_WIDTH);
        println!("   {:>8} |{}|", result.schedule.short_name(), spark);
    }
    println!();

    // =========================================================================
    // Section 5: Schedule Parameter Sensitivity
    // =========================================================================
    println!("5. Schedule Parameter Sensitivity");
    println!("   ─────────────────────────────────────────");

    // Vary base learning rate
    println!("   Base LR sensitivity (WarmupCosine):");
    println!(
        "   {:>10} {:>10} {:>10} {:>8}",
        "BaseLR", "FinalVal", "BestVal", "BestEp"
    );
    println!("   {}", "─".repeat(42));

    for &lr in &[0.001, 0.005, 0.01, 0.02, 0.05, 0.1] {
        let mut cfg = ScheduleConfig::default_config();
        cfg.base_lr = lr;
        let result =
            train_with_schedule(Schedule::WarmupCosine, &cfg, &train_data, &val_data, seed);
        println!(
            "   {:>10.4} {:>10.6} {:>10.6} {:>8}",
            lr, result.final_val_loss, result.best_val_loss, result.best_epoch
        );
    }
    println!();

    // Vary warmup length
    println!("   Warmup length sensitivity (WarmupCosine, base_lr=0.01):");
    println!(
        "   {:>10} {:>10} {:>10} {:>8}",
        "Warmup", "FinalVal", "BestVal", "BestEp"
    );
    println!("   {}", "─".repeat(42));

    for &warmup in &[0, 5, 10, 20, 30, 50] {
        let mut cfg = ScheduleConfig::default_config();
        cfg.warmup_epochs = warmup;
        let result =
            train_with_schedule(Schedule::WarmupCosine, &cfg, &train_data, &val_data, seed);
        println!(
            "   {:>10} {:>10.6} {:>10.6} {:>8}",
            warmup, result.final_val_loss, result.best_val_loss, result.best_epoch
        );
    }
    println!();

    // =========================================================================
    // Section 6: Best Schedule Selection
    // =========================================================================
    println!("6. Best Schedule Selection");
    println!("   ─────────────────────────────────────────");

    // Rank by best validation loss
    let mut ranked: Vec<&TrainResult> = results.iter().collect();
    ranked.sort_by(|a, b| a.best_val_loss.partial_cmp(&b.best_val_loss).unwrap());

    println!(
        "   {:>4} {:>15} {:>10} {:>10} {:>8}",
        "Rank", "Schedule", "BestVal", "FinalVal", "BestEp"
    );
    println!("   {}", "─".repeat(50));

    for (i, result) in ranked.iter().enumerate() {
        let marker = if i == 0 { " <-- best" } else { "" };
        println!(
            "   {:>4} {:>15} {:>10.6} {:>10.6} {:>8}{}",
            i + 1,
            result.schedule.name(),
            result.best_val_loss,
            result.final_val_loss,
            result.best_epoch,
            marker
        );
    }
    println!();

    // Generalization gap analysis
    println!("   Generalization gap (train - val loss):");
    println!(
        "   {:>15} {:>10} {:>10} {:>10}",
        "Schedule", "TrainLoss", "ValLoss", "Gap"
    );
    println!("   {}", "─".repeat(48));

    for result in &results {
        let gap = result.final_val_loss - result.final_train_loss;
        println!(
            "   {:>15} {:>10.6} {:>10.6} {:>10.6}",
            result.schedule.name(),
            result.final_train_loss,
            result.final_val_loss,
            gap
        );
    }
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_schedule_returns_base_lr() {
        let config = ScheduleConfig::default_config();
        for epoch in [0, 25, 50, 99] {
            let lr = get_lr(Schedule::Constant, epoch, &config);
            assert!(
                (lr - config.base_lr).abs() < f64::EPSILON,
                "Constant LR should be {}, got {lr}",
                config.base_lr
            );
        }
    }

    #[test]
    fn test_step_decay_decreases_at_boundary() {
        let config = ScheduleConfig::default_config();
        let lr_before = get_lr(Schedule::StepDecay, config.step_size - 1, &config);
        let lr_after = get_lr(Schedule::StepDecay, config.step_size, &config);
        assert!(
            lr_after < lr_before,
            "Step decay should decrease at boundary: {lr_before} -> {lr_after}"
        );
    }

    #[test]
    fn test_exponential_decay_monotonic() {
        let config = ScheduleConfig::default_config();
        let mut prev = get_lr(Schedule::Exponential, 0, &config);
        for epoch in 1..config.total_epochs {
            let lr = get_lr(Schedule::Exponential, epoch, &config);
            assert!(
                lr <= prev + f64::EPSILON,
                "Exponential should be monotonically decreasing: epoch {epoch}"
            );
            prev = lr;
        }
    }

    #[test]
    fn test_cosine_annealing_endpoints() {
        let config = ScheduleConfig::default_config();
        let lr_start = get_lr(Schedule::CosineAnnealing, 0, &config);
        let lr_end = get_lr(Schedule::CosineAnnealing, config.total_epochs - 1, &config);
        // Should start near base_lr and end near min_lr
        assert!(
            (lr_start - config.base_lr).abs() < 0.001,
            "Cosine should start near base_lr: got {lr_start}"
        );
        assert!(
            lr_end < config.base_lr * 0.1,
            "Cosine should end low: got {lr_end}"
        );
    }

    #[test]
    fn test_warmup_cosine_starts_low() {
        let config = ScheduleConfig::default_config();
        let lr_0 = get_lr(Schedule::WarmupCosine, 0, &config);
        let lr_warmup_end = get_lr(Schedule::WarmupCosine, config.warmup_epochs, &config);
        assert!(
            lr_0 < lr_warmup_end,
            "Warmup should start low and ramp up: {lr_0} -> {lr_warmup_end}"
        );
        assert!(
            (lr_0 - config.min_lr).abs() < 0.001,
            "Warmup should start at min_lr: got {lr_0}"
        );
    }

    #[test]
    fn test_one_cycle_peaks_then_decays() {
        let config = ScheduleConfig::default_config();
        let peak_epoch = (config.total_epochs as f64 * config.one_cycle_pct_start) as usize;
        let lr_start = get_lr(Schedule::OneCycle, 0, &config);
        let lr_peak = get_lr(Schedule::OneCycle, peak_epoch, &config);
        let lr_end = get_lr(Schedule::OneCycle, config.total_epochs - 1, &config);
        assert!(
            lr_peak > lr_start,
            "OneCycle peak should exceed start: {lr_start} -> {lr_peak}"
        );
        assert!(
            lr_end < lr_peak,
            "OneCycle should decay after peak: {lr_peak} -> {lr_end}"
        );
    }

    #[test]
    fn test_polynomial_decay_endpoints() {
        let config = ScheduleConfig::default_config();
        let lr_start = get_lr(Schedule::Polynomial, 0, &config);
        let lr_end = get_lr(Schedule::Polynomial, config.total_epochs - 1, &config);
        assert!(
            (lr_start - config.base_lr).abs() < 0.001,
            "Poly should start near base_lr: {lr_start}"
        );
        assert!(
            lr_end < config.base_lr * 0.1,
            "Poly should end near min_lr: {lr_end}"
        );
    }

    #[test]
    fn test_sparkline_output_width() {
        let values: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let spark = sparkline(&values, 20);
        assert_eq!(
            spark.chars().count(),
            20,
            "Sparkline should have exactly 20 chars"
        );
    }

    #[test]
    fn test_sparkline_monotonic_input() {
        let values: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let spark = sparkline(&values, 10);
        let chars: Vec<char> = spark.chars().collect();
        // First char should be lowest block, last should be highest
        assert!(chars[0] <= chars[chars.len() - 1]);
    }

    #[test]
    fn test_model_forward_dimensions() {
        let model = LinearModel::new(42);
        let input = vec![0.5; INPUT_DIM];
        let output = model.forward(&input);
        assert_eq!(output.len(), OUTPUT_DIM);
    }

    #[test]
    fn test_mse_loss_nonnegative() {
        let model = LinearModel::new(42);
        let (inputs, targets) = generate_data(10, 42);
        let loss = model.mse_loss(&inputs, &targets);
        assert!(loss >= 0.0, "MSE loss must be >= 0, got {loss}");
        assert!(loss.is_finite());
    }

    #[test]
    fn test_training_reduces_loss() {
        let (inputs, targets) = generate_data(20, 42);
        let mut model = LinearModel::new(42);
        let loss_before = model.mse_loss(&inputs, &targets);
        for _ in 0..10 {
            model.train_step(&inputs, &targets, 0.01);
        }
        let loss_after = model.mse_loss(&inputs, &targets);
        assert!(
            loss_after < loss_before,
            "Training should reduce loss: {loss_before} -> {loss_after}"
        );
    }

    #[test]
    fn test_generate_data_deterministic() {
        let d1 = generate_data(10, 42);
        let d2 = generate_data(10, 42);
        assert_eq!(d1.0, d2.0);
        assert_eq!(d1.1, d2.1);
    }

    #[test]
    fn test_generate_data_shapes() {
        let (inputs, targets) = generate_data(15, 42);
        assert_eq!(inputs.len(), 15);
        assert_eq!(targets.len(), 15);
        assert_eq!(inputs[0].len(), INPUT_DIM);
        assert_eq!(targets[0].len(), OUTPUT_DIM);
    }

    #[test]
    fn test_schedule_names_unique() {
        let names: Vec<&str> = ALL_SCHEDULES.iter().map(|s| s.name()).collect();
        for (i, name) in names.iter().enumerate() {
            for (j, other) in names.iter().enumerate() {
                if i != j {
                    assert_ne!(name, other, "Schedule names must be unique");
                }
            }
        }
    }

    #[test]
    fn test_all_schedules_positive_lr() {
        let config = ScheduleConfig::default_config();
        for &sched in &ALL_SCHEDULES {
            for epoch in 0..config.total_epochs {
                let lr = get_lr(sched, epoch, &config);
                assert!(
                    lr > 0.0,
                    "{:?} produced non-positive LR {lr} at epoch {epoch}",
                    sched
                );
            }
        }
    }

    #[test]
    fn test_train_with_schedule_returns_loss_curve() {
        let config = ScheduleConfig::default_config();
        let train = generate_data(20, 42);
        let val = generate_data(5, 99);
        let result = train_with_schedule(Schedule::WarmupCosine, &config, &train, &val, 42);
        assert_eq!(result.loss_curve.len(), config.total_epochs);
        assert!(result.best_epoch < config.total_epochs);
        assert!(result.best_val_loss <= result.final_val_loss + f64::EPSILON);
    }
}
