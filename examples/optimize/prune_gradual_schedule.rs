//! # Recipe: Gradual Pruning Schedules
//!
//! **Category**: optimize
//! **CLI Equivalent**: `apr prune` with various schedules
//!
//! Demonstrates gradual pruning schedules: linear, cubic, and cosine.
//! Instead of pruning all at once, gradual pruning incrementally increases
//! sparsity during training, allowing the model to adapt and recover
//! accuracy at each step.
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Clippy clean
//! 6. [x] No `unwrap()` in logic

use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Pruning schedule type.
#[derive(Clone, Copy, Debug)]
enum PruningSchedule {
    /// Sparsity increases linearly over time.
    Linear,
    /// Sparsity follows a cubic curve (slow start, fast middle, slow end).
    Cubic,
    /// Sparsity follows a cosine curve (smooth acceleration and deceleration).
    Cosine,
}

impl PruningSchedule {
    fn name(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Cubic => "Cubic",
            Self::Cosine => "Cosine",
        }
    }
}

/// Compute current sparsity for a given schedule.
///
/// - `current_step`: current training step (0-indexed)
/// - `total_steps`: total number of pruning steps
/// - `target`: final target sparsity
///
/// Returns sparsity in [0.0, target].
fn schedule_sparsity(
    schedule: PruningSchedule,
    current_step: usize,
    total_steps: usize,
    target: f64,
) -> f64 {
    if total_steps == 0 {
        return target;
    }

    let t = (current_step as f64 / total_steps as f64).clamp(0.0, 1.0);

    let fraction = match schedule {
        PruningSchedule::Linear => t,
        PruningSchedule::Cubic => {
            // Cubic schedule: s(t) = target * (1 - (1-t)^3)
            // Slow start, rapid middle, slow end
            1.0 - (1.0 - t).powi(3)
        }
        PruningSchedule::Cosine => {
            // Cosine schedule: s(t) = target * (1 - cos(pi*t)) / 2
            // Smooth S-curve
            (1.0 - (std::f64::consts::PI * t).cos()) / 2.0
        }
    };

    (fraction * target).clamp(0.0, target)
}

/// Simulate accuracy during gradual pruning.
///
/// Models accuracy recovery between pruning steps.
fn simulate_accuracy(sparsity: f64, base_accuracy: f64, recovery_factor: f64) -> f64 {
    // Higher sparsity reduces accuracy, recovery factor represents fine-tuning benefit
    let penalty = sparsity.powf(1.5) * 0.3;
    let recovery = recovery_factor * sparsity * 0.15;
    (base_accuracy - penalty + recovery).clamp(0.0, 1.0)
}

/// Deterministic proxy for simulated training loss at a given sparsity step.
fn simulated_loss(seed: u64, step: usize, sparsity: f64) -> f64 {
    let mut h = DefaultHasher::new();
    (seed, step as u64).hash(&mut h);
    let bits = h.finish();
    let noise = ((bits & 0xFFFF) as f64 / f64::from(0xFFFFu16) - 0.5) * 0.02;
    // Loss increases with sparsity, with some noise
    0.1 + sparsity * 0.4 + noise
}

/// Render an ASCII chart comparing schedule curves.
fn render_schedule_chart(total_steps: usize, target: f64) {
    let schedules = [
        PruningSchedule::Linear,
        PruningSchedule::Cubic,
        PruningSchedule::Cosine,
    ];
    let symbols = ['L', 'C', 'S']; // Linear, Cubic, coSine
    let height = 20;
    let width = 60;

    // Build grid
    let mut grid = vec![vec![' '; width]; height];

    for (sched_idx, &schedule) in schedules.iter().enumerate() {
        #[allow(clippy::needless_range_loop)]
        for col in 0..width {
            let step = col * total_steps / width;
            let sparsity = schedule_sparsity(schedule, step, total_steps, target);
            let row = ((1.0 - sparsity / target) * (height - 1) as f64).round() as usize;
            let row = row.min(height - 1);
            grid[row][col] = symbols[sched_idx];
        }
    }

    println!(
        "  Sparsity over training steps (target = {:.0}%):",
        target * 100.0
    );
    println!("  L=Linear, C=Cubic, S=coSine\n");
    for (r, row) in grid.iter().enumerate() {
        let pct = target * (1.0 - r as f64 / (height - 1) as f64) * 100.0;
        let line: String = row.iter().collect();
        println!("  {pct:>5.1}% |{line}|");
    }
    let step_label_start = "0";
    let step_label_end = format!("{total_steps}");
    let padding = width - step_label_start.len() - step_label_end.len();
    println!(
        "         {step_label_start}{}{step_label_end}",
        " ".repeat(padding)
    );
    println!("         Step ->");
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("prune_gradual_schedule")?;

    println!("=== Gradual Pruning Schedules ===\n");

    let total_steps = 100;
    let target = 0.8;
    let base_accuracy = 0.95;

    // --- Section 1: Schedule Comparison Chart ---
    println!("--- Schedule Comparison ---");
    render_schedule_chart(total_steps, target);
    println!();

    // --- Section 2: Sparsity at Each Epoch ---
    println!("--- Sparsity at Key Epochs ---");
    let schedules = [
        PruningSchedule::Linear,
        PruningSchedule::Cubic,
        PruningSchedule::Cosine,
    ];
    let checkpoints = [0, 10, 25, 50, 75, 90, 100];

    println!(
        "  {:>5} | {:>10} | {:>10} | {:>10}",
        "Step", "Linear", "Cubic", "Cosine"
    );
    println!("  {}", "-".repeat(45));

    for &step in &checkpoints {
        let vals: Vec<f64> = schedules
            .iter()
            .map(|&s| schedule_sparsity(s, step, total_steps, target))
            .collect();
        println!(
            "  {:>5} | {:>9.1}% | {:>9.1}% | {:>9.1}%",
            step,
            vals[0] * 100.0,
            vals[1] * 100.0,
            vals[2] * 100.0
        );
    }
    println!();

    // --- Section 3: Pruning at Each Epoch (Simulation) ---
    println!("--- Pruning Simulation (10-epoch intervals) ---");
    for &schedule in &schedules {
        println!("\n  {} Schedule:", schedule.name());
        println!(
            "    {:>5} | {:>10} | {:>8} | {:>8}",
            "Epoch", "Sparsity", "Accuracy", "Loss"
        );
        println!("    {}", "-".repeat(42));

        for epoch in (0..=total_steps).step_by(10) {
            let sparsity = schedule_sparsity(schedule, epoch, total_steps, target);
            let accuracy = simulate_accuracy(sparsity, base_accuracy, 0.5);
            let loss = simulated_loss(42, epoch, sparsity);
            println!(
                "    {:>5} | {:>9.1}% | {:>7.2}% | {:>8.4}",
                epoch,
                sparsity * 100.0,
                accuracy * 100.0,
                loss
            );
        }
    }
    println!();

    // --- Section 4: Final Accuracy Comparison ---
    println!("--- Final Accuracy Comparison ---");
    for &schedule in &schedules {
        let final_sparsity = schedule_sparsity(schedule, total_steps, total_steps, target);
        let final_accuracy = simulate_accuracy(final_sparsity, base_accuracy, 0.5);
        let final_loss = simulated_loss(42, total_steps, final_sparsity);
        println!(
            "  {:<8}: sparsity = {:.1}%, accuracy = {:.2}%, loss = {:.4}",
            schedule.name(),
            final_sparsity * 100.0,
            final_accuracy * 100.0,
            final_loss
        );

        let metric = format!("final_accuracy_{}", schedule.name().to_lowercase());
        ctx.record_float_metric(&metric, final_accuracy);
    }
    println!();

    // --- Section 5: Recommended Schedule ---
    println!("--- Recommended Schedule ---");
    println!("  Cubic schedule is generally recommended:");
    println!("    - Slow start: model adapts before significant pruning");
    println!("    - Fast middle: efficient pruning during stable phase");
    println!("    - Slow end: fine-grained pruning for final quality");
    println!();
    println!("  When to use each:");
    println!("    Linear:  Simple, predictable, good baseline");
    println!("    Cubic:   Best accuracy, recommended for production");
    println!("    Cosine:  Smooth transitions, good for sensitive models");
    println!();

    // --- Section 6: Schedule Derivatives (Rate of Pruning) ---
    println!("--- Pruning Rate (Sparsity Change per Step) ---");
    println!(
        "  {:>5} | {:>12} | {:>12} | {:>12}",
        "Step", "Linear d/dt", "Cubic d/dt", "Cosine d/dt"
    );
    println!("  {}", "-".repeat(50));
    let delta = 1;
    for step in (0..=total_steps).step_by(20) {
        let rates: Vec<f64> = schedules
            .iter()
            .map(|&s| {
                let s0 = schedule_sparsity(s, step, total_steps, target);
                let s1 = schedule_sparsity(s, (step + delta).min(total_steps), total_steps, target);
                (s1 - s0) / delta as f64
            })
            .collect();
        println!(
            "  {:>5} | {:>12.6} | {:>12.6} | {:>12.6}",
            step, rates[0], rates[1], rates[2]
        );
    }
    println!();

    // --- Section 7: Save Schedule Config ---
    println!("--- Save Schedule Config (APR v2) ---");
    let config_json = format!(
        r#"{{"schedule":"cubic","total_steps":{},"target_sparsity":{},"base_accuracy":{}}}"#,
        total_steps, target, base_accuracy
    );
    let config_bytes = config_json.into_bytes();

    let bundle = ModelBundleV2::new()
        .with_name("gradual_prune_schedule")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("schedule_config", vec![1, config_bytes.len()], config_bytes)
        .build();

    assert_eq!(&bundle[0..4], b"APR2");
    println!("  Bundle size: {} bytes", bundle.len());
    println!(
        "  Recommended: Cubic schedule to {:.0}% sparsity",
        target * 100.0
    );

    ctx.record_metric("total_steps", total_steps as i64);
    ctx.record_float_metric("target_sparsity", target);
    ctx.record_metric("bundle_size_bytes", bundle.len() as i64);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_reaches_target() {
        let target = 0.8;
        let s = schedule_sparsity(PruningSchedule::Linear, 100, 100, target);
        assert!(
            (s - target).abs() < 1e-10,
            "Linear should reach target: got {s}"
        );
    }

    #[test]
    fn test_cubic_reaches_target() {
        let target = 0.8;
        let s = schedule_sparsity(PruningSchedule::Cubic, 100, 100, target);
        assert!(
            (s - target).abs() < 1e-10,
            "Cubic should reach target: got {s}"
        );
    }

    #[test]
    fn test_cosine_reaches_target() {
        let target = 0.8;
        let s = schedule_sparsity(PruningSchedule::Cosine, 100, 100, target);
        assert!(
            (s - target).abs() < 1e-10,
            "Cosine should reach target: got {s}"
        );
    }

    #[test]
    fn test_all_schedules_start_at_zero() {
        let schedules = [
            PruningSchedule::Linear,
            PruningSchedule::Cubic,
            PruningSchedule::Cosine,
        ];
        for schedule in schedules {
            let s = schedule_sparsity(schedule, 0, 100, 0.8);
            assert!(s.abs() < 1e-10, "{:?} should start at 0: got {s}", schedule);
        }
    }

    #[test]
    fn test_linear_monotonic_increase() {
        let steps: Vec<f64> = (0..=100)
            .map(|s| schedule_sparsity(PruningSchedule::Linear, s, 100, 0.8))
            .collect();
        for window in steps.windows(2) {
            assert!(
                window[1] >= window[0] - f64::EPSILON,
                "Linear should be monotonic: {} -> {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_cubic_monotonic_increase() {
        let steps: Vec<f64> = (0..=100)
            .map(|s| schedule_sparsity(PruningSchedule::Cubic, s, 100, 0.8))
            .collect();
        for window in steps.windows(2) {
            assert!(
                window[1] >= window[0] - f64::EPSILON,
                "Cubic should be monotonic: {} -> {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_cosine_monotonic_increase() {
        let steps: Vec<f64> = (0..=100)
            .map(|s| schedule_sparsity(PruningSchedule::Cosine, s, 100, 0.8))
            .collect();
        for window in steps.windows(2) {
            assert!(
                window[1] >= window[0] - f64::EPSILON,
                "Cosine should be monotonic: {} -> {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_cosine_slower_than_linear_at_start() {
        // Cosine should be below linear early on (slow start)
        let step = 10;
        let linear = schedule_sparsity(PruningSchedule::Linear, step, 100, 0.8);
        let cosine = schedule_sparsity(PruningSchedule::Cosine, step, 100, 0.8);
        assert!(
            cosine < linear + 0.01,
            "Cosine ({cosine}) should be <= linear ({linear}) at early steps"
        );
    }

    #[test]
    fn test_cubic_faster_than_linear_at_middle() {
        // Cubic should exceed linear in the middle-to-late phase
        let step = 60;
        let linear = schedule_sparsity(PruningSchedule::Linear, step, 100, 0.8);
        let cubic = schedule_sparsity(PruningSchedule::Cubic, step, 100, 0.8);
        assert!(
            cubic >= linear - 0.01,
            "Cubic ({cubic}) should be >= linear ({linear}) at step 60"
        );
    }

    #[test]
    fn test_sparsity_clamped_to_target() {
        let schedules = [
            PruningSchedule::Linear,
            PruningSchedule::Cubic,
            PruningSchedule::Cosine,
        ];
        let target = 0.5;
        for schedule in schedules {
            for step in 0..=150 {
                let s = schedule_sparsity(schedule, step, 100, target);
                assert!(
                    s <= target + 1e-10,
                    "{:?} exceeded target at step {step}: {s} > {target}",
                    schedule
                );
                assert!(
                    s >= -1e-10,
                    "{:?} went negative at step {step}: {s}",
                    schedule
                );
            }
        }
    }

    #[test]
    fn test_zero_total_steps() {
        let s = schedule_sparsity(PruningSchedule::Linear, 0, 0, 0.8);
        assert!((s - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_simulate_accuracy_bounds() {
        for sparsity_pct in 0..=100 {
            let sparsity = sparsity_pct as f64 / 100.0;
            let acc = simulate_accuracy(sparsity, 0.95, 0.5);
            assert!(
                (0.0..=1.0).contains(&acc),
                "Accuracy out of bounds: {acc} at sparsity {sparsity}"
            );
        }
    }

    #[test]
    fn test_schedule_names() {
        assert_eq!(PruningSchedule::Linear.name(), "Linear");
        assert_eq!(PruningSchedule::Cubic.name(), "Cubic");
        assert_eq!(PruningSchedule::Cosine.name(), "Cosine");
    }
}
