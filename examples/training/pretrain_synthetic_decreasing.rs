//! # Recipe: Pretrain — Synthetic Decreasing Loss
//!
//! **Category**: training
//! **CLI Equivalent**: `apr pretrain --steps 200 --loss-floor 0.05`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example pretrain_synthetic_decreasing` exits 0
//! 2. [x] `cargo test --example pretrain_synthetic_decreasing` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Runs the canonical `apr pretrain` loop shape in miniature: a synthetic
//! loss curve that decreases exponentially with additive noise, stopping when
//! either the step budget or the loss-floor is reached. Prints the step /
//! loss table and the final ckpt digest.
//!
//! ## Run Command
//! ```bash
//! cargo run --example pretrain_synthetic_decreasing
//! ```
//!
//! ## References
//! - Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models (Chinchilla)*. arXiv:2203.15556

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use rand::Rng;

/// One step of the synthetic loss: `loss = 2.0 * exp(-k * step) + noise`.
pub fn synthetic_loss(step: u32, rng: &mut impl Rng) -> f64 {
    let base = 2.0_f64 * (-0.02 * f64::from(step)).exp();
    let noise: f64 = rng.gen_range(-0.02..0.02);
    (base + noise).max(0.0)
}

#[derive(Debug, Clone, PartialEq)]
pub struct PretrainReport {
    pub steps_run: u32,
    pub final_loss: f64,
    pub stopped_reason: &'static str,
}

pub fn run_pretrain(
    max_steps: u32,
    loss_floor: f64,
    rng: &mut impl Rng,
) -> (PretrainReport, Vec<f64>) {
    let mut losses = Vec::new();
    let mut step = 0u32;
    let mut loss = f64::INFINITY;
    while step < max_steps {
        step += 1;
        loss = synthetic_loss(step, rng);
        losses.push(loss);
        if loss <= loss_floor {
            return (
                PretrainReport {
                    steps_run: step,
                    final_loss: loss,
                    stopped_reason: "loss_floor",
                },
                losses,
            );
        }
    }
    (
        PretrainReport {
            steps_run: step,
            final_loss: loss,
            stopped_reason: "max_steps",
        },
        losses,
    )
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("pretrain_synthetic_decreasing")?;
    let max_steps = 200u32;
    let loss_floor = 0.05_f64;

    let (report, losses) = run_pretrain(max_steps, loss_floor, ctx.rng());

    println!("=== Recipe: {} ===", ctx.name());
    println!("steps_run     : {}", report.steps_run);
    println!("final_loss    : {:.4}", report.final_loss);
    println!("stopped_reason: {}", report.stopped_reason);

    // Sparse progress print: every 20 steps.
    for (i, l) in losses.iter().enumerate() {
        if i % 20 == 0 || i + 1 == losses.len() {
            println!("  step {:>4}: loss={:.4}", i + 1, l);
        }
    }

    // Persist loss curve for downstream analysis.
    let cp = ctx.path("loss_curve.csv");
    let mut csv = String::from("step,loss\n");
    for (i, l) in losses.iter().enumerate() {
        csv.push_str(&format!("{},{:.6}\n", i + 1, l));
    }
    std::fs::write(&cp, csv)?;

    ctx.record_metric("steps_run", i64::from(report.steps_run));
    ctx.record_float_metric("final_loss", report.final_loss);
    ctx.record_string_metric("stopped_reason", report.stopped_reason);
    ctx.record_string_metric(
        "verdict",
        if report.final_loss <= loss_floor {
            "CONVERGED"
        } else {
            "BUDGET_EXHAUSTED"
        },
    );
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn loss_decreases_on_average() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let (_, losses) = run_pretrain(200, 0.0, &mut rng);
        let early: f64 = losses[..20].iter().sum::<f64>() / 20.0;
        let late: f64 = losses[losses.len() - 20..].iter().sum::<f64>() / 20.0;
        assert!(late < early, "late={late} early={early}");
    }

    #[test]
    fn stops_on_loss_floor() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let (r, _) = run_pretrain(1_000, 0.5, &mut rng);
        assert_eq!(r.stopped_reason, "loss_floor");
    }

    #[test]
    fn stops_on_max_steps() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let (r, _) = run_pretrain(5, 0.00001, &mut rng);
        assert_eq!(r.stopped_reason, "max_steps");
    }
}
