//! # Recipe: Pretrain — NaN Guard Edge Case
//!
//! **Category**: training
//! **CLI Equivalent**: `apr pretrain --halt-on-nan`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example pretrain_nan_guard` exits 0
//! 2. [x] `cargo test --example pretrain_nan_guard` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Demonstrates the NaN-guard edge case. Mimics a loss curve that diverges
//! (bad learning rate + unstable bias update) and shows how the guard catches
//! `NaN`/`Inf` WITHIN a step by snapshotting the previous-good checkpoint.
//!
//! ## Run Command
//! ```bash
//! cargo run --example pretrain_nan_guard
//! ```
//!
//! ## References
//! - Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. DOI:10.5555/3086952

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct NanEvent {
    pub step: u32,
    pub value: f64,
    pub rolled_back_to: u32,
}

#[derive(Debug, Clone)]
pub struct NanReport {
    pub steps_run: u32,
    pub nan_events: Vec<NanEvent>,
    pub final_loss: f64,
    pub halted: bool,
}

/// Divergent loss: quadratic blow-up after a warm-up period.
pub fn diverging_loss(step: u32) -> f64 {
    if step < 10 {
        return 1.0 - f64::from(step) * 0.08;
    }
    // After step 10 grows quadratically; past step 25 we inject an Inf/NaN.
    let x = f64::from(step - 10);
    let v = 0.2 + x * x * 0.05;
    if step >= 25 {
        v * f64::INFINITY
    } else {
        v
    }
}

pub fn run_with_nan_guard(max_steps: u32) -> NanReport {
    let mut events = Vec::new();
    let mut last_good_step = 0u32;
    let mut last_good_loss = f64::NAN;
    let mut final_loss = 0.0;
    let mut halted = false;
    let mut step = 0u32;
    while step < max_steps {
        step += 1;
        let loss = diverging_loss(step);
        if loss.is_nan() || loss.is_infinite() {
            events.push(NanEvent {
                step,
                value: loss,
                rolled_back_to: last_good_step,
            });
            halted = true;
            break;
        }
        last_good_step = step;
        last_good_loss = loss;
        final_loss = loss;
    }
    if halted {
        final_loss = last_good_loss;
    }
    NanReport {
        steps_run: step,
        nan_events: events,
        final_loss,
        halted,
    }
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("pretrain_nan_guard")?;
    let report = run_with_nan_guard(200);

    println!("=== Recipe: {} ===", ctx.name());
    println!("steps_run  : {}", report.steps_run);
    println!("halted     : {}", report.halted);
    println!("final_loss : {:.4}", report.final_loss);
    if let Some(ev) = report.nan_events.first() {
        println!(
            "NaN event  : step {} value={} rolled back to step {}",
            ev.step, ev.value, ev.rolled_back_to
        );
    }

    // Persist rollback info to tempdir.
    let rp = ctx.path("nan_events.json");
    let json_events: Vec<_> = report
        .nan_events
        .iter()
        .map(|e| {
            serde_json::json!({
                "step": e.step,
                "value": if e.value.is_finite() { serde_json::Value::from(e.value) } else { serde_json::Value::String(format!("{}", e.value)) },
                "rolled_back_to": e.rolled_back_to
            })
        })
        .collect();
    std::fs::write(
        &rp,
        serde_json::to_vec_pretty(&json_events)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    ctx.record_metric("steps_run", i64::from(report.steps_run));
    ctx.record_float_metric("final_loss", report.final_loss);
    ctx.record_metric("nan_events", report.nan_events.len() as i64);
    ctx.record_string_metric("verdict", if report.halted { "HALTED" } else { "OK" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_detects_nonfinite() {
        let r = run_with_nan_guard(100);
        assert!(r.halted);
        assert_eq!(r.nan_events.len(), 1);
        assert!(r.nan_events[0].rolled_back_to < r.steps_run);
    }

    #[test]
    fn final_loss_is_last_good() {
        let r = run_with_nan_guard(100);
        assert!(r.final_loss.is_finite());
    }

    #[test]
    fn early_budget_exits_before_nan() {
        let r = run_with_nan_guard(5);
        assert!(!r.halted);
        assert_eq!(r.nan_events.len(), 0);
    }
}
