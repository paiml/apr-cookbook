//! # Recipe: Gradient-Norm — Monotonic Divergence Run
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr grad-norm --history-file h.json --divergence-window 8`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates monotonic-divergence detection. Unlike a single-step spike,
//! divergence is a *trend*: gradient norm grows step over step for ≥ N
//! consecutive steps. This is the failure mode for unstable mixed-precision
//! training (FP16 gradient overflow that survives clipping). The detector
//! reports each maximal monotonic-increasing run of length ≥ N together
//! with its start/end step and the geometric growth rate.
//!
//! ## Run Command
//! ```bash
//! cargo run --example grad_norm_divergence_run
//! ```
//!
//! ## References
//! - aprender CRUX-F-09 (divergence rule).
//! - Micikevicius et al. (2018). *Mixed Precision Training*. arXiv:1710.03740 (FP16 overflow).
//!
//! Added by PMAT-092 (expand-cookbooks followup — embeddings/search/grad-norm lint).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct DivergenceRun {
    pub start_step: u64,
    pub end_step: u64,
    pub length: usize,
    pub geometric_growth: f64, // (end / start) ^ (1/(length-1))
}

pub fn detect_divergence(history: &Value, min_length: usize) -> Vec<DivergenceRun> {
    let mut out = Vec::new();
    let Some(arr) = history.get("steps").and_then(Value::as_array) else {
        return out;
    };
    let series: Vec<(u64, f64)> = arr
        .iter()
        .filter_map(|r| {
            let s = r.get("step").and_then(Value::as_u64)?;
            let v = r.get("pre_clip").and_then(Value::as_f64)?;
            Some((s, v))
        })
        .collect();

    let mut i = 0;
    while i < series.len() {
        // Find longest monotonic-increasing run starting at i.
        let mut j = i + 1;
        while j < series.len() && series[j].1 > series[j - 1].1 {
            j += 1;
        }
        let length = j - i;
        if length >= min_length {
            let (start_step, start_val) = series[i];
            let (end_step, end_val) = series[j - 1];
            let geom = if length > 1 && start_val > 0.0 {
                (end_val / start_val).powf(1.0 / (length - 1) as f64)
            } else {
                1.0
            };
            out.push(DivergenceRun {
                start_step,
                end_step,
                length,
                geometric_growth: geom,
            });
        }
        i = j.max(i + 1);
    }
    out
}

fn build_diverging_history() -> Value {
    let mut steps = Vec::new();
    // 4 calm steps
    for s in 0..4u64 {
        steps.push(json!({ "step": s, "pre_clip": 0.5 }));
    }
    // 12 monotonically increasing steps (1.5× growth each)
    let mut v = 0.5_f64;
    for s in 4..16u64 {
        v *= 1.5;
        steps.push(json!({ "step": s, "pre_clip": v }));
    }
    json!({ "steps": steps })
}

fn build_stable_history() -> Value {
    let mut steps = Vec::new();
    for s in 0..16u64 {
        steps.push(json!({ "step": s, "pre_clip": 0.5 + 0.05 * (s as f64).sin() }));
    }
    json!({ "steps": steps })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("grad_norm_divergence_run")?;
    println!("=== Recipe: {} ===", ctx.name());

    let stable = detect_divergence(&build_stable_history(), 8);
    println!("stable:    {} divergence runs", stable.len());

    let diverging = detect_divergence(&build_diverging_history(), 8);
    println!("diverging: {} divergence runs", diverging.len());
    for r in &diverging {
        println!(
            "  steps {}..{}  length={}  geometric_growth={:.3}×/step",
            r.start_step, r.end_step, r.length, r.geometric_growth
        );
    }

    ctx.record_metric("divergence_runs", diverging.len() as i64);
    ctx.record_string_metric(
        "verdict",
        if diverging.is_empty() { "PASS" } else { "FAIL" },
    );
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divergence_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn stable_history_has_no_divergence() {
        let r = detect_divergence(&build_stable_history(), 8);
        assert!(r.is_empty(), "expected stable: {r:?}");
    }

    #[test]
    fn diverging_history_yields_one_run() {
        let r = detect_divergence(&build_diverging_history(), 8);
        assert_eq!(r.len(), 1);
        assert!(r[0].length >= 8);
        // Each step grows 1.5× → geometric_growth ≈ 1.5
        assert!((r[0].geometric_growth - 1.5).abs() < 0.01);
    }

    #[test]
    fn run_shorter_than_min_length_skipped() {
        // 5 monotonic steps, threshold 8 → no finding.
        let mut steps = Vec::new();
        let mut v = 0.5_f64;
        for s in 0..5u64 {
            v *= 1.5;
            steps.push(json!({ "step": s, "pre_clip": v }));
        }
        let h = json!({ "steps": steps });
        assert!(detect_divergence(&h, 8).is_empty());
    }

    #[test]
    fn equal_consecutive_breaks_run() {
        // Strict monotonic increase only — equality breaks the run.
        let h = json!({
            "steps": [
                { "step": 0, "pre_clip": 0.5 },
                { "step": 1, "pre_clip": 0.6 },
                { "step": 2, "pre_clip": 0.6 }, // break
                { "step": 3, "pre_clip": 0.7 }
            ]
        });
        // No run of length ≥ 4
        assert!(detect_divergence(&h, 4).is_empty());
    }

    #[test]
    fn multiple_independent_runs_reported_separately() {
        // Two divergence episodes separated by a recovery.
        let mut steps = Vec::new();
        for (s, v) in (0u64..4).zip([0.1, 0.5, 1.0, 2.0]) {
            steps.push(json!({ "step": s, "pre_clip": v }));
        }
        steps.push(json!({ "step": 4u64, "pre_clip": 0.3 })); // recovery
        for (idx, v) in (5u64..9).zip([0.4, 1.0, 2.5, 5.0]) {
            steps.push(json!({ "step": idx, "pre_clip": v }));
        }
        let h = json!({ "steps": steps });
        let r = detect_divergence(&h, 4);
        assert_eq!(r.len(), 2);
    }
}
