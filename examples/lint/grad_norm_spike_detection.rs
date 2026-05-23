//! # Recipe: Gradient-Norm — Rolling-Median Spike Detection
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr grad-norm --history-file h.json --spike-window 16 --spike-multiplier 10`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the rolling-median spike detector. A loss-explosion or
//! divergent-batch typically presents as a single step where
//! `pre_clip ≫ k × median(pre_clip_{t-W..t-1})`. The default window is 16
//! steps and the default multiplier is 10×. The detector must be
//! warm-up-aware: the first W steps have no comparison window and so are
//! never flagged.
//!
//! ## Run Command
//! ```bash
//! cargo run --example grad_norm_spike_detection
//! ```
//!
//! ## References
//! - aprender CRUX-F-09 (spike detection rule).
//! - Hampel (1974). *The Influence Curve and its Role in Robust Estimation*
//!   (rolling median is the canonical robust statistic for outlier detection).
//!
//! Added by PMAT-092 (expand-cookbooks followup — embeddings/search/grad-norm lint).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct SpikeFinding {
    pub step: u64,
    pub value: f64,
    pub rolling_median: f64,
    pub multiplier: f64,
}

pub fn detect_spikes(history: &Value, window: usize, mult: f64) -> Vec<SpikeFinding> {
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

    for i in window..series.len() {
        let (step, val) = series[i];
        let mut win: Vec<f64> = series[i - window..i].iter().map(|x| x.1).collect();
        win.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = win[win.len() / 2];
        if median > 0.0 && val >= mult * median {
            out.push(SpikeFinding {
                step,
                value: val,
                rolling_median: median,
                multiplier: val / median,
            });
        }
    }
    out
}

fn build_spiky_history() -> Value {
    let mut steps = Vec::new();
    // 20 calm steps near 0.5
    for s in 0..20u64 {
        steps.push(json!({ "step": s, "pre_clip": 0.5 + 0.01 * (s as f64).sin() }));
    }
    // Spike at step 20: 25.0 (50× over rolling median).
    steps.push(json!({ "step": 20u64, "pre_clip": 25.0 }));
    // Recovery
    for s in 21..30u64 {
        steps.push(json!({ "step": s, "pre_clip": 0.6 }));
    }
    json!({ "schema_version": 1, "steps": steps })
}

fn build_calm_history() -> Value {
    let mut steps = Vec::new();
    for s in 0..30u64 {
        steps.push(json!({ "step": s, "pre_clip": 0.5 + 0.05 * (s as f64).sin() }));
    }
    json!({ "schema_version": 1, "steps": steps })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("grad_norm_spike_detection")?;
    println!("=== Recipe: {} ===", ctx.name());

    let calm = detect_spikes(&build_calm_history(), 16, 10.0);
    println!("calm:  {} spikes", calm.len());

    let spiky = detect_spikes(&build_spiky_history(), 16, 10.0);
    println!("spiky: {} spikes", spiky.len());
    for f in &spiky {
        println!(
            "  step={}  value={:.3}  median={:.3}  mult={:.2}×",
            f.step, f.value, f.rolling_median, f.multiplier
        );
    }
    ctx.record_metric("spikes", spiky.len() as i64);
    ctx.record_string_metric("verdict", if spiky.is_empty() { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spike_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn calm_history_has_no_spikes() {
        let f = detect_spikes(&build_calm_history(), 16, 10.0);
        assert!(f.is_empty(), "expected calm: {f:?}");
    }

    #[test]
    fn isolated_spike_detected() {
        let f = detect_spikes(&build_spiky_history(), 16, 10.0);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].step, 20);
        assert!(f[0].multiplier > 10.0);
    }

    #[test]
    fn warmup_steps_never_flagged() {
        // First W steps have no comparison window — even an extreme value
        // there must NOT be flagged (would require historical context that
        // does not exist).
        let h = json!({
            "steps": [
                { "step": 0, "pre_clip": 100.0 },  // would flag if we naively compared
                { "step": 1, "pre_clip": 0.5 },
                { "step": 2, "pre_clip": 0.5 }
            ]
        });
        assert!(detect_spikes(&h, 2, 5.0).is_empty());
    }

    #[test]
    fn higher_multiplier_catches_fewer_spikes() {
        // Same data, stricter threshold = fewer findings.
        let h = build_spiky_history();
        let lax = detect_spikes(&h, 16, 5.0).len();
        let strict = detect_spikes(&h, 16, 100.0).len();
        assert!(strict <= lax);
    }

    #[test]
    fn zero_median_window_does_not_panic_or_divide() {
        // All-zero window → rolling_median = 0; rule explicitly skips
        // (median > 0 guard) so no findings and no divide-by-zero.
        let mut steps = Vec::new();
        for s in 0..20u64 {
            steps.push(json!({ "step": s, "pre_clip": 0.0 }));
        }
        steps.push(json!({ "step": 20u64, "pre_clip": 5.0 }));
        let h = json!({ "steps": steps });
        let f = detect_spikes(&h, 16, 10.0);
        assert!(f.is_empty());
    }
}
