//! # Recipe: Runs Diff — Compare Two Experiment Runs
//!
//! **Category**: cli
//! **CLI Equivalent**: `apr runs diff run-A run-B`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example runs_diff` exits 0
//! 2. [x] `cargo test --example runs_diff` passes
//! 3. [x] Deterministic output (fixed fixtures)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr runs diff` in-process (no shell-out)
//! 10. [x] Unit tests cover hyperparam diff, metric delta, shared/unique keys
//!
//! ## Learning Objective
//! Demonstrates pairwise comparison of experiment runs: compute symmetric
//! differences on hyperparameters and signed deltas on shared metrics. The
//! output mirrors how `apr runs diff` surfaces regressions and improvements.
//!
//! ## Run Command
//! ```bash
//! cargo run --example runs_diff
//! ```
//!
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning: A Survey of Case Studies*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct RunRecord {
    pub id: String,
    pub hyperparams: BTreeMap<String, String>,
    pub metrics: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyperparamDiff {
    pub key: String,
    pub left: Option<String>,
    pub right: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricDelta {
    pub key: String,
    pub left: f64,
    pub right: f64,
    pub delta: f64,
}

#[derive(Debug, Clone)]
pub struct RunsDiff {
    pub hyperparam_diffs: Vec<HyperparamDiff>,
    pub metric_deltas: Vec<MetricDelta>,
    pub only_in_left: Vec<String>,
    pub only_in_right: Vec<String>,
}

pub fn diff_runs(left: &RunRecord, right: &RunRecord) -> RunsDiff {
    let mut hp_diffs = Vec::new();
    let all_hp_keys: std::collections::BTreeSet<&String> = left
        .hyperparams
        .keys()
        .chain(right.hyperparams.keys())
        .collect();
    for k in all_hp_keys {
        let lv = left.hyperparams.get(k);
        let rv = right.hyperparams.get(k);
        if lv != rv {
            hp_diffs.push(HyperparamDiff {
                key: k.clone(),
                left: lv.cloned(),
                right: rv.cloned(),
            });
        }
    }

    let mut metric_deltas = Vec::new();
    let mut only_left = Vec::new();
    let mut only_right = Vec::new();
    for (k, lv) in &left.metrics {
        if let Some(rv) = right.metrics.get(k) {
            metric_deltas.push(MetricDelta {
                key: k.clone(),
                left: *lv,
                right: *rv,
                delta: rv - lv,
            });
        } else {
            only_left.push(k.clone());
        }
    }
    for k in right.metrics.keys() {
        if !left.metrics.contains_key(k) {
            only_right.push(k.clone());
        }
    }

    RunsDiff {
        hyperparam_diffs: hp_diffs,
        metric_deltas,
        only_in_left: only_left,
        only_in_right: only_right,
    }
}

fn demo_run(id: &str, lr: &str, batch: &str, loss: f64, acc: f64) -> RunRecord {
    let mut hp = BTreeMap::new();
    hp.insert("learning_rate".into(), lr.to_string());
    hp.insert("batch_size".into(), batch.to_string());
    hp.insert("optimizer".into(), "adamw".into());
    let mut m = BTreeMap::new();
    m.insert("final_loss".into(), loss);
    m.insert("accuracy".into(), acc);
    RunRecord {
        id: id.to_string(),
        hyperparams: hp,
        metrics: m,
    }
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("runs_diff")?;
    println!("=== Recipe: {} ===", ctx.name());

    let mut a = demo_run("run-A", "1e-4", "32", 0.321, 0.892);
    a.metrics.insert("latency_ms".into(), 12.4);
    let b = demo_run("run-B", "3e-4", "32", 0.287, 0.901);

    let diff = diff_runs(&a, &b);
    println!("Compared {} vs {}", a.id, b.id);
    println!("\n-- Hyperparam Diffs ({}) --", diff.hyperparam_diffs.len());
    for d in &diff.hyperparam_diffs {
        println!(
            "  {}: {:?} -> {:?}",
            d.key,
            d.left.as_deref().unwrap_or("(missing)"),
            d.right.as_deref().unwrap_or("(missing)")
        );
    }
    println!("\n-- Metric Deltas ({}) --", diff.metric_deltas.len());
    for d in &diff.metric_deltas {
        println!(
            "  {}: {:.4} -> {:.4}  (Δ={:+.4})",
            d.key, d.left, d.right, d.delta
        );
    }
    if !diff.only_in_left.is_empty() {
        println!("\nOnly in {}: {:?}", a.id, diff.only_in_left);
    }
    if !diff.only_in_right.is_empty() {
        println!("Only in {}: {:?}", b.id, diff.only_in_right);
    }

    let report = json!({
        "recipe": ctx.name(),
        "left": a.id,
        "right": b.id,
        "n_hyperparam_diffs": diff.hyperparam_diffs.len(),
        "n_metric_deltas": diff.metric_deltas.len(),
        "metric_deltas": diff.metric_deltas.iter().map(|d| json!({
            "key": d.key,
            "left": d.left,
            "right": d.right,
            "delta": d.delta,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("runs-diff.json");
    std::fs::write(
        &out,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn differing_lr_appears_in_diff() {
        let a = demo_run("a", "1e-4", "32", 0.3, 0.8);
        let b = demo_run("b", "3e-4", "32", 0.3, 0.8);
        let d = diff_runs(&a, &b);
        assert!(d.hyperparam_diffs.iter().any(|h| h.key == "learning_rate"));
    }

    #[test]
    fn identical_runs_yield_empty_diff() {
        let a = demo_run("a", "1e-4", "32", 0.3, 0.8);
        let b = demo_run("a", "1e-4", "32", 0.3, 0.8);
        let d = diff_runs(&a, &b);
        assert!(d.hyperparam_diffs.is_empty());
        assert!(d.metric_deltas.iter().all(|m| m.delta.abs() < 1e-12));
    }

    #[test]
    fn metric_delta_is_right_minus_left() {
        let a = demo_run("a", "1e-4", "32", 0.4, 0.8);
        let b = demo_run("b", "1e-4", "32", 0.3, 0.8);
        let d = diff_runs(&a, &b);
        let loss_delta = d
            .metric_deltas
            .iter()
            .find(|m| m.key == "final_loss")
            .expect("should have final_loss");
        assert!((loss_delta.delta - (-0.1)).abs() < 1e-9);
    }

    #[test]
    fn detects_only_in_left() {
        let mut a = demo_run("a", "1e-4", "32", 0.3, 0.8);
        a.metrics.insert("lat".into(), 1.0);
        let b = demo_run("b", "1e-4", "32", 0.3, 0.8);
        let d = diff_runs(&a, &b);
        assert!(d.only_in_left.iter().any(|k| k == "lat"));
    }

    #[test]
    fn detects_only_in_right() {
        let a = demo_run("a", "1e-4", "32", 0.3, 0.8);
        let mut b = demo_run("b", "1e-4", "32", 0.3, 0.8);
        b.metrics.insert("throughput".into(), 42.0);
        let d = diff_runs(&a, &b);
        assert!(d.only_in_right.iter().any(|k| k == "throughput"));
    }
}
