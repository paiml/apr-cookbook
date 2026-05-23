//! # Recipe: DRY-Sampling Lint — CI Pipeline Composition
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr dry-sampling-lint --glob 'runs/*/dry.json'`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example dry_sampling_lint_pipeline` exits 0
//! 2. [x] `cargo test --example dry_sampling_lint_pipeline` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Wires `dry-sampling-lint` into a multi-run sweep: lint several generation
//! observations simultaneously, pick the run with the lowest repetition ratio,
//! and emit a recommendation block suitable for a CI summary comment.
//!
//! ## Run Command
//! ```bash
//! cargo run --example dry_sampling_lint_pipeline
//! ```
//!
//! ## References
//! - Xu, C. et al. (2024). *DRY: A Modern Repetition Penalty That Preserves Creativity*. arXiv:2409.00509

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct RunSummary {
    pub name: String,
    pub multiplier: f64,
    pub repetition_ratio: f64,
    pub errors: usize,
}

pub fn summarise(name: &str, obs: &Value) -> RunSummary {
    let mult = obs.get("multiplier").and_then(Value::as_f64).unwrap_or(0.0);
    let lm = obs
        .get("longest_match")
        .and_then(Value::as_u64)
        .unwrap_or(0) as f64;
    let em = obs
        .get("emitted_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .max(1) as f64;
    let ratio = lm / em;
    let errors = usize::from(ratio > 0.10) + usize::from(mult <= 0.0);
    RunSummary {
        name: name.to_string(),
        multiplier: mult,
        repetition_ratio: ratio,
        errors,
    }
}

pub fn pick_best(runs: &[RunSummary]) -> Option<&RunSummary> {
    runs.iter()
        .filter(|r| r.errors == 0)
        .min_by(|a, b| a.repetition_ratio.total_cmp(&b.repetition_ratio))
}

fn build_runs() -> Vec<(String, Value)> {
    vec![
        (
            "run_a_baseline".into(),
            json!({
                "multiplier": 0.2, "base": 1.75, "allowed_length": 2,
                "emitted_tokens": 256, "longest_match": 48
            }),
        ),
        (
            "run_b_tuned".into(),
            json!({
                "multiplier": 0.8, "base": 1.75, "allowed_length": 2,
                "emitted_tokens": 256, "longest_match": 9
            }),
        ),
        (
            "run_c_aggressive".into(),
            json!({
                "multiplier": 1.6, "base": 2.0, "allowed_length": 2,
                "emitted_tokens": 256, "longest_match": 4
            }),
        ),
    ]
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("dry_sampling_lint_pipeline")?;

    let runs = build_runs();
    let mut summaries = Vec::new();
    for (name, obs) in &runs {
        let p = ctx.path(&format!("{name}.json"));
        std::fs::write(
            &p,
            serde_json::to_vec(&obs).map_err(|e| CookbookError::Serialization(e.to_string()))?,
        )?;
        summaries.push(summarise(name, obs));
    }

    let best = pick_best(&summaries);

    println!("=== Recipe: {} ===", ctx.name());
    println!(
        "{:<20}  {:>10}  {:>10}  {:>6}",
        "run", "multiplier", "rep_ratio", "errors"
    );
    for s in &summaries {
        println!(
            "{:<20}  {:>10.2}  {:>10.4}  {:>6}",
            s.name, s.multiplier, s.repetition_ratio, s.errors
        );
    }
    if let Some(b) = best {
        println!(
            "\nRecommendation: {} (multiplier={}, rep_ratio={:.4})",
            b.name, b.multiplier, b.repetition_ratio
        );
        ctx.record_string_metric("recommended_run", &b.name);
    }

    let failing = summaries.iter().filter(|s| s.errors > 0).count();
    ctx.record_metric("failing_runs", failing as i64);
    ctx.record_metric("total_runs", summaries.len() as i64);
    ctx.record_string_metric("verdict", if failing == 0 { "PASS" } else { "PARTIAL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_picks_lowest_repetition() {
        let runs = build_runs();
        let summaries: Vec<RunSummary> = runs.iter().map(|(n, v)| summarise(n, v)).collect();
        let best = pick_best(&summaries).expect("expected a clean run");
        assert_eq!(best.name, "run_c_aggressive");
    }

    #[test]
    fn baseline_has_errors() {
        let (n, v) = &build_runs()[0];
        let s = summarise(n, v);
        assert!(s.errors >= 1);
    }

    #[test]
    fn tuned_run_is_clean() {
        let (n, v) = &build_runs()[1];
        let s = summarise(n, v);
        assert_eq!(s.errors, 0);
    }
}
