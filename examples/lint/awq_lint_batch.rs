//! # Recipe: AWQ Lint — Batch / Pipeline Composition
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr awq-lint *.json | jq ...`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example awq_lint_batch` exits 0
//! 2. [x] `cargo test --example awq_lint_batch` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Lints a BATCH of AWQ observation files in a single sweep (the composition
//! pattern a CI gate uses) and emits a SARIF-like aggregate report so downstream
//! tools can fail-the-build on any error-severity finding.
//!
//! ## Run Command
//! ```bash
//! cargo run --example awq_lint_batch
//! ```
//!
//! ## References
//! - Lin, J. et al. (2024). *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration*. arXiv:2306.00978
//! - OASIS. *Static Analysis Results Interchange Format (SARIF) v2.1.0*.

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct Report {
    pub file: String,
    pub errors: usize,
    pub warnings: usize,
}

pub fn lint_group_size(obs: &Value) -> bool {
    matches!(obs.get("group_size").and_then(Value::as_u64),
        Some(g) if (32..=256).contains(&g) && g.is_power_of_two())
}

pub fn lint_bits(obs: &Value) -> bool {
    matches!(obs.get("bits").and_then(Value::as_u64), Some(3 | 4))
}

pub fn lint_clip_ratio(obs: &Value) -> bool {
    matches!(obs.get("clip_ratio").and_then(Value::as_f64),
        Some(c) if c > 0.0 && c <= 1.0)
}

pub fn lint_one(obs: &Value) -> (usize, usize) {
    let mut errors = 0usize;
    let warnings = 0usize;
    if !lint_group_size(obs) {
        errors += 1;
    }
    if !lint_bits(obs) {
        errors += 1;
    }
    if !lint_clip_ratio(obs) {
        errors += 1;
    }
    (errors, warnings)
}

pub fn aggregate(reports: &[Report]) -> Value {
    let total_err: usize = reports.iter().map(|r| r.errors).sum();
    let total_warn: usize = reports.iter().map(|r| r.warnings).sum();
    json!({
        "tool": "awq-lint",
        "version": "1.0.0",
        "files": reports.iter().map(|r| json!({
            "path": r.file,
            "errors": r.errors,
            "warnings": r.warnings,
        })).collect::<Vec<_>>(),
        "totals": { "errors": total_err, "warnings": total_warn },
        "verdict": if total_err == 0 { "PASS" } else { "FAIL" },
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("awq_lint_batch")?;

    let batch = vec![
        (
            "ok_a.json",
            json!({"schema_version":1,"group_size":128,"bits":4,"clip_ratio":0.9}),
        ),
        (
            "ok_b.json",
            json!({"schema_version":1,"group_size":64, "bits":3,"clip_ratio":0.85}),
        ),
        (
            "bad_group.json",
            json!({"schema_version":1,"group_size":96, "bits":4,"clip_ratio":0.9}),
        ),
        (
            "bad_bits.json",
            json!({"schema_version":1,"group_size":128,"bits":8,"clip_ratio":0.9}),
        ),
    ];

    let mut reports = Vec::new();
    for (name, obs) in &batch {
        let p = ctx.path(name);
        std::fs::write(
            &p,
            serde_json::to_vec(&obs).map_err(|e| CookbookError::Serialization(e.to_string()))?,
        )?;
        let (errors, warnings) = lint_one(obs);
        reports.push(Report {
            file: (*name).to_string(),
            errors,
            warnings,
        });
    }

    let report = aggregate(&reports);
    let verdict = report["verdict"].as_str().unwrap_or("FAIL");
    let total_err = report["totals"]["errors"].as_u64().unwrap_or(0);

    println!("=== Recipe: {} ===", ctx.name());
    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?
    );

    ctx.record_metric("files", reports.len() as i64);
    ctx.record_metric("total_errors", total_err as i64);
    ctx.record_string_metric("verdict", verdict);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_counts_errors() {
        let reports = vec![
            Report {
                file: "a.json".into(),
                errors: 0,
                warnings: 0,
            },
            Report {
                file: "b.json".into(),
                errors: 2,
                warnings: 1,
            },
        ];
        let r = aggregate(&reports);
        assert_eq!(r["totals"]["errors"], 2);
        assert_eq!(r["verdict"], "FAIL");
    }

    #[test]
    fn lint_one_good() {
        let obs = json!({"schema_version":1,"group_size":128,"bits":4,"clip_ratio":0.9});
        assert_eq!(lint_one(&obs), (0, 0));
    }

    #[test]
    fn lint_one_bad_group() {
        let obs = json!({"schema_version":1,"group_size":100,"bits":4,"clip_ratio":0.9});
        assert_eq!(lint_one(&obs), (1, 0));
    }
}
