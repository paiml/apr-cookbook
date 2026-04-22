//! # Recipe: OOM-Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr oom-lint postmortem.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example oom_lint_happy` exits 0
//! 2. [x] `cargo test --example oom_lint_happy` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Validates a CUDA OOM postmortem bundle: the device identity
//! (`gpu_index`, `gpu_name`), the memory triplet (`requested`, `free`,
//! `total`) and the root-cause breadcrumb trail (a bottom-up frame list
//! leading to the allocation site). The happy-path report has all fields
//! populated and the triplet is internally consistent.
//!
//! ## Run Command
//! ```bash
//! cargo run --example oom_lint_happy
//! ```
//!
//! ## References
//! - Ren, J. et al. (2021). *ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning*. arXiv:2104.07857

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub rule: &'static str,
    pub severity: &'static str,
    pub message: String,
}

pub fn lint_oom(obs: &Value) -> Vec<Finding> {
    let mut out = Vec::new();
    for f in [
        "gpu_index",
        "gpu_name",
        "requested_bytes",
        "free_bytes",
        "total_bytes",
    ] {
        if obs.get(f).is_none() {
            out.push(Finding {
                rule: "OOM-001",
                severity: "error",
                message: format!("missing required field `{f}`"),
            });
        }
    }
    let (req, free, total) = (
        obs.get("requested_bytes")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        obs.get("free_bytes").and_then(Value::as_u64).unwrap_or(0),
        obs.get("total_bytes").and_then(Value::as_u64).unwrap_or(0),
    );
    if free > total {
        out.push(Finding {
            rule: "OOM-002",
            severity: "error",
            message: "free_bytes > total_bytes".into(),
        });
    }
    if req <= free {
        out.push(Finding {
            rule: "OOM-003",
            severity: "warn",
            message: "requested_bytes <= free_bytes — not an OOM".into(),
        });
    }
    let frames = obs
        .get("breadcrumbs")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    if frames == 0 {
        out.push(Finding {
            rule: "OOM-004",
            severity: "error",
            message: "breadcrumbs empty — no stack to diagnose".into(),
        });
    }
    out
}

fn build_happy() -> Value {
    json!({
        "schema_version": 1,
        "gpu_index": 0,
        "gpu_name": "NVIDIA A100 80GB",
        "requested_bytes": 20_000_000_000u64,
        "free_bytes":       4_500_000_000u64,
        "total_bytes":     80_000_000_000u64,
        "allocator": "caching",
        "breadcrumbs": [
            {"frame": 0, "fn": "cudaMalloc"},
            {"frame": 1, "fn": "nn::Linear::forward"},
            {"frame": 2, "fn": "train_step"},
            {"frame": 3, "fn": "main"}
        ],
        "timestamp": "2026-04-22T12:00:00Z"
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("oom_lint_happy")?;
    let obs = build_happy();
    let p = ctx.path("oom.json");
    std::fs::write(
        &p,
        serde_json::to_vec_pretty(&obs).map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_oom(&obs);
    let errors = findings.iter().filter(|f| f.severity == "error").count();

    println!("=== Recipe: {} ===", ctx.name());
    println!("Postmortem: {}", p.display());
    println!("Findings: {}", findings.len());
    for f in &findings {
        println!("  [{}] {} — {}", f.severity, f.rule, f.message);
    }

    ctx.record_metric("errors", errors as i64);
    ctx.record_string_metric("verdict", if errors == 0 { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_is_clean_of_errors() {
        let f = lint_oom(&build_happy());
        assert_eq!(f.iter().filter(|x| x.severity == "error").count(), 0);
    }

    #[test]
    fn empty_breadcrumbs_flags_oom_004() {
        let mut obs = build_happy();
        obs["breadcrumbs"] = json!([]);
        let f = lint_oom(&obs);
        assert!(f.iter().any(|x| x.rule == "OOM-004"));
    }

    #[test]
    fn free_gt_total_flags_oom_002() {
        let mut obs = build_happy();
        obs["free_bytes"] = json!(100_000_000_000u64);
        let f = lint_oom(&obs);
        assert!(f.iter().any(|x| x.rule == "OOM-002"));
    }
}
