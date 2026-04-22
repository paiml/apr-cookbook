//! # Recipe: OOM-Lint — Missing Breadcrumbs Edge Case
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr oom-lint postmortem.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example oom_lint_missing_breadcrumb` exits 0
//! 2. [x] `cargo test --example oom_lint_missing_breadcrumb` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! A postmortem bundle with a present but EMPTY `breadcrumbs` array is the
//! most common real-world failure — the framework caught the OOM but the
//! stack-unwind attempt failed. Shows how `OOM-004` catches it and why a
//! fallback to `/proc/self/maps`-style diagnostics would help.
//!
//! ## Run Command
//! ```bash
//! cargo run --example oom_lint_missing_breadcrumb
//! ```
//!
//! ## References
//! - Ren, J. et al. (2021). *ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning*. arXiv:2104.07857

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct Finding {
    pub rule: &'static str,
    pub severity: &'static str,
    pub message: String,
}

pub fn lint_oom(obs: &Value) -> Vec<Finding> {
    let mut out = Vec::new();
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
    if frames > 0 && frames < 3 {
        out.push(Finding {
            rule: "OOM-005",
            severity: "warn",
            message: "breadcrumbs has < 3 frames — likely incomplete unwind".into(),
        });
    }
    out
}

/// What to suggest when the stack is missing.
pub fn fallback_suggestions(obs: &Value) -> Vec<&'static str> {
    let mut out = vec!["enable CUDA_LAUNCH_BLOCKING=1 to serialise kernel launches"];
    if obs.get("allocator").and_then(Value::as_str) == Some("caching") {
        out.push("set PYTORCH_NO_CUDA_MEMORY_CACHING=1 to bypass the caching allocator");
    }
    out.push("capture nvidia-smi --query-gpu=memory.used --format=csv before OOM");
    out
}

fn build_missing() -> Value {
    json!({
        "schema_version": 1,
        "gpu_index": 0,
        "gpu_name": "NVIDIA A100 80GB",
        "requested_bytes": 24_000_000_000u64,
        "free_bytes":       2_000_000_000u64,
        "total_bytes":     80_000_000_000u64,
        "allocator": "caching",
        "breadcrumbs": [], // deliberately empty
        "timestamp": "2026-04-22T12:00:00Z"
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("oom_lint_missing_breadcrumb")?;
    let obs = build_missing();
    let p = ctx.path("oom.json");
    std::fs::write(
        &p,
        serde_json::to_vec_pretty(&obs).map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_oom(&obs);
    let suggestions = fallback_suggestions(&obs);

    println!("=== Recipe: {} ===", ctx.name());
    for f in &findings {
        println!("  [{}] {} — {}", f.severity, f.rule, f.message);
    }
    println!("\nFallback suggestions:");
    for s in &suggestions {
        println!("  - {s}");
    }

    ctx.record_metric(
        "errors",
        findings.iter().filter(|f| f.severity == "error").count() as i64,
    );
    ctx.record_metric("suggestions", suggestions.len() as i64);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_breadcrumbs_flags_oom_004() {
        let f = lint_oom(&build_missing());
        assert!(f.iter().any(|x| x.rule == "OOM-004"));
    }

    #[test]
    fn partial_breadcrumbs_warns_oom_005() {
        let mut obs = build_missing();
        obs["breadcrumbs"] = json!([{"frame":0,"fn":"cudaMalloc"}]);
        let f = lint_oom(&obs);
        assert!(f.iter().any(|x| x.rule == "OOM-005"));
    }

    #[test]
    fn caching_allocator_gets_extra_suggestion() {
        let s = fallback_suggestions(&build_missing());
        assert!(s
            .iter()
            .any(|x| x.contains("PYTORCH_NO_CUDA_MEMORY_CACHING")));
    }
}
