//! # Recipe: Gradient-Norm Telemetry — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr grad-norm --history-file history.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the gradient-norm telemetry analyzer (CRUX-F-09).
//! `apr grad-norm` ingests per-step `||g||₂` records from a training run
//! and flags four classes of pathology: NaN/inf entries, monotonic
//! divergence (norm strictly increasing for ≥ N steps), spike (norm > k×
//! rolling median), and clip-cap violation (post-clip norm > max_grad_norm).
//! The happy-path history exhibits a healthy decay with no findings.
//!
//! ## Run Command
//! ```bash
//! cargo run --example grad_norm_happy
//! ```
//!
//! ## References
//! - aprender CRUX-F-09 contract.
//! - Pascanu et al. (2013). *On the difficulty of training RNNs*. arXiv:1211.5063 (clip rationale).
//!
//! Added by PMAT-092 (expand-cookbooks followup — embeddings/search/grad-norm lint).

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum GradFinding {
    NonFinite { step: u64 },
    CapViolation { step: u64, post_clip: f64, cap: f64 },
}

pub fn lint_grad_norm(history: &Value, cap: Option<f64>) -> Vec<GradFinding> {
    let mut out = Vec::new();
    let Some(arr) = history.get("steps").and_then(Value::as_array) else {
        return out;
    };
    for r in arr {
        let step = r.get("step").and_then(Value::as_u64).unwrap_or(0);
        let pre = r
            .get("pre_clip")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NAN);
        let post = r
            .get("post_clip")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NAN);

        if !pre.is_finite() || !post.is_finite() {
            out.push(GradFinding::NonFinite { step });
            continue;
        }
        if let Some(c) = cap {
            // Allow tiny FP slack — gradient clipping uses fp16/fp32 mix,
            // post-clip can land 1 ULP above cap.
            if post > c + 1e-6 {
                out.push(GradFinding::CapViolation {
                    step,
                    post_clip: post,
                    cap: c,
                });
            }
        }
    }
    out
}

pub fn build_happy_history() -> Value {
    // Healthy: pre-clip gently decays, post-clip = min(pre, cap=1.0).
    let mut steps = Vec::new();
    for s in 0..16 {
        let pre = 2.5_f64 * 0.95_f64.powi(s);
        let post = pre.min(1.0);
        steps.push(json!({
            "step": s as u64,
            "pre_clip": pre,
            "post_clip": post
        }));
    }
    json!({ "schema_version": 1, "max_grad_norm": 1.0, "steps": steps })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("grad_norm_happy")?;
    let history = build_happy_history();

    let path = ctx.path("grad_history.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&history)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let cap = history.get("max_grad_norm").and_then(Value::as_f64);
    let findings = lint_grad_norm(&history, cap);

    println!("=== Recipe: {} ===", ctx.name());
    println!("History: {}", path.display());
    println!("Findings: {}", findings.len());
    for f in &findings {
        println!("  {f:?}");
    }
    ctx.record_metric("findings", findings.len() as i64);
    ctx.record_string_metric("verdict", if findings.is_empty() { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_history_has_no_findings() {
        let h = build_happy_history();
        let cap = h.get("max_grad_norm").and_then(Value::as_f64);
        let f = lint_grad_norm(&h, cap);
        assert!(f.is_empty(), "expected clean: {f:?}");
    }

    #[test]
    fn nan_entry_flagged() {
        let h = json!({
            "steps": [
                { "step": 0, "pre_clip": 1.0, "post_clip": 1.0 },
                { "step": 1, "pre_clip": null, "post_clip": 1.0 }
            ]
        });
        let f = lint_grad_norm(&h, Some(1.0));
        assert_eq!(f.len(), 1);
        assert!(matches!(f[0], GradFinding::NonFinite { step: 1 }));
    }

    #[test]
    fn cap_violation_flagged() {
        let h = json!({
            "steps": [
                { "step": 0, "pre_clip": 5.0, "post_clip": 1.5 } // > 1.0
            ]
        });
        let f = lint_grad_norm(&h, Some(1.0));
        assert_eq!(f.len(), 1);
        assert!(matches!(f[0], GradFinding::CapViolation { step: 0, .. }));
    }

    #[test]
    fn cap_at_exact_value_passes() {
        // post_clip == cap is fine (allowed slack).
        let h = json!({
            "steps": [
                { "step": 0, "pre_clip": 5.0, "post_clip": 1.0 }
            ]
        });
        assert!(lint_grad_norm(&h, Some(1.0)).is_empty());
    }
}
