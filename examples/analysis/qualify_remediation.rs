//! # Recipe: Qualification Workflow with Remediation Suggestions
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr qualify model.apr --remediate --plan markdown`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example qualify_remediation` exits 0
//! 2. [x] `cargo test --example qualify_remediation` passes
//! 3. [x] Deterministic output (no RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr qualify --remediate` in-process (no shell-out)
//! 10. [x] Unit tests cover suggestion per gap, ordering, no-op clean model
//!
//! ## Learning Objective
//! Demonstrates a remediation workflow: scan a model's gaps, emit a prioritized
//! action plan keyed on severity + cost, then produce a machine-readable list
//! of suggestions. Mirrors `apr qualify --remediate --plan markdown`.
//!
//! ## Run Command
//! ```bash
//! cargo run --example qualify_remediation
//! ```
//!
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning: A Survey of Case Studies*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gap {
    pub check: &'static str,
    pub severity: Severity,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Blocker,
    Major,
    Minor,
}

impl Severity {
    pub fn label(&self) -> &'static str {
        match self {
            Severity::Blocker => "blocker",
            Severity::Major => "major",
            Severity::Minor => "minor",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Suggestion {
    pub gap: &'static str,
    pub action: String,
    pub effort_hours: u32,
    pub severity: Severity,
}

pub fn suggest(gap: &Gap) -> Suggestion {
    let (action, hours) = match gap.check {
        "schema" => (
            "Re-export model with schema_version=2 via apr convert".to_string(),
            1u32,
        ),
        "hash" => ("Run `apr qa` to attach blake3 content hash".to_string(), 1),
        "signature" => (
            "Sign manifest with `apr publish --sign` and rotate key if stale".to_string(),
            2,
        ),
        "license" => (
            "Declare SPDX license tag in model card (e.g., Apache-2.0)".to_string(),
            1,
        ),
        "benchmark" => (
            "Improve benchmark score: distill, quantize, or replace backbone".to_string(),
            24,
        ),
        "memory" => (
            "Reduce activation memory via FlashAttention or tensor parallel".to_string(),
            8,
        ),
        "latency" => (
            "Reduce p99 latency via kernel fusion or speculative decoding".to_string(),
            8,
        ),
        "docs" => ("Author a HF-style model card (README.md)".to_string(), 2),
        _ => (format!("No recipe for check={}", gap.check), 0),
    };
    Suggestion {
        gap: gap.check,
        action,
        effort_hours: hours,
        severity: gap.severity,
    }
}

pub fn prioritize(suggestions: &mut [Suggestion]) {
    suggestions.sort_by(|a, b| {
        a.severity
            .cmp(&b.severity)
            .then_with(|| a.effort_hours.cmp(&b.effort_hours))
            .then_with(|| a.gap.cmp(b.gap))
    });
}

pub fn make_plan(gaps: &[Gap]) -> Vec<Suggestion> {
    let mut s: Vec<_> = gaps.iter().map(suggest).collect();
    prioritize(&mut s);
    s
}

fn demo_gaps() -> Vec<Gap> {
    vec![
        Gap {
            check: "docs",
            severity: Severity::Minor,
            description: "no model card".into(),
        },
        Gap {
            check: "signature",
            severity: Severity::Major,
            description: "unsigned manifest".into(),
        },
        Gap {
            check: "latency",
            severity: Severity::Blocker,
            description: "p99 780ms".into(),
        },
        Gap {
            check: "license",
            severity: Severity::Blocker,
            description: "missing SPDX".into(),
        },
    ]
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("qualify_remediation")?;
    println!("=== Recipe: {} ===", ctx.name());

    let gaps = demo_gaps();
    let plan = make_plan(&gaps);

    for s in &plan {
        println!(
            "  [{:<7}] {:<10} eff={}h  {}",
            s.severity.label(),
            s.gap,
            s.effort_hours,
            s.action
        );
    }

    let total_effort: u32 = plan.iter().map(|s| s.effort_hours).sum();
    let report = json!({
        "recipe": ctx.name(),
        "gap_count": gaps.len(),
        "total_effort_hours": total_effort,
        "plan": plan.iter().map(|s| json!({
            "gap": s.gap,
            "severity": s.severity.label(),
            "action": s.action,
            "effort_hours": s.effort_hours,
        })).collect::<Vec<_>>(),
    });
    let path = ctx.path("qualify-remediation.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    ctx.record_metric("gap_count", gaps.len() as i64);
    ctx.record_metric("total_effort_hours", i64::from(total_effort));
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_model_has_empty_plan() {
        let plan = make_plan(&[]);
        assert!(plan.is_empty());
    }

    #[test]
    fn signature_suggestion_is_nontrivial() {
        let g = Gap {
            check: "signature",
            severity: Severity::Major,
            description: "x".into(),
        };
        let s = suggest(&g);
        assert!(s.effort_hours > 0);
        assert!(s.action.contains("sign") || s.action.contains("Sign"));
    }

    #[test]
    fn priority_puts_blocker_first() {
        let plan = make_plan(&demo_gaps());
        assert_eq!(plan[0].severity, Severity::Blocker);
    }

    #[test]
    fn priority_is_stable_within_severity() {
        let gaps = vec![
            Gap {
                check: "latency",
                severity: Severity::Blocker,
                description: "".into(),
            },
            Gap {
                check: "license",
                severity: Severity::Blocker,
                description: "".into(),
            },
        ];
        let plan = make_plan(&gaps);
        // license (1h) < latency (8h) → license first
        assert_eq!(plan[0].gap, "license");
    }

    #[test]
    fn unknown_gap_suggests_zero_hours() {
        let g = Gap {
            check: "nosuch",
            severity: Severity::Minor,
            description: "".into(),
        };
        let s = suggest(&g);
        assert_eq!(s.effort_hours, 0);
    }
}
