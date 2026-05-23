//! # Recipe: Lint — Suppression Handling (allow / deny / expect)
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr lint model.apr --suppressions .aprlint-allow`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example lint_suppression` exits 0
//! 2. [x] `cargo test --example lint_suppression` passes
//! 3. [x] Deterministic output (pure)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr lint --suppressions` in-process (no shell-out)
//! 10. [x] Unit tests cover allow/deny/expect, matching, unused suppressions
//!
//! ## Learning Objective
//! Implements a tri-state lint suppression engine: `allow` hides a finding,
//! `deny` upgrades it to error, and `expect` requires the finding to appear
//! (emits a new finding if it doesn't — catching stale suppressions, a
//! common technical-debt source).
//!
//! ## Run Command
//! ```bash
//! cargo run --example lint_suppression
//! ```
//!
//! ## References
//! - Baker, B.S. (1995). *On Finding Duplication in Large Software Systems*. WCRE. DOI: 10.1109/WCRE.1995.514697

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
struct RawFinding {
    rule: String,
    tensor: String,
    severity: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Policy {
    Allow,
    Deny,
    Expect,
}

#[derive(Debug, Clone)]
struct Suppression {
    rule: String,
    tensor_glob: String,
    policy: Policy,
}

#[derive(Debug, Clone)]
struct ProcessedFinding {
    rule: String,
    tensor: String,
    severity: String,
    suppressed_by: Option<Policy>,
}

fn matches_glob(glob: &str, name: &str) -> bool {
    if glob == "*" {
        return true;
    }
    if let Some(prefix) = glob.strip_suffix('*') {
        return name.starts_with(prefix);
    }
    glob == name
}

fn apply_suppressions(
    findings: &[RawFinding],
    suppressions: &[Suppression],
) -> (Vec<ProcessedFinding>, Vec<Suppression>) {
    let mut out = Vec::new();
    let mut used: BTreeSet<(String, String, Policy)> = BTreeSet::new();
    for f in findings {
        let mut applied: Option<Policy> = None;
        let mut new_severity = f.severity.clone();
        for s in suppressions {
            if s.rule == f.rule && matches_glob(&s.tensor_glob, &f.tensor) {
                applied = Some(s.policy);
                used.insert((s.rule.clone(), s.tensor_glob.clone(), s.policy));
                match s.policy {
                    Policy::Allow => {}
                    Policy::Deny => new_severity = "error".into(),
                    Policy::Expect => {}
                }
                break;
            }
        }
        if matches!(applied, Some(Policy::Allow)) {
            // Drop from findings entirely.
            continue;
        }
        out.push(ProcessedFinding {
            rule: f.rule.clone(),
            tensor: f.tensor.clone(),
            severity: new_severity,
            suppressed_by: applied,
        });
    }
    // Unused suppressions: an `expect` that never matched becomes a stale finding.
    let unused: Vec<Suppression> = suppressions
        .iter()
        .filter(|s| {
            s.policy == Policy::Expect
                && !used.contains(&(s.rule.clone(), s.tensor_glob.clone(), s.policy))
        })
        .cloned()
        .collect();
    (out, unused)
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("lint_suppression")?;
    println!("=== Recipe: {} ===", ctx.name());

    let findings = vec![
        RawFinding {
            rule: "NAME-001".into(),
            tensor: "LAYER.0.attn.qkv".into(),
            severity: "error".into(),
        },
        RawFinding {
            rule: "NAME-003".into(),
            tensor: "rogue_tensor".into(),
            severity: "warn".into(),
        },
        RawFinding {
            rule: "PERF-010".into(),
            tensor: "layer.3.ffn.up".into(),
            severity: "warn".into(),
        },
    ];

    let suppressions = vec![
        Suppression {
            rule: "NAME-003".into(),
            tensor_glob: "rogue_*".into(),
            policy: Policy::Allow,
        },
        Suppression {
            rule: "PERF-010".into(),
            tensor_glob: "layer.*".into(),
            policy: Policy::Deny,
        },
        Suppression {
            rule: "STALE-999".into(), // never fires
            tensor_glob: "*".into(),
            policy: Policy::Expect,
        },
    ];

    let (processed, unused) = apply_suppressions(&findings, &suppressions);
    println!("\n--- Processed findings ---");
    for f in &processed {
        println!(
            "  [{}] {} {} {}",
            f.severity,
            f.rule,
            f.tensor,
            match f.suppressed_by {
                Some(p) => format!("({:?})", p),
                None => String::new(),
            }
        );
    }
    if !unused.is_empty() {
        println!("\n--- Stale suppressions (expected but not found) ---");
        for u in &unused {
            println!("  expect {} on '{}' unused", u.rule, u.tensor_glob);
        }
    }

    let report = json!({
        "recipe": ctx.name(),
        "findings": processed.iter().map(|f| json!({
            "rule": f.rule,
            "tensor": f.tensor,
            "severity": f.severity,
            "suppressed_by": f.suppressed_by.map(|p| format!("{:?}", p)),
        })).collect::<Vec<_>>(),
        "unused_suppressions": unused.iter().map(|s| json!({
            "rule": s.rule,
            "tensor_glob": s.tensor_glob,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("lint-suppression.json");
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out, bytes)?;

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(rule: &str, tensor: &str, sev: &str) -> RawFinding {
        RawFinding {
            rule: rule.into(),
            tensor: tensor.into(),
            severity: sev.into(),
        }
    }

    fn s(rule: &str, glob: &str, p: Policy) -> Suppression {
        Suppression {
            rule: rule.into(),
            tensor_glob: glob.into(),
            policy: p,
        }
    }

    #[test]
    fn glob_star_matches_everything() {
        assert!(matches_glob("*", "anything"));
    }

    #[test]
    fn glob_prefix_star_matches() {
        assert!(matches_glob("layer.*", "layer.0.attn"));
        assert!(!matches_glob("layer.*", "embed.weight"));
    }

    #[test]
    fn allow_drops_finding() {
        let (out, _) = apply_suppressions(&[f("R1", "x", "warn")], &[s("R1", "x", Policy::Allow)]);
        assert!(out.is_empty());
    }

    #[test]
    fn deny_upgrades_to_error() {
        let (out, _) = apply_suppressions(&[f("R1", "x", "warn")], &[s("R1", "x", Policy::Deny)]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].severity, "error");
    }

    #[test]
    fn unused_expect_reported() {
        let (_, unused) =
            apply_suppressions(&[f("R1", "x", "warn")], &[s("NONE", "*", Policy::Expect)]);
        assert_eq!(unused.len(), 1);
    }

    #[test]
    fn used_expect_not_reported() {
        let (_, unused) =
            apply_suppressions(&[f("R1", "x", "warn")], &[s("R1", "x", Policy::Expect)]);
        assert!(unused.is_empty());
    }
}
