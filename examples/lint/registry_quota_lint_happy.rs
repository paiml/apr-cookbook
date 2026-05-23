//! # Recipe: Registry Quota Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr registry-quota-lint --observation-file observation.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the registry byte-quota lint pipeline (CRUX-A-22). The
//! observation records the registry's current size, the configured ceiling,
//! the per-tenant high-water marks, and the atomic-write counter. The lint
//! enforces five invariants: ceiling > 0, current_bytes <= ceiling, no
//! tenant exceeds its individual quota, atomic_writes == atomic_commits,
//! and the schema_version is current.
//!
//! ## Run Command
//! ```bash
//! cargo run --example registry_quota_lint_happy
//! ```
//!
//! ## References
//! - aprender CRUX-A-22 contract (registry byte-quota observation).
//! - SHIP-009 sovereign-stack registry layout.
//!
//! Added by PMAT-090 (expand-cookbooks followup — registry/cache lint coverage).

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintFinding {
    pub rule: String,
    pub severity: &'static str,
    pub message: String,
}

pub fn lint_quota_observation(obs: &Value) -> Vec<LintFinding> {
    let mut out = Vec::new();

    // Rule 1: schema_version present and >= 1.
    match obs.get("schema_version").and_then(Value::as_u64) {
        Some(v) if v >= 1 => {}
        _ => out.push(LintFinding {
            rule: "QUOTA-001".into(),
            severity: "error",
            message: "schema_version missing or < 1".into(),
        }),
    }

    // Rule 2: ceiling_bytes must be > 0.
    let ceiling = obs.get("ceiling_bytes").and_then(Value::as_u64);
    match ceiling {
        Some(c) if c > 0 => {}
        _ => out.push(LintFinding {
            rule: "QUOTA-002".into(),
            severity: "error",
            message: "ceiling_bytes must be > 0".into(),
        }),
    }

    // Rule 3: current_bytes must be <= ceiling_bytes.
    if let (Some(cur), Some(ceil)) = (obs.get("current_bytes").and_then(Value::as_u64), ceiling) {
        if cur > ceil {
            out.push(LintFinding {
                rule: "QUOTA-003".into(),
                severity: "error",
                message: format!("current_bytes={cur} exceeds ceiling_bytes={ceil}"),
            });
        }
    }

    // Rule 4: per-tenant usage must not exceed per-tenant quota.
    if let Some(tenants) = obs.get("tenants").and_then(Value::as_array) {
        for t in tenants {
            let id = t.get("id").and_then(Value::as_str).unwrap_or("?");
            let used = t.get("bytes_used").and_then(Value::as_u64).unwrap_or(0);
            let quota = t.get("bytes_quota").and_then(Value::as_u64).unwrap_or(0);
            if used > quota {
                out.push(LintFinding {
                    rule: "QUOTA-004".into(),
                    severity: "error",
                    message: format!("tenant {id} used={used} > quota={quota}"),
                });
            }
        }
    }

    // Rule 5: atomic_writes must equal atomic_commits (no half-completed transactions).
    let writes = obs.get("atomic_writes").and_then(Value::as_u64);
    let commits = obs.get("atomic_commits").and_then(Value::as_u64);
    if let (Some(w), Some(c)) = (writes, commits) {
        if w != c {
            out.push(LintFinding {
                rule: "QUOTA-005".into(),
                severity: "error",
                message: format!("atomic_writes={w} != atomic_commits={c} (torn write)"),
            });
        }
    }

    out
}

pub fn build_happy_observation() -> Value {
    json!({
        "schema_version": 1,
        "ceiling_bytes": 50_000_000_000u64, // 50 GB cap
        "current_bytes": 12_400_000_000u64,
        "atomic_writes": 8421,
        "atomic_commits": 8421,
        "tenants": [
            { "id": "team-alpha", "bytes_used":  6_000_000_000u64, "bytes_quota": 10_000_000_000u64 },
            { "id": "team-bravo", "bytes_used":  4_400_000_000u64, "bytes_quota":  8_000_000_000u64 },
            { "id": "team-eval",  "bytes_used":  2_000_000_000u64, "bytes_quota":  4_000_000_000u64 }
        ]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("registry_quota_lint_happy")?;
    let obs = build_happy_observation();

    let obs_path = ctx.path("quota_observation.json");
    std::fs::write(
        &obs_path,
        serde_json::to_vec_pretty(&obs).map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_quota_observation(&obs);
    let errors = findings.iter().filter(|f| f.severity == "error").count();

    println!("=== Recipe: {} ===", ctx.name());
    println!("Observation: {}", obs_path.display());
    println!("Findings: {errors} errors");
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
    fn happy_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_observation_has_no_errors() {
        let f = lint_quota_observation(&build_happy_observation());
        assert!(f.is_empty(), "expected clean: {f:?}");
    }

    #[test]
    fn rejects_zero_ceiling() {
        let mut obs = build_happy_observation();
        obs["ceiling_bytes"] = json!(0);
        let f = lint_quota_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "QUOTA-002"));
    }

    #[test]
    fn flags_global_overage() {
        let mut obs = build_happy_observation();
        obs["current_bytes"] = json!(60_000_000_000u64);
        let f = lint_quota_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "QUOTA-003"));
    }
}
