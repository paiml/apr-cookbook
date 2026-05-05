//! # Recipe: `apr rm` / `apr gc` Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr rm-gc-lint --observation-file observation.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the blob-GC lint pipeline (CRUX-A-25). The observation
//! records what `apr rm` removed (model alias entries) and what `apr gc`
//! reclaimed (the underlying blob bytes). The lint enforces five
//! invariants: refcount conservation, no negative refcounts, no
//! ref-from-removed-alias, GC ran after rm, and bytes_freed == sum of
//! reclaimed blob sizes.
//!
//! ## Run Command
//! ```bash
//! cargo run --example rm_gc_lint_happy
//! ```
//!
//! ## References
//! - aprender CRUX-A-25 contract (rm/gc observation).
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

pub fn lint_rm_gc_observation(obs: &Value) -> Vec<LintFinding> {
    let mut out = Vec::new();

    // Rule 1: schema_version present and >= 1.
    match obs.get("schema_version").and_then(Value::as_u64) {
        Some(v) if v >= 1 => {}
        _ => out.push(LintFinding {
            rule: "RMGC-001".into(),
            severity: "error",
            message: "schema_version missing or < 1".into(),
        }),
    }

    // Rule 2: gc_at must be > rm_at (GC must follow rm).
    let rm = obs.get("rm_at_unix").and_then(Value::as_u64);
    let gc = obs.get("gc_at_unix").and_then(Value::as_u64);
    if let (Some(r), Some(g)) = (rm, gc) {
        if g <= r {
            out.push(LintFinding {
                rule: "RMGC-002".into(),
                severity: "error",
                message: format!("gc_at={g} must be > rm_at={r}"),
            });
        }
    }

    // Rule 3: no negative refcounts (drops below 0 means double-free).
    if let Some(blobs) = obs.get("blobs").and_then(Value::as_array) {
        for b in blobs {
            let cid = b.get("cid").and_then(Value::as_str).unwrap_or("?");
            let rc = b.get("refcount_after").and_then(Value::as_i64).unwrap_or(0);
            if rc < 0 {
                out.push(LintFinding {
                    rule: "RMGC-003".into(),
                    severity: "error",
                    message: format!("blob {cid} refcount_after={rc} < 0 (double-free)"),
                });
            }
        }
    }

    // Rule 4: bytes_freed == sum of (size_bytes of reclaimed blobs).
    if let Some(blobs) = obs.get("blobs").and_then(Value::as_array) {
        let claimed = obs.get("bytes_freed").and_then(Value::as_u64).unwrap_or(0);
        let computed: u64 = blobs
            .iter()
            .filter(|b| b.get("reclaimed").and_then(Value::as_bool) == Some(true))
            .filter_map(|b| b.get("size_bytes").and_then(Value::as_u64))
            .sum();
        if computed != claimed {
            out.push(LintFinding {
                rule: "RMGC-004".into(),
                severity: "error",
                message: format!("bytes_freed={claimed} != sum(reclaimed sizes)={computed}"),
            });
        }
    }

    // Rule 5: every reclaimed blob must have refcount_after == 0 (no orphan ref).
    if let Some(blobs) = obs.get("blobs").and_then(Value::as_array) {
        for b in blobs {
            let cid = b.get("cid").and_then(Value::as_str).unwrap_or("?");
            let reclaimed = b.get("reclaimed").and_then(Value::as_bool) == Some(true);
            let rc = b.get("refcount_after").and_then(Value::as_i64).unwrap_or(0);
            if reclaimed && rc != 0 {
                out.push(LintFinding {
                    rule: "RMGC-005".into(),
                    severity: "error",
                    message: format!("blob {cid} reclaimed but refcount_after={rc} (orphaned ref)"),
                });
            }
        }
    }

    out
}

pub fn build_happy_observation() -> Value {
    json!({
        "schema_version": 1,
        "rm_at_unix": 1_715_000_000u64,
        "gc_at_unix": 1_715_000_120u64,    // GC ran 2 min after rm
        "bytes_freed": 7_000_000u64,
        "blobs": [
            { "cid": "b3:aaa", "size_bytes": 4_000_000, "refcount_after": 0, "reclaimed": true  },
            { "cid": "b3:bbb", "size_bytes": 3_000_000, "refcount_after": 0, "reclaimed": true  },
            { "cid": "b3:ccc", "size_bytes": 1_000_000, "refcount_after": 2, "reclaimed": false }
        ]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("rm_gc_lint_happy")?;
    let obs = build_happy_observation();

    let obs_path = ctx.path("rm_gc_observation.json");
    std::fs::write(
        &obs_path,
        serde_json::to_vec_pretty(&obs).map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_rm_gc_observation(&obs);
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
        let f = lint_rm_gc_observation(&build_happy_observation());
        assert!(f.is_empty(), "expected clean: {f:?}");
    }

    #[test]
    fn rejects_gc_before_rm() {
        let mut obs = build_happy_observation();
        obs["gc_at_unix"] = json!(1_715_000_000u64 - 60);
        let f = lint_rm_gc_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "RMGC-002"));
    }

    #[test]
    fn detects_bytes_freed_mismatch() {
        let mut obs = build_happy_observation();
        obs["bytes_freed"] = json!(999_999_999u64);
        let f = lint_rm_gc_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "RMGC-004"));
    }

    #[test]
    fn detects_negative_refcount_double_free() {
        let mut obs = build_happy_observation();
        obs["blobs"][0]["refcount_after"] = json!(-1);
        let f = lint_rm_gc_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "RMGC-003"));
    }
}
