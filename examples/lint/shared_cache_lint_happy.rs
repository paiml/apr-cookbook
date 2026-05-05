//! # Recipe: APR_MODELS Shared-Cache Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr shared-cache-lint --observation-file observation.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the shared-cache lint pipeline (CRUX-A-21). The
//! `$APR_MODELS` directory is multi-user / multi-process so the
//! observation must record (a) per-blob dedup status, (b) POSIX mode bits,
//! (c) ownership group, and (d) effective umask. The lint enforces five
//! rules: schema_version, dedup ratio plausibility, mode is 0o644 or
//! stricter, group is the configured `apr` group, and umask is 0o022 or
//! stricter.
//!
//! ## Run Command
//! ```bash
//! cargo run --example shared_cache_lint_happy
//! ```
//!
//! ## References
//! - aprender CRUX-A-21 contract (shared-cache observation).
//! - POSIX 1003.1-2017 §3.252 (file mode bits).
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

pub fn lint_shared_cache_observation(obs: &Value) -> Vec<LintFinding> {
    let mut out = Vec::new();

    // Rule 1: schema_version present and >= 1.
    match obs.get("schema_version").and_then(Value::as_u64) {
        Some(v) if v >= 1 => {}
        _ => out.push(LintFinding {
            rule: "SHCACHE-001".into(),
            severity: "error",
            message: "schema_version missing or < 1".into(),
        }),
    }

    // Rule 2: dedup_ratio in (1.0, 100.0]. Below 1.0 means dedup made cache *bigger*.
    match obs.get("dedup_ratio").and_then(Value::as_f64) {
        Some(r) if r.is_finite() && (1.0..=100.0).contains(&r) => {}
        _ => out.push(LintFinding {
            rule: "SHCACHE-002".into(),
            severity: "error",
            message: "dedup_ratio must be finite, in [1.0, 100.0]".into(),
        }),
    }

    // Rule 3: mode bits must be 0o644 (or stricter — i.e., world cannot write).
    match obs.get("mode_octal").and_then(Value::as_u64) {
        Some(m) if (m & 0o022) == 0 => {}
        _ => out.push(LintFinding {
            rule: "SHCACHE-003".into(),
            severity: "error",
            message: "mode bits allow group/world write — strip 022".into(),
        }),
    }

    // Rule 4: group must be the configured shared group ("apr" or "apr-models").
    match obs.get("group").and_then(Value::as_str) {
        Some("apr" | "apr-models") => {}
        _ => out.push(LintFinding {
            rule: "SHCACHE-004".into(),
            severity: "error",
            message: "group must be \"apr\" or \"apr-models\"".into(),
        }),
    }

    // Rule 5: umask must be 0o022 or stricter (no group/world write inheritance).
    match obs.get("umask_octal").and_then(Value::as_u64) {
        Some(u) if (u & 0o022) == 0o022 => {}
        _ => out.push(LintFinding {
            rule: "SHCACHE-005".into(),
            severity: "error",
            message: "umask must be 0o022 or stricter (must include group+world write bits)".into(),
        }),
    }

    out
}

pub fn build_happy_observation() -> Value {
    json!({
        "schema_version": 1,
        "cache_path": "/var/lib/apr/models",
        "dedup_ratio": 3.4,
        "mode_octal": 0o644,
        "group": "apr-models",
        "umask_octal": 0o022,
        "blobs_total": 1024,
        "blobs_unique": 301
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("shared_cache_lint_happy")?;
    let obs = build_happy_observation();

    let obs_path = ctx.path("shared_cache_observation.json");
    std::fs::write(
        &obs_path,
        serde_json::to_vec_pretty(&obs).map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_shared_cache_observation(&obs);
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
        let f = lint_shared_cache_observation(&build_happy_observation());
        assert!(f.is_empty(), "expected clean: {f:?}");
    }

    #[test]
    fn rejects_world_writable_mode() {
        let mut obs = build_happy_observation();
        obs["mode_octal"] = json!(0o646);
        let f = lint_shared_cache_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "SHCACHE-003"));
    }

    #[test]
    fn rejects_wrong_group() {
        let mut obs = build_happy_observation();
        obs["group"] = json!("nobody");
        let f = lint_shared_cache_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "SHCACHE-004"));
    }

    #[test]
    fn rejects_loose_umask() {
        let mut obs = build_happy_observation();
        obs["umask_octal"] = json!(0o002);
        let f = lint_shared_cache_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "SHCACHE-005"));
    }
}
