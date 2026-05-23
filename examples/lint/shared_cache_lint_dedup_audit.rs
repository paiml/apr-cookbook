//! # Recipe: Shared-Cache Lint — Dedup Audit
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr shared-cache-lint --observation-file observation.json` (dedup path)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates dedup-ratio plausibility checks. The shared cache uses
//! content-addressed blobs, so the dedup ratio (`blobs_total / blobs_unique`)
//! must equal the recorded `dedup_ratio` to ±1%. A mismatch indicates the
//! reporter is using a stale snapshot, or content addressing has been
//! disabled but blobs are still being deduped at a different layer (silent
//! corruption risk).
//!
//! ## Run Command
//! ```bash
//! cargo run --example shared_cache_lint_dedup_audit
//! ```
//!
//! ## References
//! - aprender CRUX-A-21 (dedup parity invariant).
//! - rkyv content-addressing layer (blob-as-CID).
//!
//! Added by PMAT-090 (expand-cookbooks followup — registry/cache lint coverage).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct DedupAudit {
    pub claimed_ratio: f64,
    pub computed_ratio: f64,
    pub abs_drift: f64,
    pub passes: bool,
}

const TOLERANCE_PCT: f64 = 0.01; // 1%

pub fn audit_dedup(obs: &Value) -> Option<DedupAudit> {
    let claimed = obs.get("dedup_ratio").and_then(Value::as_f64)?;
    let total = obs.get("blobs_total").and_then(Value::as_u64)?;
    let unique = obs.get("blobs_unique").and_then(Value::as_u64)?;
    if unique == 0 {
        return None;
    }
    let computed = total as f64 / unique as f64;
    let abs_drift = (claimed - computed).abs();
    let passes = abs_drift <= TOLERANCE_PCT * computed;
    Some(DedupAudit {
        claimed_ratio: claimed,
        computed_ratio: computed,
        abs_drift,
        passes,
    })
}

fn build_consistent_observation() -> Value {
    json!({
        "dedup_ratio": 3.4,
        "blobs_total": 1024,
        "blobs_unique": 301        // 1024 / 301 ≈ 3.402
    })
}

fn build_drifted_observation() -> Value {
    json!({
        "dedup_ratio": 3.4,
        "blobs_total": 5000,       // 5000 / 301 ≈ 16.6 — far from claimed 3.4
        "blobs_unique": 301
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("shared_cache_lint_dedup_audit")?;

    for (label, obs) in [
        ("consistent", build_consistent_observation()),
        ("drifted", build_drifted_observation()),
    ] {
        match audit_dedup(&obs) {
            Some(a) => println!(
                "{label:>11}  claimed={:.3}  computed={:.3}  |drift|={:.3}  pass={}",
                a.claimed_ratio, a.computed_ratio, a.abs_drift, a.passes
            ),
            None => println!("{label:>11}  unable to audit (missing fields)"),
        }
    }

    ctx.record_string_metric("verdict", "matrix_printed");
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_audit_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn consistent_ratio_passes() {
        let a = audit_dedup(&build_consistent_observation()).unwrap();
        assert!(a.passes, "audit: {a:?}");
    }

    #[test]
    fn drifted_ratio_fails() {
        let a = audit_dedup(&build_drifted_observation()).unwrap();
        assert!(!a.passes, "audit: {a:?}");
    }

    #[test]
    fn unique_zero_returns_none() {
        // Cannot compute ratio if no unique blobs (cache is empty or stale).
        let obs = json!({ "dedup_ratio": 3.4, "blobs_total": 100, "blobs_unique": 0 });
        assert!(audit_dedup(&obs).is_none());
    }

    #[test]
    fn missing_field_returns_none() {
        // Lint must not silently pass when input is incomplete.
        let obs = json!({ "blobs_total": 100, "blobs_unique": 50 });
        assert!(audit_dedup(&obs).is_none());
    }

    #[test]
    fn within_one_percent_of_drift_passes() {
        // 1024/301 = 3.4019..., a claim of 3.4150 (drift ~0.013, ~0.4%) passes.
        let obs = json!({
            "dedup_ratio": 3.415,
            "blobs_total": 1024,
            "blobs_unique": 301
        });
        let a = audit_dedup(&obs).unwrap();
        assert!(a.passes, "should pass: {a:?}");
    }
}
