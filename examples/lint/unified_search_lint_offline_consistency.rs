//! # Recipe: Unified-Search Lint — Offline-Mode Consistency
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr unified-search-lint --observation-file observation.json` (offline path)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the offline-mode consistency rule. When the search was run
//! with `--offline`, the observation must satisfy two invariants:
//!  1. `hub_results_count == 0` (no hub round-trip happened)
//!  2. every result has `source == "local"`
//!
//! A non-empty hub_results_count with `offline=true` indicates the request
//! actually contacted the hub — a Sovereign AI compliance violation
//! (Section 9 of the spec).
//!
//! ## Run Command
//! ```bash
//! cargo run --example unified_search_lint_offline_consistency
//! ```
//!
//! ## References
//! - aprender CRUX-A-23 (offline guarantee).
//! - APR Sovereign AI Spec §9 (network egress).
//!
//! Added by PMAT-092 (expand-cookbooks followup — embeddings/search/grad-norm lint).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OfflineFinding {
    HubResultsLeaked { count: u64 },
    HubSourcedResult { index: usize },
}

pub fn audit_offline(obs: &Value) -> Vec<OfflineFinding> {
    let mut out = Vec::new();
    let offline = obs.get("offline").and_then(Value::as_bool) == Some(true);
    if !offline {
        return out;
    }

    let hub_count = obs
        .get("hub_results_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if hub_count > 0 {
        out.push(OfflineFinding::HubResultsLeaked { count: hub_count });
    }

    if let Some(arr) = obs.get("results").and_then(Value::as_array) {
        for (i, r) in arr.iter().enumerate() {
            if r.get("source").and_then(Value::as_str) == Some("hub") {
                out.push(OfflineFinding::HubSourcedResult { index: i });
            }
        }
    }
    out
}

fn build_clean_offline_observation() -> Value {
    json!({
        "offline": true,
        "query": "qwen-coder",
        "hub_results_count": 0,
        "local_results_count": 2,
        "results": [
            { "model_id": "qwen-coder-7b-q4km",  "source": "local" },
            { "model_id": "qwen-coder-7b-q5km",  "source": "local" }
        ]
    })
}

fn build_leaky_offline_observation() -> Value {
    json!({
        "offline": true,
        "query": "qwen-coder",
        "hub_results_count": 5,        // ⚠ leak
        "local_results_count": 2,
        "results": [
            { "model_id": "qwen-coder-7b-q4km", "source": "local" },
            { "model_id": "Qwen/Qwen2.5-7B",     "source": "hub"   } // ⚠ leak
        ]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("unified_search_lint_offline_consistency")?;
    println!("=== Recipe: {} ===", ctx.name());
    println!(
        "clean offline:  {:?}",
        audit_offline(&build_clean_offline_observation())
    );
    println!(
        "leaky offline:  {:?}",
        audit_offline(&build_leaky_offline_observation())
    );

    ctx.record_string_metric("verdict", "matrix_printed");
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn clean_offline_observation_has_no_findings() {
        let f = audit_offline(&build_clean_offline_observation());
        assert!(f.is_empty(), "expected clean: {f:?}");
    }

    #[test]
    fn hub_results_count_leak_flagged() {
        let f = audit_offline(&build_leaky_offline_observation());
        assert!(f
            .iter()
            .any(|x| matches!(x, OfflineFinding::HubResultsLeaked { count: 5 })));
    }

    #[test]
    fn hub_sourced_result_flagged() {
        let f = audit_offline(&build_leaky_offline_observation());
        assert!(f
            .iter()
            .any(|x| matches!(x, OfflineFinding::HubSourcedResult { index: 1 })));
    }

    #[test]
    fn online_observation_skipped() {
        // When offline=false, the rule is vacuous — caller's responsibility.
        let mut obs = build_leaky_offline_observation();
        obs["offline"] = json!(false);
        assert!(audit_offline(&obs).is_empty());
    }

    #[test]
    fn missing_offline_field_treated_as_online() {
        // Conservative default: only enforce the rule when offline=true is
        // explicitly set — avoids false positives on legacy observations
        // that omit the field.
        let mut obs = build_leaky_offline_observation();
        obs.as_object_mut().unwrap().remove("offline");
        assert!(audit_offline(&obs).is_empty());
    }
}
