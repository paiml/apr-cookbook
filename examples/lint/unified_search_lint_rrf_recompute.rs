//! # Recipe: Unified-Search Lint — RRF Score Recompute
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr unified-search-lint --observation-file observation.json` (rrf path)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the reciprocal-rank-fusion (RRF) score recompute. RRF
//! combines two ranked lists: `rrf(d) = Σ 1 / (k + rank_s(d))` where `k`
//! is a smoothing constant (default 60). This recipe takes the producer's
//! claimed `rrf_score` and recomputes it from the (hub_rank, local_rank)
//! pair, asserting the difference is < 1e-6. A drift indicates the
//! producer used a different fusion (Borda count, weighted sum) and
//! mislabelled it.
//!
//! ## Run Command
//! ```bash
//! cargo run --example unified_search_lint_rrf_recompute
//! ```
//!
//! ## References
//! - Cormack et al. (2009). *Reciprocal Rank Fusion outperforms Condorcet
//!   and individual Rank Learning Methods*. SIGIR '09.
//!
//! Added by PMAT-092 (expand-cookbooks followup — embeddings/search/grad-norm lint).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

const RRF_K: f64 = 60.0;
const TOL: f64 = 1e-6;

#[derive(Debug, Clone, PartialEq)]
pub struct RrfFinding {
    pub model_id: String,
    pub claimed: f64,
    pub computed: f64,
    pub abs_drift: f64,
}

pub fn rrf_score(hub_rank: Option<u64>, local_rank: Option<u64>) -> f64 {
    let mut s = 0.0;
    if let Some(r) = hub_rank {
        s += 1.0 / (RRF_K + r as f64);
    }
    if let Some(r) = local_rank {
        s += 1.0 / (RRF_K + r as f64);
    }
    s
}

pub fn audit_rrf(obs: &Value) -> Vec<RrfFinding> {
    let mut out = Vec::new();
    let Some(arr) = obs.get("results").and_then(Value::as_array) else {
        return out;
    };
    for r in arr {
        let id = r
            .get("model_id")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string();
        let Some(claimed) = r.get("rrf_score").and_then(Value::as_f64) else {
            continue;
        };
        let hub = r.get("hub_rank").and_then(Value::as_u64);
        let local = r.get("local_rank").and_then(Value::as_u64);
        let computed = rrf_score(hub, local);
        let drift = (claimed - computed).abs();
        if drift > TOL {
            out.push(RrfFinding {
                model_id: id,
                claimed,
                computed,
                abs_drift: drift,
            });
        }
    }
    out
}

fn build_consistent_observation() -> Value {
    // hub_rank=0 → 1/60 = 0.01667
    // hub_rank=2 + local_rank=4 → 1/62 + 1/64 = 0.01613 + 0.01563 = 0.031755
    json!({
        "results": [
            {
                "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "hub_rank": 0,
                "local_rank": null,
                "rrf_score": 0.01666666666666667
            },
            {
                "model_id": "qwen-coder-7b-q4km",
                "hub_rank": 2,
                "local_rank": 4,
                "rrf_score": 0.031754032258064516
            }
        ]
    })
}

fn build_drifted_observation() -> Value {
    json!({
        "results": [
            {
                "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "hub_rank": 0,
                "local_rank": null,
                "rrf_score": 0.5    // wrong — should be ~0.0167
            }
        ]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("unified_search_lint_rrf_recompute")?;
    println!("=== Recipe: {} ===", ctx.name());

    let consistent = audit_rrf(&build_consistent_observation());
    println!("consistent: {} findings", consistent.len());

    let drifted = audit_rrf(&build_drifted_observation());
    println!("drifted:    {} findings", drifted.len());
    for f in &drifted {
        println!(
            "  {} claimed={:.6}  computed={:.6}  drift={:.6}",
            f.model_id, f.claimed, f.computed, f.abs_drift
        );
    }

    ctx.record_metric("drifted_results", drifted.len() as i64);
    ctx.record_string_metric("verdict", if drifted.is_empty() { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn rrf_score_hub_only() {
        // rank 0 with k=60 → 1/60.
        let s = rrf_score(Some(0), None);
        assert!((s - 1.0 / 60.0).abs() < 1e-12);
    }

    #[test]
    fn rrf_score_both_sources_sums() {
        let s = rrf_score(Some(2), Some(4));
        let want = 1.0 / 62.0 + 1.0 / 64.0;
        assert!((s - want).abs() < 1e-12);
    }

    #[test]
    fn rrf_score_no_ranks_is_zero() {
        // Edge case: a result that appears in neither list ⇒ 0.0 (unreachable
        // in practice but the function must not panic).
        assert_eq!(rrf_score(None, None), 0.0);
    }

    #[test]
    fn consistent_observation_has_no_findings() {
        let f = audit_rrf(&build_consistent_observation());
        assert!(f.is_empty(), "expected clean: {f:?}");
    }

    #[test]
    fn drifted_observation_flagged() {
        let f = audit_rrf(&build_drifted_observation());
        assert_eq!(f.len(), 1);
        assert!(f[0].abs_drift > 0.4);
    }
}
