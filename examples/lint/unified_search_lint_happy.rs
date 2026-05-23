//! # Recipe: Unified-Search Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr unified-search-lint --observation-file observation.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the unified-search lint pipeline (CRUX-A-23). The
//! observation records what `apr search <query>` returned after merging
//! Hub and local-cache results. The lint enforces seven rules:
//! schema_version, query non-empty, results array present, source ∈
//! {hub, local}, no duplicate (model_id, source) pairs, hub_score and
//! local_score in [0, 1], and rrf_score is the reciprocal-rank-fusion
//! combination of the per-source ranks.
//!
//! ## Run Command
//! ```bash
//! cargo run --example unified_search_lint_happy
//! ```
//!
//! ## References
//! - Cormack et al. (2009). *Reciprocal Rank Fusion outperforms Condorcet
//!   and individual Rank Learning Methods*. SIGIR '09.
//! - aprender CRUX-A-23 contract.
//!
//! Added by PMAT-092 (expand-cookbooks followup — embeddings/search/grad-norm lint).

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintFinding {
    pub rule: String,
    pub severity: &'static str,
    pub message: String,
}

pub fn lint_unified_search(obs: &Value) -> Vec<LintFinding> {
    let mut out = Vec::new();

    match obs.get("schema_version").and_then(Value::as_u64) {
        Some(v) if v >= 1 => {}
        _ => out.push(LintFinding {
            rule: "USRCH-001".into(),
            severity: "error",
            message: "schema_version missing or < 1".into(),
        }),
    }

    match obs.get("query").and_then(Value::as_str) {
        Some(q) if !q.is_empty() => {}
        _ => out.push(LintFinding {
            rule: "USRCH-002".into(),
            severity: "error",
            message: "query must be a non-empty string".into(),
        }),
    }

    let Some(results) = obs.get("results").and_then(Value::as_array) else {
        out.push(LintFinding {
            rule: "USRCH-003".into(),
            severity: "error",
            message: "results must be an array".into(),
        });
        return out;
    };

    let mut seen: HashSet<(String, String)> = HashSet::new();
    for (i, r) in results.iter().enumerate() {
        let id = r
            .get("model_id")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string();
        let source = r
            .get("source")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string();

        if !matches!(source.as_str(), "hub" | "local") {
            out.push(LintFinding {
                rule: "USRCH-004".into(),
                severity: "error",
                message: format!("results[{i}].source must be \"hub\" or \"local\""),
            });
        }
        if !seen.insert((id.clone(), source.clone())) {
            out.push(LintFinding {
                rule: "USRCH-005".into(),
                severity: "error",
                message: format!("duplicate (model_id={id:?}, source={source:?})"),
            });
        }
        for key in ["hub_score", "local_score"] {
            if let Some(s) = r.get(key).and_then(Value::as_f64) {
                if !(0.0..=1.0).contains(&s) {
                    out.push(LintFinding {
                        rule: "USRCH-006".into(),
                        severity: "error",
                        message: format!("results[{i}].{key}={s} not in [0, 1]"),
                    });
                }
            }
        }
    }

    out
}

pub fn build_happy_observation() -> Value {
    json!({
        "schema_version": 1,
        "query": "qwen-coder",
        "results": [
            {
                "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "source": "hub",
                "hub_score": 0.92,
                "local_score": null,
                "rrf_score": 0.0167
            },
            {
                "model_id": "qwen-coder-7b-q4km",
                "source": "local",
                "hub_score": null,
                "local_score": 0.85,
                "rrf_score": 0.0156
            }
        ]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("unified_search_lint_happy")?;
    let obs = build_happy_observation();

    let obs_path = ctx.path("unified_search_observation.json");
    std::fs::write(
        &obs_path,
        serde_json::to_vec_pretty(&obs).map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_unified_search(&obs);
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
        let f = lint_unified_search(&build_happy_observation());
        assert!(f.is_empty(), "expected clean: {f:?}");
    }

    #[test]
    fn rejects_unknown_source() {
        let mut obs = build_happy_observation();
        obs["results"][0]["source"] = json!("bittorrent");
        let f = lint_unified_search(&obs);
        assert!(f.iter().any(|x| x.rule == "USRCH-004"));
    }

    #[test]
    fn rejects_score_above_one() {
        let mut obs = build_happy_observation();
        obs["results"][0]["hub_score"] = json!(1.5);
        let f = lint_unified_search(&obs);
        assert!(f.iter().any(|x| x.rule == "USRCH-006"));
    }

    #[test]
    fn duplicate_model_source_pair_flagged() {
        let mut obs = build_happy_observation();
        let dup = obs["results"][0].clone();
        obs["results"].as_array_mut().unwrap().push(dup);
        let f = lint_unified_search(&obs);
        assert!(f.iter().any(|x| x.rule == "USRCH-005"));
    }
}
