//! # Recipe: Typical-P Lint — Entropy Truncation Audit
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr typical-p-lint --observation-file observation.json` (entropy path)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the entropy-truncation audit. Typical-p selects tokens whose
//! per-token surprise `-log p_i` is closest to the conditional entropy
//! `H(X)`. The lint recomputes `H(X)` from the candidate distribution and
//! compares to the producer-claimed `entropy_target`. A drift > 0.02 nats
//! indicates the producer used a different scoring rule (top-p, top-k, or
//! temperature alone).
//!
//! ## Run Command
//! ```bash
//! cargo run --example typical_p_lint_entropy_truncation
//! ```
//!
//! ## References
//! - Meister, C. et al. (2023). *Locally Typical Sampling*. arXiv:2202.00666, §2 (entropy formulation).
//!
//! Added by PMAT-091 (expand-cookbooks followup — Ollama/sampling/imatrix lint).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct EntropyAudit {
    pub claimed: f64,
    pub computed: f64,
    pub drift_nats: f64,
    pub passes: bool,
}

const DRIFT_TOLERANCE_NATS: f64 = 0.02;

pub fn audit_entropy(obs: &Value) -> Option<EntropyAudit> {
    let claimed = obs.get("entropy_target").and_then(Value::as_f64)?;
    let arr = obs.get("candidate_dist").and_then(Value::as_array)?;
    let probs: Vec<f64> = arr.iter().filter_map(Value::as_f64).collect();
    if probs.is_empty() {
        return None;
    }
    let computed: f64 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    let drift = (claimed - computed).abs();
    Some(EntropyAudit {
        claimed,
        computed,
        drift_nats: drift,
        passes: drift <= DRIFT_TOLERANCE_NATS,
    })
}

fn build_consistent_observation() -> Value {
    json!({
        "entropy_target": 1.0397, // -(0.5*ln 0.5 + 0.3*ln 0.3 + 0.2*ln 0.2) ≈ 1.030 nats
        "candidate_dist": [0.5, 0.3, 0.2]
    })
}

fn build_drifted_observation() -> Value {
    json!({
        "entropy_target": 0.30,        // claim much smaller than truth
        "candidate_dist": [0.5, 0.3, 0.2]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("typical_p_lint_entropy_truncation")?;

    for (label, obs) in [
        ("consistent", build_consistent_observation()),
        ("drifted", build_drifted_observation()),
    ] {
        if let Some(a) = audit_entropy(&obs) {
            println!(
                "{label:>11} claimed={:.4}  computed={:.4}  drift={:.4}  pass={}",
                a.claimed, a.computed, a.drift_nats, a.passes
            );
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
    fn entropy_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn consistent_entropy_passes() {
        let a = audit_entropy(&build_consistent_observation()).unwrap();
        assert!(a.passes, "audit: {a:?}");
    }

    #[test]
    fn drifted_entropy_fails() {
        let a = audit_entropy(&build_drifted_observation()).unwrap();
        assert!(!a.passes, "audit: {a:?}");
    }

    #[test]
    fn empty_distribution_returns_none() {
        let obs = json!({ "entropy_target": 1.0, "candidate_dist": [] });
        assert!(audit_entropy(&obs).is_none());
    }

    #[test]
    fn missing_field_returns_none() {
        let obs = json!({ "candidate_dist": [0.5, 0.5] });
        assert!(audit_entropy(&obs).is_none());
    }

    #[test]
    fn delta_distribution_has_zero_entropy() {
        // Single-token "distribution" — entropy is 0; claim of 0 must pass.
        let obs = json!({ "entropy_target": 0.0, "candidate_dist": [1.0] });
        let a = audit_entropy(&obs).unwrap();
        assert!(a.passes);
        assert!(a.computed.abs() < 1e-9);
    }
}
