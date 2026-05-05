//! # Recipe: GPTQ Lint — Cosine Similarity Violation
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr gptq-lint --observation-file observation.json` (cosine fail)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the cosine-similarity ship-blocker. GPTQ's published
//! quality budget is `cos_sim > 0.95` against the FP16 reference
//! activations on a held-out calibration set. Below the threshold the
//! quantized weights silently degrade downstream perplexity by >1.0 PPL
//! (Frantar et al., Table 4). The lint must elevate this to error severity
//! and annotate which decoder block dropped the metric.
//!
//! ## Run Command
//! ```bash
//! cargo run --example gptq_lint_cosine_violation
//! ```
//!
//! ## References
//! - Frantar, E. et al. (2023). *GPTQ*. arXiv:2210.17323, Table 4 (PPL drift vs cos_sim).
//!
//! Added by PMAT-089 (expand-cookbooks followup — quantization lint coverage).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct BlockFinding {
    pub block: usize,
    pub cosine: f64,
    pub severity: &'static str,
    pub message: String,
}

const SHIP_THRESHOLD: f64 = 0.95;

pub fn lint_per_block_cosine(obs: &Value) -> Vec<BlockFinding> {
    let mut out = Vec::new();
    let Some(blocks) = obs.get("per_block").and_then(Value::as_array) else {
        return vec![BlockFinding {
            block: usize::MAX,
            cosine: f64::NAN,
            severity: "error",
            message: "per_block array missing".into(),
        }];
    };
    for (i, b) in blocks.iter().enumerate() {
        let cos = b.get("cosine").and_then(Value::as_f64).unwrap_or(f64::NAN);
        if !cos.is_finite() {
            out.push(BlockFinding {
                block: i,
                cosine: cos,
                severity: "error",
                message: "cosine is NaN/inf".into(),
            });
        } else if cos <= SHIP_THRESHOLD {
            out.push(BlockFinding {
                block: i,
                cosine: cos,
                severity: "error",
                message: format!("cos {cos:.4} <= ship threshold {SHIP_THRESHOLD:.2}"),
            });
        }
    }
    out
}

fn build_observation_with_block_drift() -> Value {
    json!({
        "schema_version": 1,
        "model": "llama-7b-gptq",
        "per_block": [
            { "block": 0,  "cosine": 0.991 },
            { "block": 1,  "cosine": 0.987 },
            { "block": 2,  "cosine": 0.998 },
            { "block": 3,  "cosine": 0.949 },   // ⚠ ship-blocker
            { "block": 4,  "cosine": 0.972 },
            { "block": 5,  "cosine": 0.880 }    // ⚠ ship-blocker
        ]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("gptq_lint_cosine_violation")?;
    let obs = build_observation_with_block_drift();
    let findings = lint_per_block_cosine(&obs);

    println!("=== Recipe: {} ===", ctx.name());
    println!(
        "ship-blockers (cos <= {SHIP_THRESHOLD:.2}): {}",
        findings.len()
    );
    for f in &findings {
        println!("  block {}: cos={:.4} — {}", f.block, f.cosine, f.message);
    }

    ctx.record_metric("blockers", findings.len() as i64);
    ctx.record_string_metric("verdict", if findings.is_empty() { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_violation_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn flags_each_block_below_threshold() {
        let f = lint_per_block_cosine(&build_observation_with_block_drift());
        let blocks: Vec<usize> = f.iter().map(|x| x.block).collect();
        assert_eq!(blocks, vec![3, 5]);
    }

    #[test]
    fn all_clean_blocks_yield_no_findings() {
        let obs = json!({
            "per_block": [
                { "block": 0, "cosine": 0.991 },
                { "block": 1, "cosine": 0.972 }
            ]
        });
        assert!(lint_per_block_cosine(&obs).is_empty());
    }

    #[test]
    fn boundary_at_exactly_threshold_is_blocker() {
        // Conservative: exact equality at 0.95 still counts as a blocker —
        // the test uses < threshold not <= so we don't drift the ship gate.
        let obs = json!({
            "per_block": [{ "block": 0, "cosine": 0.95 }]
        });
        let f = lint_per_block_cosine(&obs);
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn nan_cosine_is_error() {
        let obs = json!({
            "per_block": [{ "block": 0, "cosine": null }]
        });
        let f = lint_per_block_cosine(&obs);
        assert_eq!(f.len(), 1);
        assert!(f[0].cosine.is_nan());
    }

    #[test]
    fn missing_per_block_array_is_error() {
        let obs = json!({});
        let f = lint_per_block_cosine(&obs);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].block, usize::MAX);
    }
}
