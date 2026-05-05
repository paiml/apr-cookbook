//! # Recipe: Typical-P Lint — min_keep Floor Enforcement
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr typical-p-lint --observation-file observation.json` (min_keep path)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the `min_keep` floor rule. When typical-p truncates the
//! distribution down to the locally-typical set, it can produce an empty
//! set if `typical_p` is small enough to discard every candidate. The
//! `min_keep` floor is the safety net: the sampled set must contain at
//! least `min_keep` tokens regardless of the typical-p threshold. The lint
//! flags two related defects: (a) sampled_set has fewer than min_keep
//! tokens, (b) min_keep > vocab_size (impossible to satisfy).
//!
//! ## Run Command
//! ```bash
//! cargo run --example typical_p_lint_min_keep_floor
//! ```
//!
//! ## References
//! - Meister, C. et al. (2023). *Locally Typical Sampling*. arXiv:2202.00666
//! - llama.cpp typical-p implementation (min_keep parameter).
//!
//! Added by PMAT-091 (expand-cookbooks followup — Ollama/sampling/imatrix lint).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MinKeepFinding {
    BelowFloor { kept: usize, min_keep: usize },
    ImpossibleFloor { min_keep: usize, vocab: usize },
}

pub fn check_min_keep(obs: &Value) -> Vec<MinKeepFinding> {
    let mut out = Vec::new();
    let min_keep = obs.get("min_keep").and_then(Value::as_u64).unwrap_or(0) as usize;
    let vocab = obs.get("vocab_size").and_then(Value::as_u64).unwrap_or(0) as usize;
    let kept = obs
        .get("sampled_set")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);

    if vocab > 0 && min_keep > vocab {
        out.push(MinKeepFinding::ImpossibleFloor { min_keep, vocab });
    }
    if kept < min_keep {
        out.push(MinKeepFinding::BelowFloor { kept, min_keep });
    }
    out
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("typical_p_lint_min_keep_floor")?;

    let happy = json!({
        "min_keep": 1,
        "vocab_size": 32_000,
        "sampled_set": [{ "token_id": 17, "prob": 0.42 }]
    });
    let below_floor = json!({
        "min_keep": 5,
        "vocab_size": 32_000,
        "sampled_set": [{ "token_id": 17, "prob": 1.0 }]
    });
    let impossible = json!({
        "min_keep": 50_000,
        "vocab_size": 32_000,
        "sampled_set": []
    });

    println!("=== Recipe: {} ===", ctx.name());
    println!("happy:       {:?}", check_min_keep(&happy));
    println!("below floor: {:?}", check_min_keep(&below_floor));
    println!("impossible:  {:?}", check_min_keep(&impossible));

    ctx.record_string_metric("verdict", "matrix_printed");
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_keep_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_observation_has_no_findings() {
        let obs = json!({
            "min_keep": 1,
            "vocab_size": 32_000,
            "sampled_set": [{ "token_id": 17, "prob": 0.42 }]
        });
        assert!(check_min_keep(&obs).is_empty());
    }

    #[test]
    fn detects_below_floor() {
        let obs = json!({
            "min_keep": 5,
            "vocab_size": 32_000,
            "sampled_set": [{ "token_id": 17, "prob": 1.0 }]
        });
        let f = check_min_keep(&obs);
        assert_eq!(f.len(), 1);
        assert!(matches!(
            f[0],
            MinKeepFinding::BelowFloor {
                kept: 1,
                min_keep: 5
            }
        ));
    }

    #[test]
    fn detects_impossible_floor_above_vocab() {
        let obs = json!({
            "min_keep": 50_000,
            "vocab_size": 32_000,
            "sampled_set": []
        });
        let f = check_min_keep(&obs);
        // Impossible AND below-floor — both reported separately.
        assert!(f
            .iter()
            .any(|x| matches!(x, MinKeepFinding::ImpossibleFloor { .. })));
    }

    #[test]
    fn vocab_zero_skips_impossibility_check() {
        // If vocab_size is missing/0, can't determine impossibility — only
        // floor check runs. Avoids false positives on observations that
        // omit vocab metadata.
        let obs = json!({
            "min_keep": 1,
            "sampled_set": [{ "token_id": 17, "prob": 1.0 }]
        });
        assert!(check_min_keep(&obs).is_empty());
    }

    #[test]
    fn equal_to_floor_passes() {
        // sampled_set.len() == min_keep is exactly satisfied — must not flag.
        let obs = json!({
            "min_keep": 2,
            "sampled_set": [
                { "token_id": 1, "prob": 0.5 },
                { "token_id": 2, "prob": 0.5 }
            ]
        });
        assert!(check_min_keep(&obs).is_empty());
    }
}
