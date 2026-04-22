//! # Recipe: Oracle Ensemble Vote Over 5 Predictors
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr oracle model.apr --ensemble --predictors 5 --voting majority`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example oracle_ensemble_vote` exits 0
//! 2. [x] `cargo test --example oracle_ensemble_vote` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr oracle --ensemble` in-process (no shell-out)
//! 10. [x] Unit tests cover majority, plurality, agreement rate
//!
//! ## Learning Objective
//! Demonstrates ensemble classification using five independent heuristic
//! predictors. Each predictor emits a label; the oracle aggregates via majority
//! rule with plurality fallback and reports agreement-rate as a confidence
//! proxy. Matches the ensemble-vote semantics the real `apr oracle --ensemble`
//! exposes on multi-signal classifiers.
//!
//! ## Run Command
//! ```bash
//! cargo run --example oracle_ensemble_vote
//! ```
//!
//! ## References
//! - Dietterich, T. G. (2000). *Ensemble Methods in Machine Learning*. Multiple Classifier Systems. DOI: 10.1007/3-540-45014-9_1

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Prediction {
    pub predictor: String,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VoteResult {
    pub winner: String,
    pub vote_count: usize,
    pub total: usize,
    pub agreement_rate: f64,
    pub runner_up: Option<(String, usize)>,
}

/// Aggregate predictions by majority/plurality with stable tie-break on label order.
pub fn majority_vote(preds: &[Prediction]) -> Option<VoteResult> {
    if preds.is_empty() {
        return None;
    }
    let mut tally: HashMap<String, usize> = HashMap::new();
    for p in preds {
        *tally.entry(p.label.clone()).or_insert(0) += 1;
    }
    let mut tallies: Vec<(String, usize)> = tally.into_iter().collect();
    tallies.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let (winner, count) = tallies[0].clone();
    let total = preds.len();
    let runner_up = tallies.get(1).cloned();
    Some(VoteResult {
        winner,
        vote_count: count,
        total,
        agreement_rate: count as f64 / total as f64,
        runner_up,
    })
}

/// Simulate five heuristic predictors for a synthetic model.
fn run_predictors(evidence: &[&str]) -> Vec<Prediction> {
    vec![
        Prediction {
            predictor: "name-pattern".into(),
            label: pick_label(evidence, &["q_proj", "gate_proj"], "llama")
                .unwrap_or_else(|| "unknown".to_string()),
        },
        Prediction {
            predictor: "shape-ratio".into(),
            label: pick_label(evidence, &["4096", "11008"], "llama")
                .unwrap_or_else(|| "gpt".into()),
        },
        Prediction {
            predictor: "config-file".into(),
            label: pick_label(evidence, &["llama_config"], "llama")
                .unwrap_or_else(|| "bert".into()),
        },
        Prediction {
            predictor: "embed-dim".into(),
            label: pick_label(evidence, &["4096"], "llama").unwrap_or_else(|| "gpt".into()),
        },
        Prediction {
            predictor: "norm-style".into(),
            label: pick_label(evidence, &["rmsnorm"], "llama").unwrap_or_else(|| "bert".into()),
        },
    ]
}

fn pick_label(evidence: &[&str], needles: &[&str], label: &str) -> Option<String> {
    if evidence
        .iter()
        .any(|e| needles.iter().any(|n| e.contains(n)))
    {
        Some(label.to_string())
    } else {
        None
    }
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("oracle_ensemble_vote")?;
    println!("=== Recipe: {} ===", ctx.name());

    let evidence = vec![
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "rmsnorm.weight",
        "llama_config.json",
        "embed_dim=4096",
        "ffn_dim=11008",
    ];

    let preds = run_predictors(&evidence);
    for p in &preds {
        println!("  [{}] -> {}", p.predictor, p.label);
    }

    let Some(result) = majority_vote(&preds) else {
        println!("No predictions available.");
        ctx.report()?;
        return Ok(());
    };
    println!(
        "Winner: {} ({}/{} votes, agreement {:.2}%)",
        result.winner,
        result.vote_count,
        result.total,
        result.agreement_rate * 100.0,
    );
    if let Some((r, v)) = &result.runner_up {
        println!("Runner-up: {} ({} votes)", r, v);
    }

    let report = json!({
        "recipe": ctx.name(),
        "predictors": preds.iter().map(|p| json!({
            "predictor": p.predictor,
            "label": p.label,
        })).collect::<Vec<_>>(),
        "winner": result.winner,
        "vote_count": result.vote_count,
        "agreement_rate": result.agreement_rate,
    });
    let path = ctx.path("ensemble-vote.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    ctx.record_metric("vote_count", result.vote_count as i64);
    ctx.record_float_metric("agreement_rate", result.agreement_rate);
    ctx.record_string_metric("winner", result.winner);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(who: &str, lbl: &str) -> Prediction {
        Prediction {
            predictor: who.into(),
            label: lbl.into(),
        }
    }

    #[test]
    fn empty_vote_returns_none() {
        assert!(majority_vote(&[]).is_none());
    }

    #[test]
    fn unanimous_vote_is_full_agreement() {
        let preds = vec![p("a", "llama"), p("b", "llama"), p("c", "llama")];
        let r = majority_vote(&preds).expect("non-empty");
        assert_eq!(r.winner, "llama");
        assert_eq!(r.vote_count, 3);
        assert!((r.agreement_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn majority_rule_picks_largest() {
        let preds = vec![
            p("a", "llama"),
            p("b", "llama"),
            p("c", "llama"),
            p("d", "gpt"),
            p("e", "bert"),
        ];
        let r = majority_vote(&preds).expect("non-empty");
        assert_eq!(r.winner, "llama");
        assert_eq!(r.vote_count, 3);
    }

    #[test]
    fn tie_breaks_alphabetically() {
        let preds = vec![p("a", "gpt"), p("b", "llama")];
        let r = majority_vote(&preds).expect("non-empty");
        assert_eq!(r.winner, "gpt"); // "gpt" < "llama"
        assert_eq!(r.vote_count, 1);
    }

    #[test]
    fn runner_up_reported() {
        let preds = vec![p("a", "llama"), p("b", "llama"), p("c", "gpt")];
        let r = majority_vote(&preds).expect("non-empty");
        assert!(r.runner_up.is_some());
        if let Some((lbl, cnt)) = r.runner_up {
            assert_eq!(lbl, "gpt");
            assert_eq!(cnt, 1);
        }
    }
}
