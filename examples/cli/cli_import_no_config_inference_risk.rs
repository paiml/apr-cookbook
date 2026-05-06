//! # apr import — `--allow-no-config` Inference Risk Report
//!
//! `apr import <SOURCE> --allow-no-config` permits import when
//! `config.json` is missing. Without it, hyperparameters like `rope_theta`
//! are inferred from tensor shapes and may be wrong (GH-223). This recipe
//! builds the inference-risk report: which hyperparameters CAN be
//! inferred reliably from shapes, and which CANNOT (must come from
//! config.json).
//!
//! Demonstrates the **IMPORT.6** recipe for PMAT-099 (apr import coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GH-223 + transformer architectural conventions
//!
//! Run with: cargo run --example cli_import_no_config_inference_risk
//!
//! Added by PMAT-099 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceConfidence {
    Reliable,  // Can be derived deterministically from tensor shapes
    Heuristic, // Best-guess based on common conventions
    Unknown,   // Operator MUST supply via config.json
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HyperparamRisk {
    pub name: &'static str,
    pub confidence: InferenceConfidence,
    pub note: &'static str,
}

pub fn risk_report() -> Vec<HyperparamRisk> {
    vec![
        HyperparamRisk {
            name: "hidden_size",
            confidence: InferenceConfidence::Reliable,
            note: "second dim of any *_proj.weight matrix",
        },
        HyperparamRisk {
            name: "n_layers",
            confidence: InferenceConfidence::Reliable,
            note: "max layer index across model.layers.{n}.* names",
        },
        HyperparamRisk {
            name: "vocab_size",
            confidence: InferenceConfidence::Reliable,
            note: "first dim of embed_tokens or lm_head",
        },
        HyperparamRisk {
            name: "n_heads",
            confidence: InferenceConfidence::Heuristic,
            note: "hidden_size / typical head_dim (64 or 128) — guesses for non-standard models",
        },
        HyperparamRisk {
            name: "n_kv_heads",
            confidence: InferenceConfidence::Heuristic,
            note: "k_proj first dim / head_dim — wrong for MoE shared KV",
        },
        HyperparamRisk {
            name: "rope_theta",
            confidence: InferenceConfidence::Unknown,
            note: "default 10000.0 will be wrong for long-context (1M+) models",
        },
        HyperparamRisk {
            name: "max_position_embeddings",
            confidence: InferenceConfidence::Unknown,
            note: "no shape encodes the trained context window",
        },
        HyperparamRisk {
            name: "tie_word_embeddings",
            confidence: InferenceConfidence::Unknown,
            note: "must check whether lm_head.weight aliases embed_tokens.weight",
        },
        HyperparamRisk {
            name: "rms_norm_eps",
            confidence: InferenceConfidence::Unknown,
            note: "tiny scalar that affects layer-norm output bit-exactly",
        },
    ]
}

pub fn count_by_confidence(report: &[HyperparamRisk], conf: InferenceConfidence) -> usize {
    report.iter().filter(|r| r.confidence == conf).count()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_import_no_config_inference_risk")?;
    let report = risk_report();
    println!("=== Hyperparameter Inference Risk (no config.json) ===");
    for r in &report {
        println!("  {:?}  {}  // {}", r.confidence, r.name, r.note);
    }
    println!(
        "\nReliable: {}, Heuristic: {}, Unknown: {}",
        count_by_confidence(&report, InferenceConfidence::Reliable),
        count_by_confidence(&report, InferenceConfidence::Heuristic),
        count_by_confidence(&report, InferenceConfidence::Unknown)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn risk_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn risk_report_includes_critical_hyperparams() {
        let report = risk_report();
        let names: Vec<&str> = report.iter().map(|r| r.name).collect();
        for required in [
            "hidden_size",
            "n_layers",
            "vocab_size",
            "rope_theta",
            "tie_word_embeddings",
        ] {
            assert!(
                names.contains(&required),
                "missing critical hyperparam: {required}"
            );
        }
    }

    #[test]
    fn rope_theta_is_unknown_class() {
        let report = risk_report();
        let rope = report.iter().find(|r| r.name == "rope_theta").unwrap();
        assert_eq!(rope.confidence, InferenceConfidence::Unknown);
    }

    #[test]
    fn shape_derivable_params_are_reliable() {
        // hidden_size, n_layers, vocab_size are all directly readable from
        // tensor shapes — no guess required.
        let report = risk_report();
        for n in ["hidden_size", "n_layers", "vocab_size"] {
            let r = report.iter().find(|r| r.name == n).unwrap();
            assert_eq!(r.confidence, InferenceConfidence::Reliable);
        }
    }

    #[test]
    fn every_unknown_param_has_actionable_note() {
        // The note must explain WHY it's unknown so the operator knows what
        // to fix in config.json.
        let report = risk_report();
        for r in &report {
            if r.confidence == InferenceConfidence::Unknown {
                assert!(!r.note.is_empty(), "missing note: {r:?}");
            }
        }
    }

    #[test]
    fn count_by_confidence_sums_to_total() {
        let report = risk_report();
        let total = count_by_confidence(&report, InferenceConfidence::Reliable)
            + count_by_confidence(&report, InferenceConfidence::Heuristic)
            + count_by_confidence(&report, InferenceConfidence::Unknown);
        assert_eq!(total, report.len());
    }
}
