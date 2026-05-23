//! # apr diagnose — Five Whys Chain Builder
//!
//! `apr diagnose <CHECKPOINT_DIR>` runs an automated Five Whys analysis on
//! a training checkpoint that diverged. The output is an ordered chain of
//! ≤5 cause hypotheses, each linking to a downstream symptom. This recipe
//! models the chain-builder as a pure function so a CI pipeline can
//! preview which hypothesis tree would be generated for a given symptom
//! signal.
//!
//! Demonstrates the **DIAGNOSE.3** recipe for PMAT-095 (apr diagnose coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DIAGNOSE-001 + Toyota Production System (Five Whys)
//!
//! Run with: cargo run --example cli_diagnose_five_whys_chain
//!
//! Added by PMAT-095 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symptom {
    LossNan,
    LossPlateau,
    AccuracyZero,
    GradOverflow,
    OomDuringForward,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CauseLink {
    pub depth: u32,
    pub cause: &'static str,
    pub remediation: &'static str,
}

pub fn build_five_whys(symptom: Symptom) -> Vec<CauseLink> {
    match symptom {
        Symptom::LossNan => vec![
            CauseLink {
                depth: 1,
                cause: "loss diverged to NaN",
                remediation: "lower lr or enable gradient clipping",
            },
            CauseLink {
                depth: 2,
                cause: "logits hit ±inf during softmax",
                remediation: "use log_softmax + cross_entropy with logits",
            },
            CauseLink {
                depth: 3,
                cause: "FP16 underflow in attention scores",
                remediation: "switch to bf16 or scale by 1/sqrt(d_k)",
            },
            CauseLink {
                depth: 4,
                cause: "input ids included unembedded vocab id",
                remediation: "validate token ids ≤ vocab_size before forward",
            },
            CauseLink {
                depth: 5,
                cause: "tokenizer.encode produced UNK as -1",
                remediation: "fix tokenizer to map UNK to vocab_size-1, not -1",
            },
        ],
        Symptom::LossPlateau => vec![
            CauseLink {
                depth: 1,
                cause: "loss flat for ≥1000 steps",
                remediation: "increase lr or reduce batch size",
            },
            CauseLink {
                depth: 2,
                cause: "gradient flowing but small",
                remediation: "check layer-norm placement; pre-norm > post-norm",
            },
            CauseLink {
                depth: 3,
                cause: "weight matrices in dead-relu regime",
                remediation: "switch to GELU or SiLU activation",
            },
        ],
        Symptom::AccuracyZero => vec![CauseLink {
            depth: 1,
            cause: "all predictions are class 0",
            remediation: "check class balance; add stratified sampling",
        }],
        Symptom::GradOverflow => vec![
            CauseLink {
                depth: 1,
                cause: "post-clip grad-norm > max_grad_norm",
                remediation: "lower max_grad_norm; verify clip is actually applied",
            },
            CauseLink {
                depth: 2,
                cause: "loss includes unbounded scale factor",
                remediation: "remove unscaled regularizer or normalize before sum",
            },
        ],
        Symptom::OomDuringForward => vec![
            CauseLink {
                depth: 1,
                cause: "VRAM exceeded forward pass",
                remediation: "reduce batch size or enable activation checkpointing",
            },
            CauseLink {
                depth: 2,
                cause: "attention requires O(n²) memory",
                remediation: "switch to FlashAttention or chunked attention",
            },
            CauseLink {
                depth: 3,
                cause: "kv-cache grew unbounded in generation",
                remediation: "set max_new_tokens or enable kv-cache eviction",
            },
        ],
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_diagnose_five_whys_chain")?;

    for sym in [
        Symptom::LossNan,
        Symptom::LossPlateau,
        Symptom::OomDuringForward,
    ] {
        println!("=== {sym:?} ===");
        for link in build_five_whys(sym) {
            println!(
                "  why #{}: {} → {}",
                link.depth, link.cause, link.remediation
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_whys_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn each_chain_has_at_most_five_levels() {
        for sym in [
            Symptom::LossNan,
            Symptom::LossPlateau,
            Symptom::AccuracyZero,
            Symptom::GradOverflow,
            Symptom::OomDuringForward,
        ] {
            let chain = build_five_whys(sym);
            assert!(chain.len() <= 5, "{sym:?} chain too long: {}", chain.len());
        }
    }

    #[test]
    fn each_chain_has_at_least_one_link() {
        // Every symptom must produce at least one hypothesis — silence is
        // worse than a wrong guess for the operator.
        for sym in [
            Symptom::LossNan,
            Symptom::LossPlateau,
            Symptom::AccuracyZero,
            Symptom::GradOverflow,
            Symptom::OomDuringForward,
        ] {
            assert!(!build_five_whys(sym).is_empty());
        }
    }

    #[test]
    fn depths_are_strictly_increasing() {
        for sym in [
            Symptom::LossNan,
            Symptom::LossPlateau,
            Symptom::OomDuringForward,
        ] {
            let chain = build_five_whys(sym);
            for w in chain.windows(2) {
                assert!(w[1].depth > w[0].depth);
            }
        }
    }

    #[test]
    fn every_link_has_actionable_remediation() {
        for sym in [
            Symptom::LossNan,
            Symptom::LossPlateau,
            Symptom::AccuracyZero,
            Symptom::GradOverflow,
            Symptom::OomDuringForward,
        ] {
            for link in build_five_whys(sym) {
                assert!(
                    !link.remediation.is_empty(),
                    "missing remediation: {link:?}"
                );
            }
        }
    }
}
