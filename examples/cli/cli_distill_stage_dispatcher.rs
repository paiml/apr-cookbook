//! # apr distill --stage — Two-Stage KD Dispatcher
//!
//! `apr distill --stage <S>` runs one of three sub-stages of the
//! ALB-011 two-stage distillation: `precompute` (cache teacher logits),
//! `train` (logit KD against cached logits), `generate` (text-based KD,
//! GH-455). Stages must run in order; running `train` before `precompute`
//! is a hard error.
//!
//! Demonstrates the **DISTILL.11** recipe for PMAT-106 (apr distill coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender ALB-011 + GH-455
//!
//! Run with: cargo run --example cli_distill_stage_dispatcher
//!
//! Added by PMAT-106 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    Precompute,
    Train,
    Generate,
}

impl Stage {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "precompute" => Some(Stage::Precompute),
            "train" => Some(Stage::Train),
            "generate" => Some(Stage::Generate),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum StageVerdict {
    Ok,
    UnknownStage(String),
    PrerequisitesUnmet { missing: Vec<Stage> },
}

pub fn dispatch(stage: &str, completed: &[Stage]) -> StageVerdict {
    let Some(s) = Stage::from_str_strict(stage) else {
        return StageVerdict::UnknownStage(stage.into());
    };
    let prereqs: Vec<Stage> = match s {
        Stage::Precompute => vec![],
        Stage::Train => vec![Stage::Precompute],
        Stage::Generate => vec![Stage::Precompute, Stage::Train],
    };
    let missing: Vec<Stage> = prereqs
        .into_iter()
        .filter(|p| !completed.contains(p))
        .collect();
    if missing.is_empty() {
        StageVerdict::Ok
    } else {
        StageVerdict::PrerequisitesUnmet { missing }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_distill_stage_dispatcher")?;

    let cases: &[(&str, &[Stage])] = &[
        ("precompute fresh", &[]),
        ("train fresh", &[]),
        ("train after precompute", &[Stage::Precompute]),
        ("generate fresh", &[]),
        ("generate after both", &[Stage::Precompute, Stage::Train]),
    ];
    for (label, completed) in cases {
        let stage = match completed {
            &[] => "precompute",
            _ => "train",
        };
        // Iterate over each stage to show full dispatch matrix.
        for s in ["precompute", "train", "generate"] {
            println!(
                "{label:>22}  →  --stage {s:<10}  {:?}",
                dispatch(s, completed)
            );
        }
        let _ = stage;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn precompute_has_no_prerequisites() {
        assert_eq!(dispatch("precompute", &[]), StageVerdict::Ok);
    }

    #[test]
    fn train_requires_precompute() {
        match dispatch("train", &[]) {
            StageVerdict::PrerequisitesUnmet { missing } => {
                assert_eq!(missing, vec![Stage::Precompute]);
            }
            v => panic!("expected PrerequisitesUnmet, got {v:?}"),
        }
    }

    #[test]
    fn train_after_precompute_passes() {
        assert_eq!(dispatch("train", &[Stage::Precompute]), StageVerdict::Ok);
    }

    #[test]
    fn generate_requires_both_prior_stages() {
        let v = dispatch("generate", &[]);
        if let StageVerdict::PrerequisitesUnmet { missing } = v {
            assert_eq!(missing, vec![Stage::Precompute, Stage::Train]);
        } else {
            panic!("expected PrerequisitesUnmet");
        }
    }

    #[test]
    fn generate_after_train_only_still_needs_precompute() {
        // Pathological — precompute somehow skipped.
        let v = dispatch("generate", &[Stage::Train]);
        if let StageVerdict::PrerequisitesUnmet { missing } = v {
            assert_eq!(missing, vec![Stage::Precompute]);
        } else {
            panic!("expected PrerequisitesUnmet");
        }
    }

    #[test]
    fn generate_after_full_chain_passes() {
        assert_eq!(
            dispatch("generate", &[Stage::Precompute, Stage::Train]),
            StageVerdict::Ok
        );
    }

    #[test]
    fn unknown_stage_rejected() {
        assert!(matches!(
            dispatch("unknown", &[]),
            StageVerdict::UnknownStage(_)
        ));
    }
}
