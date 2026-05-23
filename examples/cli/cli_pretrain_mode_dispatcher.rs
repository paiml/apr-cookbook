//! # apr pretrain — `--mode` Dispatcher (finetune / from-scratch)
//!
//! `apr pretrain --mode <MODE>` switches between MODEL-1 (finetune from
//! existing checkpoint) and MODEL-2 (from-scratch cold start). MODEL-2
//! requires `--vocab-init` because there's no source vocab to inherit.
//! This recipe builds the validator and asserts the contract.
//!
//! Demonstrates the **PRETRAIN.5** recipe for PMAT-104 (apr pretrain coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SHIP-TWO-001 + MODEL-1/MODEL-2 distinction
//!
//! Run with: cargo run --example cli_pretrain_mode_dispatcher
//!
//! Added by PMAT-104 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Finetune,    // MODEL-1: warm start from existing checkpoint
    FromScratch, // MODEL-2: cold start, no source weights
}

impl Mode {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "finetune" => Some(Mode::Finetune),
            "from-scratch" => Some(Mode::FromScratch),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Ok(Mode),
    UnknownMode(String),
    FinetuneRequiresSource,
    FromScratchRequiresVocabInit,
}

pub fn dispatch(mode: &str, has_source_checkpoint: bool, has_vocab_init: bool) -> DispatchVerdict {
    let Some(m) = Mode::from_str_strict(mode) else {
        return DispatchVerdict::UnknownMode(mode.into());
    };
    match m {
        Mode::Finetune if !has_source_checkpoint => DispatchVerdict::FinetuneRequiresSource,
        Mode::FromScratch if !has_vocab_init => DispatchVerdict::FromScratchRequiresVocabInit,
        _ => DispatchVerdict::Ok(m),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_pretrain_mode_dispatcher")?;

    let cases = [
        ("happy finetune", "finetune", true, false),
        ("happy from-scratch", "from-scratch", false, true),
        ("finetune no source", "finetune", false, false),
        ("from-scratch no vocab", "from-scratch", false, false),
        ("typo", "scratch", false, false),
    ];
    for (label, mode, src, vocab) in cases {
        println!("{label:>22}  →  {:?}", dispatch(mode, src, vocab));
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
    fn finetune_with_source_passes() {
        assert_eq!(
            dispatch("finetune", true, false),
            DispatchVerdict::Ok(Mode::Finetune)
        );
    }

    #[test]
    fn from_scratch_with_vocab_passes() {
        assert_eq!(
            dispatch("from-scratch", false, true),
            DispatchVerdict::Ok(Mode::FromScratch)
        );
    }

    #[test]
    fn finetune_without_source_rejected() {
        assert_eq!(
            dispatch("finetune", false, false),
            DispatchVerdict::FinetuneRequiresSource
        );
    }

    #[test]
    fn from_scratch_without_vocab_rejected() {
        assert_eq!(
            dispatch("from-scratch", false, false),
            DispatchVerdict::FromScratchRequiresVocabInit
        );
    }

    #[test]
    fn unknown_mode_rejected() {
        assert!(matches!(
            dispatch("scratch", false, false),
            DispatchVerdict::UnknownMode(_)
        ));
    }

    #[test]
    fn known_modes_round_trip() {
        for m in ["finetune", "from-scratch"] {
            assert!(Mode::from_str_strict(m).is_some());
        }
    }

    #[test]
    fn finetune_with_source_and_vocab_still_passes() {
        // Source checkpoint present + vocab also present is fine for finetune.
        assert_eq!(
            dispatch("finetune", true, true),
            DispatchVerdict::Ok(Mode::Finetune)
        );
    }
}
