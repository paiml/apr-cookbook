//! # apr distill — `--strategy` Dispatcher (standard/progressive/ensemble)
//!
//! `apr distill --strategy <S>` accepts {standard, progressive, ensemble}.
//! Each requires different supporting structure: `progressive` walks
//! through layer counts, `ensemble` requires multiple teacher checkpoints.
//! This recipe builds the dispatcher and asserts the contract.
//!
//! Demonstrates the **DISTILL.9** recipe for PMAT-106 (apr distill coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender ALB-011 + Hinton et al. (2015) KD + progressive KD
//!
//! Run with: cargo run --example cli_distill_strategy_dispatcher
//!
//! Added by PMAT-106 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Standard,
    Progressive,
    Ensemble,
}

impl Strategy {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "standard" => Some(Strategy::Standard),
            "progressive" => Some(Strategy::Progressive),
            "ensemble" => Some(Strategy::Ensemble),
            _ => None,
        }
    }

    pub fn requires_layer_progression(self) -> bool {
        matches!(self, Strategy::Progressive)
    }

    pub fn requires_multiple_teachers(self) -> bool {
        matches!(self, Strategy::Ensemble)
    }
}

#[derive(Debug, PartialEq)]
pub enum DistillVerdict {
    Ok,
    UnknownStrategy(String),
    ProgressiveNeedsLayerProgression,
    EnsembleNeedsMultipleTeachers { observed: usize },
}

pub fn validate(strategy: &str, n_teachers: usize, has_layer_progression: bool) -> DistillVerdict {
    let Some(s) = Strategy::from_str_strict(strategy) else {
        return DistillVerdict::UnknownStrategy(strategy.into());
    };
    if s.requires_layer_progression() && !has_layer_progression {
        return DistillVerdict::ProgressiveNeedsLayerProgression;
    }
    if s.requires_multiple_teachers() && n_teachers < 2 {
        return DistillVerdict::EnsembleNeedsMultipleTeachers {
            observed: n_teachers,
        };
    }
    DistillVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_distill_strategy_dispatcher")?;

    let cases = [
        ("standard ok", "standard", 1, false),
        ("progressive no layers", "progressive", 1, false),
        ("progressive with layers", "progressive", 1, true),
        ("ensemble single teacher", "ensemble", 1, false),
        ("ensemble multi", "ensemble", 3, false),
        ("typo", "standardd", 1, false),
    ];

    for (label, s, n, layers) in cases {
        println!("{label:>22}  →  {:?}", validate(s, n, layers));
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
    fn standard_strategy_passes_with_one_teacher() {
        assert_eq!(validate("standard", 1, false), DistillVerdict::Ok);
    }

    #[test]
    fn progressive_without_layer_progression_rejected() {
        assert_eq!(
            validate("progressive", 1, false),
            DistillVerdict::ProgressiveNeedsLayerProgression
        );
    }

    #[test]
    fn progressive_with_layer_progression_passes() {
        assert_eq!(validate("progressive", 1, true), DistillVerdict::Ok);
    }

    #[test]
    fn ensemble_with_single_teacher_rejected() {
        // Ensemble needs ≥2 teachers — single teacher = pointless.
        assert_eq!(
            validate("ensemble", 1, false),
            DistillVerdict::EnsembleNeedsMultipleTeachers { observed: 1 }
        );
    }

    #[test]
    fn ensemble_with_multiple_teachers_passes() {
        assert_eq!(validate("ensemble", 3, false), DistillVerdict::Ok);
    }

    #[test]
    fn unknown_strategy_rejected() {
        assert!(matches!(
            validate("standardd", 1, false),
            DistillVerdict::UnknownStrategy(_)
        ));
    }

    #[test]
    fn known_strategies_round_trip() {
        for s in ["standard", "progressive", "ensemble"] {
            assert!(Strategy::from_str_strict(s).is_some());
        }
    }
}
