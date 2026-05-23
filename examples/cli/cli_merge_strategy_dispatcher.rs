//! # apr merge — `--strategy` Dispatcher + Required-Args Matrix
//!
//! `apr merge --strategy <S>` accepts {average, weighted, slerp, ties,
//! dare}. Each requires different supporting args: `weighted` needs
//! `--weights`, `ties`/`dare` need `--base-model`, `slerp` needs exactly
//! 2 input models. This recipe builds the required-args matrix.
//!
//! Demonstrates the **MERGE.8** recipe for PMAT-105 (apr merge coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MERGE-001 + Wortsman 2022 (model soups) + Yadav 2023 (TIES) + Yu 2024 (DARE)
//!
//! Run with: cargo run --example cli_merge_strategy_dispatcher
//!
//! Added by PMAT-105 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Average,
    Weighted,
    Slerp,
    Ties,
    Dare,
}

impl Strategy {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "average" => Some(Strategy::Average),
            "weighted" => Some(Strategy::Weighted),
            "slerp" => Some(Strategy::Slerp),
            "ties" => Some(Strategy::Ties),
            "dare" => Some(Strategy::Dare),
            _ => None,
        }
    }

    pub fn requires_weights(self) -> bool {
        matches!(self, Strategy::Weighted)
    }

    pub fn requires_base_model(self) -> bool {
        matches!(self, Strategy::Ties | Strategy::Dare)
    }

    pub fn requires_exactly_two_inputs(self) -> bool {
        matches!(self, Strategy::Slerp)
    }
}

#[derive(Debug, PartialEq)]
pub enum MergeVerdict {
    Ok,
    UnknownStrategy(String),
    MissingWeights,
    MissingBaseModel,
    SlerpRequiresExactlyTwoInputs { observed: usize },
    NeedAtLeastTwoInputs,
}

pub fn validate(
    strategy: &str,
    n_inputs: usize,
    has_weights: bool,
    has_base: bool,
) -> MergeVerdict {
    let Some(s) = Strategy::from_str_strict(strategy) else {
        return MergeVerdict::UnknownStrategy(strategy.into());
    };
    if n_inputs < 2 {
        return MergeVerdict::NeedAtLeastTwoInputs;
    }
    if s.requires_exactly_two_inputs() && n_inputs != 2 {
        return MergeVerdict::SlerpRequiresExactlyTwoInputs { observed: n_inputs };
    }
    if s.requires_weights() && !has_weights {
        return MergeVerdict::MissingWeights;
    }
    if s.requires_base_model() && !has_base {
        return MergeVerdict::MissingBaseModel;
    }
    MergeVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_merge_strategy_dispatcher")?;

    let cases = [
        ("average ok", "average", 3, false, false),
        ("weighted no weights", "weighted", 3, false, false),
        ("ties no base", "ties", 3, false, false),
        ("slerp 3 inputs", "slerp", 3, false, false),
        ("slerp 2 inputs", "slerp", 2, false, false),
        ("only 1 input", "average", 1, false, false),
        ("typo", "averge", 3, false, false),
    ];

    for (label, s, n, w, b) in cases {
        println!("{label:>22}  →  {:?}", validate(s, n, w, b));
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
    fn average_with_two_inputs_passes() {
        assert_eq!(validate("average", 2, false, false), MergeVerdict::Ok);
    }

    #[test]
    fn single_input_rejected_for_all_strategies() {
        for s in ["average", "weighted", "slerp", "ties", "dare"] {
            let v = validate(s, 1, true, true);
            assert_eq!(v, MergeVerdict::NeedAtLeastTwoInputs);
        }
    }

    #[test]
    fn weighted_without_weights_rejected() {
        assert_eq!(
            validate("weighted", 3, false, false),
            MergeVerdict::MissingWeights
        );
    }

    #[test]
    fn ties_without_base_rejected() {
        assert_eq!(
            validate("ties", 3, false, false),
            MergeVerdict::MissingBaseModel
        );
    }

    #[test]
    fn dare_without_base_rejected() {
        assert_eq!(
            validate("dare", 3, false, false),
            MergeVerdict::MissingBaseModel
        );
    }

    #[test]
    fn slerp_with_three_inputs_rejected() {
        // Slerp = spherical linear interpolation between EXACTLY 2 vectors.
        assert_eq!(
            validate("slerp", 3, false, false),
            MergeVerdict::SlerpRequiresExactlyTwoInputs { observed: 3 }
        );
    }

    #[test]
    fn slerp_with_two_inputs_passes() {
        assert_eq!(validate("slerp", 2, false, false), MergeVerdict::Ok);
    }

    #[test]
    fn unknown_strategy_rejected() {
        assert!(matches!(
            validate("averge", 3, false, false),
            MergeVerdict::UnknownStrategy(_)
        ));
    }
}
