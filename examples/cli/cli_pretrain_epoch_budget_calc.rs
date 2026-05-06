//! # apr pretrain --epochs — Token Budget Calculator
//!
//! Pretrain budget = epochs × dataset_tokens; under Chinchilla
//! scaling, budget should be ~20× param_count for compute-optimal.
//! This recipe builds the calculator + Chinchilla-ratio reporter.
//!
//! Demonstrates the **PRE.5** recipe for PMAT-117 (apr pretrain coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PRE-001 + Hoffmann et al. 2022 (Chinchilla)
//!
//! Run with: cargo run --example cli_pretrain_epoch_budget_calc
//!
//! Added by PMAT-117 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const CHINCHILLA_RATIO: u64 = 20; // ~20 tokens / param

#[derive(Debug, PartialEq)]
pub enum BudgetTier {
    UnderTrained { ratio: f64 },
    NearOptimal,
    OverTrained { ratio: f64 },
    Invalid,
}

pub fn total_tokens(epochs: u32, dataset_tokens: u64) -> Option<u64> {
    if epochs == 0 || dataset_tokens == 0 {
        return None;
    }
    u64::from(epochs).checked_mul(dataset_tokens)
}

pub fn classify(epochs: u32, dataset_tokens: u64, params: u64) -> BudgetTier {
    if params == 0 {
        return BudgetTier::Invalid;
    }
    let Some(total) = total_tokens(epochs, dataset_tokens) else {
        return BudgetTier::Invalid;
    };
    let ratio = total as f64 / params as f64;
    if ratio < CHINCHILLA_RATIO as f64 / 2.0 {
        BudgetTier::UnderTrained { ratio }
    } else if ratio > CHINCHILLA_RATIO as f64 * 2.0 {
        BudgetTier::OverTrained { ratio }
    } else {
        BudgetTier::NearOptimal
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_pretrain_epoch_budget_calc")?;

    // Llama-3-8B: 15T training tokens, ~8B params → ratio ≈ 1875× (over-trained intentionally)
    let cases = [
        (1u32, 140_000_000_000u64, 7_000_000_000u64), // 20× ratio
        (1, 70_000_000_000, 7_000_000_000),           // under
        (5, 1_000_000_000_000, 7_000_000_000),        // over
    ];
    for (e, d, p) in cases {
        println!(
            "epochs={e} ds={d} p={p}  →  total={:?}  tier={:?}",
            total_tokens(e, d),
            classify(e, d, p)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calc_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn total_tokens_multiplies_correctly() {
        assert_eq!(total_tokens(3, 1_000_000), Some(3_000_000));
    }

    #[test]
    fn zero_inputs_yield_none() {
        assert!(total_tokens(0, 1_000_000).is_none());
        assert!(total_tokens(3, 0).is_none());
    }

    #[test]
    fn at_chinchilla_optimal_near_optimal() {
        // 20× ratio is exactly Chinchilla.
        let v = classify(1, 140_000_000_000, 7_000_000_000);
        assert_eq!(v, BudgetTier::NearOptimal);
    }

    #[test]
    fn under_10x_under_trained() {
        // 7B with 50B tokens → 7.14× (under).
        let v = classify(1, 50_000_000_000, 7_000_000_000);
        assert!(matches!(v, BudgetTier::UnderTrained { .. }));
    }

    #[test]
    fn over_40x_over_trained() {
        // 7B with 5T tokens → 714× (Llama-3-style intentional over-train).
        let v = classify(1, 5_000_000_000_000, 7_000_000_000);
        assert!(matches!(v, BudgetTier::OverTrained { .. }));
    }

    #[test]
    fn zero_params_invalid() {
        assert_eq!(classify(1, 1_000_000, 0), BudgetTier::Invalid);
    }

    #[test]
    fn zero_epochs_invalid() {
        assert_eq!(classify(0, 1_000_000, 1_000_000), BudgetTier::Invalid);
    }

    #[test]
    fn boundary_at_10x_under() {
        // ratio = 10 (= CHINCHILLA/2): strictly < → under.
        let v = classify(1, 9_000_000_000, 1_000_000_000);
        assert!(matches!(v, BudgetTier::UnderTrained { .. }));
    }

    #[test]
    fn boundary_at_40x_over() {
        // ratio > 40 → over.
        let v = classify(1, 41_000_000_000, 1_000_000_000);
        assert!(matches!(v, BudgetTier::OverTrained { .. }));
    }
}
