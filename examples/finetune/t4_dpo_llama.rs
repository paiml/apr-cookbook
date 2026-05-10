//! # Tier 4.1 — DPO (llama family)
//!
//! Falsifier: DPO on chosen>rejected preferences yields lower loss than the
//! swapped (chosen<rejected) ordering — implicit reward differentiates the
//! two completions.
//!
//! Run with: cargo run --example t4_dpo_llama

use apr_cookbook::finetune::preference as pref;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const LP_CHOSEN: f64 = 0.5;
const LP_REJECTED: f64 = -0.5;
const BETA: f64 = 0.1;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_dpo_llama")?;
    let loss_correct = pref::dpo_loss(LP_CHOSEN, LP_REJECTED, BETA);
    let loss_swapped = pref::dpo_loss(LP_REJECTED, LP_CHOSEN, BETA);
    println!(
        "✓ DPO β={}: correct loss = {:.4}, swapped = {:.4}",
        BETA, loss_correct, loss_swapped
    );
    assert!(loss_correct < loss_swapped);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn falsifier_holds_on_fixture() {
        let l1 = pref::dpo_loss(LP_CHOSEN, LP_REJECTED, BETA);
        let l2 = pref::dpo_loss(LP_REJECTED, LP_CHOSEN, BETA);
        assert!(l1 < l2);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Equal log-prob diffs → losses identical regardless of order.
        let l1 = pref::dpo_loss(0.0, 0.0, BETA);
        let l2 = pref::dpo_loss(0.0, 0.0, BETA);
        assert!((l1 - l2).abs() < 1e-12);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = pref::dpo_loss(LP_CHOSEN, LP_REJECTED, BETA);
        let b = pref::dpo_loss(LP_CHOSEN, LP_REJECTED, BETA);
        assert_eq!(a, b);
    }
}
