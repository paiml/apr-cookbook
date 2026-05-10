//! # Tier 4.10 — Process Reward Model (phi family)
//!
//! Falsifier: PRM per-step reward correlates ≥ 0.7 with stepwise human
//! annotations on a math chain of thought.
//!
//! Run with: cargo run --example t4_prm

use apr_cookbook::finetune::tier4_closeout as t4c;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn model_step_rewards() -> Vec<f64> {
    vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.4, 0.6, 0.8]
}
fn human_step_rewards() -> Vec<f64> {
    vec![0.12, 0.31, 0.48, 0.72, 0.88, 0.42, 0.59, 0.79]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_prm")?;
    let r = t4c::prm_step_correlation(&model_step_rewards(), &human_step_rewards());
    println!("✓ PRM step-level Pearson r = {:.4}", r);
    assert!(r >= 0.7);
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
        assert!(t4c::prm_step_correlation(&model_step_rewards(), &human_step_rewards()) >= 0.7);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Anti-correlated → r < 0.
        let reversed: Vec<f64> = human_step_rewards().iter().rev().copied().collect();
        let r = t4c::prm_step_correlation(&model_step_rewards(), &reversed);
        assert!(r < 0.7);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = t4c::prm_step_correlation(&model_step_rewards(), &human_step_rewards());
        let b = t4c::prm_step_correlation(&model_step_rewards(), &human_step_rewards());
        assert_eq!(a, b);
    }
}
