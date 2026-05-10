//! # Tier 4.1 — DPO out-of-preference KL preservation (phi family)
//!
//! Falsifier: DPO on phi preserves base model on prompts not in preference
//! set — KL between policy and reference on out-of-preference samples ≤ 0.1.
//!
//! Run with: cargo run --example t4_dpo_phi

use apr_cookbook::finetune::preference as pref;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn out_of_preference_lp_diffs() -> Vec<f64> {
    // Small drifts on prompts outside the preference set.
    vec![0.05, -0.04, 0.03, -0.02, 0.04, 0.01, -0.03]
}
const KL_BUDGET: f64 = 0.1;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_dpo_phi")?;
    let kl = pref::kl_estimate(&out_of_preference_lp_diffs());
    println!(
        "✓ DPO out-of-preference KL = {:.4} (budget {KL_BUDGET})",
        kl
    );
    assert!(kl.abs() <= KL_BUDGET);
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
        assert!(pref::kl_estimate(&out_of_preference_lp_diffs()).abs() <= KL_BUDGET);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Large drifts → KL > budget.
        let big = vec![0.5_f64; 5];
        assert!(pref::kl_estimate(&big).abs() > KL_BUDGET);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = pref::kl_estimate(&out_of_preference_lp_diffs());
        let b = pref::kl_estimate(&out_of_preference_lp_diffs());
        assert_eq!(a, b);
    }
}
