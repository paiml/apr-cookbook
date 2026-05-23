//! # Tier 4.1 — DPO implicit reward (qwen3 family)
//!
//! Falsifier: DPO with implicit reward model — implicit r(chosen) > r(rejected)
//! for every training pair after convergence.
//!
//! Run with: cargo run --example t4_dpo_qwen

use apr_cookbook::finetune::preference as pref;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn pairs() -> Vec<(f64, f64)> {
    // (lp_diff_chosen, lp_diff_rejected) post-DPO.
    vec![
        (0.5, -0.3),
        (0.4, -0.2),
        (0.6, -0.4),
        (0.3, -0.1),
        (0.5, 0.0),
    ]
}
const BETA: f64 = 0.1;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_dpo_qwen")?;
    for (lp_c, lp_r) in pairs() {
        let r_c = pref::dpo_implicit_reward(lp_c, BETA);
        let r_r = pref::dpo_implicit_reward(lp_r, BETA);
        assert!(
            r_c > r_r,
            "implicit reward must rank chosen above rejected: {r_c} <= {r_r}"
        );
    }
    println!(
        "✓ DPO implicit reward: chosen > rejected for all {} pairs",
        pairs().len()
    );
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
        for (lp_c, lp_r) in pairs() {
            assert!(pref::dpo_implicit_reward(lp_c, BETA) > pref::dpo_implicit_reward(lp_r, BETA));
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Pre-training: random init has chosen ≤ rejected on some pairs.
        let bogus = (-0.1, 0.5);
        assert!(
            pref::dpo_implicit_reward(bogus.0, BETA) < pref::dpo_implicit_reward(bogus.1, BETA)
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let a = pref::dpo_implicit_reward(0.5, BETA);
        let b = pref::dpo_implicit_reward(0.5, BETA);
        assert_eq!(a, b);
    }
}
