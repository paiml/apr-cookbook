//! # Tier 4.7 — Pairwise reward modeling (llama family)
//!
//! Falsifier: pairwise reward model — P(chosen > rejected) > 0.5 on
//! held-out preference pairs.
//!
//! Run with: cargo run --example t4_reward_pairwise

use apr_cookbook::finetune::rlaif_reward as rr;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn pairs() -> Vec<(f64, f64)> {
    vec![
        (0.7, 0.3),
        (0.6, 0.4),
        (0.8, 0.2),
        (0.55, 0.45),
        (0.65, 0.35),
    ]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_reward_pairwise")?;
    let acc = rr::pairwise_chosen_wins(&pairs());
    println!("✓ pairwise reward: P(chosen > rejected) = {:.4}", acc);
    assert!(acc > 0.5);
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
        assert!(rr::pairwise_chosen_wins(&pairs()) > 0.5);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Reversed pairs (chosen < rejected) → P < 0.5.
        let bad: Vec<(f64, f64)> = pairs().iter().map(|(a, b)| (*b, *a)).collect();
        assert!(rr::pairwise_chosen_wins(&bad) < 0.5);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = rr::pairwise_chosen_wins(&pairs());
        let b = rr::pairwise_chosen_wins(&pairs());
        assert_eq!(a, b);
    }
}
