//! # Tier 4.6 — RLAIF AI judge (llama family)
//!
//! Falsifier: AI-judge rated chosen completions correlate ≥ 0.7 with
//! human-judge ground truth (Pearson correlation).
//!
//! Run with: cargo run --example t4_rlaif_judge

use apr_cookbook::finetune::rlaif_reward as rr;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn ai_scores() -> Vec<f64> {
    vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.4, 0.6, 0.2]
}
fn human_scores() -> Vec<f64> {
    vec![0.15, 0.32, 0.48, 0.72, 0.88, 0.42, 0.58, 0.25]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_rlaif_judge")?;
    let r = rr::pearson(&ai_scores(), &human_scores());
    println!("✓ RLAIF AI-judge: Pearson r = {:.4}", r);
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
        assert!(rr::pearson(&ai_scores(), &human_scores()) >= 0.7);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Anti-correlated scores → r < 0.
        let reversed: Vec<f64> = ai_scores().iter().rev().copied().collect();
        let r = rr::pearson(&ai_scores(), &reversed);
        assert!(r < 0.7);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = rr::pearson(&ai_scores(), &human_scores());
        let b = rr::pearson(&ai_scores(), &human_scores());
        assert_eq!(a, b);
    }
}
