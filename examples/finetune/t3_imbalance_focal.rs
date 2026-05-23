//! # Tier 3.4 — Focal loss for imbalance (tabular-only)
//!
//! Falsifier: focal loss with γ=2 down-weights easy-positive (p=0.9)
//! contribution by ≥ 50%. Closed-form: factor = (1−p)^γ = 0.01.
//!
//! Run with: cargo run --example t3_imbalance_focal

use apr_cookbook::finetune::imbalance as imb;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const GAMMA: f64 = 2.0;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_imbalance_focal")?;
    let easy = imb::focal_factor(0.9, GAMMA);
    let hard = imb::focal_factor(0.5, GAMMA);
    println!(
        "✓ focal γ={}: easy_p=0.9 → factor={:.4}, hard_p=0.5 → factor={:.4}",
        GAMMA, easy, hard
    );
    assert!(easy < 0.5, "easy positive must be down-weighted by ≥50%");
    assert!(hard > easy, "hard positive must keep more weight than easy");
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
        assert!(imb::focal_factor(0.9, GAMMA) < 0.5);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // γ=0 → factor = 1.0 (no down-weighting).
        assert_eq!(imb::focal_factor(0.9, 0.0), 1.0);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = imb::focal_factor(0.9, GAMMA);
        let b = imb::focal_factor(0.9, GAMMA);
        assert_eq!(a, b);
    }
}
