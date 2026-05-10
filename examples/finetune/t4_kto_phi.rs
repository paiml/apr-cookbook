//! # Tier 4.3 — KTO desirability=0.5 unbiased (phi family)
//!
//! Falsifier: KTO with desirability=0.5 produces unbiased gradient over
//! balanced positive/negative feedback — pos_loss + neg_loss = constant 0.5.
//!
//! Run with: cargo run --example t4_kto_phi

use apr_cookbook::finetune::preference as pref;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const BETA: f64 = 0.1;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_kto_phi")?;
    for &lp_diff in &[0.0_f64, 0.5, 1.0, -0.3] {
        let pos = pref::kto_loss(lp_diff, BETA, true, 0.5);
        let neg = pref::kto_loss(lp_diff, BETA, false, 0.5);
        let sum = pos + neg;
        assert!(
            (sum - 0.5).abs() < 1e-12,
            "KTO sum at lp_diff={lp_diff} should be 0.5, got {sum}"
        );
    }
    println!("✓ KTO desirability=0.5: pos_loss + neg_loss = 0.5 (unbiased)");
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
        let pos = pref::kto_loss(0.5, BETA, true, 0.5);
        let neg = pref::kto_loss(0.5, BETA, false, 0.5);
        assert!((pos + neg - 0.5).abs() < 1e-12);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Asymmetric desirability + extreme lp_diff: sum diverges from 0.5.
        // sum = 0.7 - 0.4σ when desirability=0.7; at lp_diff → -∞ → sum→0.7.
        let pos = pref::kto_loss(-100.0, BETA, true, 0.7);
        let neg = pref::kto_loss(-100.0, BETA, false, 0.7);
        assert!((pos + neg - 0.5).abs() > 0.01);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = pref::kto_loss(0.5, BETA, true, 0.5);
        let b = pref::kto_loss(0.5, BETA, true, 0.5);
        assert_eq!(a, b);
    }
}
