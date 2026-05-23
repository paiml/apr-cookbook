//! # Tier 4.3 — KTO binary feedback (llama family)
//!
//! Falsifier: KTO on positive (helpful) samples reduces loss as log-prob
//! ratio increases, and rises for negative samples — gradient is correctly
//! signed for binary feedback.
//!
//! Run with: cargo run --example t4_kto_llama

use apr_cookbook::finetune::preference as pref;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const BETA: f64 = 0.1;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_kto_llama")?;
    let pos_low = pref::kto_loss(0.0, BETA, true, 0.5);
    let pos_high = pref::kto_loss(1.0, BETA, true, 0.5);
    println!(
        "✓ KTO positive: lp_diff=0 → {:.4}, lp_diff=1 → {:.4}",
        pos_low, pos_high
    );
    assert!(
        pos_high < pos_low,
        "positive sample loss must drop as lp_diff grows"
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
        let l0 = pref::kto_loss(0.0, BETA, true, 0.5);
        let l1 = pref::kto_loss(1.0, BETA, true, 0.5);
        assert!(l1 < l0);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Negative sample: loss rises as lp_diff grows.
        let n0 = pref::kto_loss(0.0, BETA, false, 0.5);
        let n1 = pref::kto_loss(1.0, BETA, false, 0.5);
        assert!(n1 > n0);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = pref::kto_loss(0.5, BETA, true, 0.5);
        let b = pref::kto_loss(0.5, BETA, true, 0.5);
        assert_eq!(a, b);
    }
}
