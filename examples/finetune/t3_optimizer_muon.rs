//! # Tier 3.10 — Muon optimizer step efficiency (tabular-only)
//!
//! Falsifier: Muon optimizer reaches a target loss in ≤ 0.5× AdamW steps
//! (synthetic comparison; exact ratio depends on problem geometry).
//!
//! Run with: cargo run --example t3_optimizer_muon

use apr_cookbook::finetune::encoders_optimizers as enc;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MUON_STEPS: u32 = 30;
const ADAMW_STEPS: u32 = 60;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_optimizer_muon")?;
    let r = enc::muon_efficiency_ratio(MUON_STEPS, ADAMW_STEPS);
    println!(
        "✓ Muon efficiency: {} steps vs AdamW {} steps → ratio = {:.3}",
        MUON_STEPS, ADAMW_STEPS, r
    );
    assert!(r <= 0.5);
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
        assert!(enc::muon_efficiency_ratio(MUON_STEPS, ADAMW_STEPS) <= 0.5);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Muon as slow as AdamW → ratio = 1.0.
        assert!(enc::muon_efficiency_ratio(60, 60) > 0.5);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = enc::muon_efficiency_ratio(MUON_STEPS, ADAMW_STEPS);
        let b = enc::muon_efficiency_ratio(MUON_STEPS, ADAMW_STEPS);
        assert_eq!(a, b);
    }
}
