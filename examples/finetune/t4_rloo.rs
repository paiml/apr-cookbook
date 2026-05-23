//! # Tier 4.8 — RLOO leave-one-out baseline (llama family)
//!
//! Falsifier: RLOO REINFORCE-leave-one-out gradient variance ≤ vanilla
//! REINFORCE variance × (1−1/n)² < 1.
//!
//! Run with: cargo run --example t4_rloo

use apr_cookbook::finetune::online_alt as oa;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn rewards() -> Vec<f64> {
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 4.5, 2.5]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_rloo")?;
    let ratio = oa::rloo_variance_ratio(&rewards());
    println!("✓ RLOO variance ratio = {:.4} (< 1 means reduced)", ratio);
    assert!(ratio < 1.0);
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
        assert!(oa::rloo_variance_ratio(&rewards()) < 1.0);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // n=1 — no leave-one-out possible, ratio = 1.
        let single = vec![1.0];
        assert_eq!(oa::rloo_variance_ratio(&single), 1.0);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = oa::rloo_variance_ratio(&rewards());
        let b = oa::rloo_variance_ratio(&rewards());
        assert_eq!(a, b);
    }
}
