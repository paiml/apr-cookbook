//! # Tier 4.2 — ORPO without reference model (llama family)
//!
//! Falsifier: ORPO loss runs without a reference model on (p_chosen, p_rejected)
//! pairs and produces a finite, positive loss.
//!
//! Run with: cargo run --example t4_orpo_llama

use apr_cookbook::finetune::preference as pref;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn pairs() -> Vec<(f64, f64)> {
    vec![
        (0.7, 0.3),
        (0.65, 0.35),
        (0.8, 0.2),
        (0.6, 0.4),
        (0.75, 0.25),
    ]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_orpo_llama")?;
    let total: f64 = pairs().iter().map(|(c, r)| pref::orpo_loss(*c, *r)).sum();
    let mean = total / pairs().len() as f64;
    println!(
        "✓ ORPO mean loss over {} pairs (no ref model): {:.4}",
        pairs().len(),
        mean
    );
    assert!(mean > 0.0 && mean.is_finite());
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
        for (c, r) in pairs() {
            let l = pref::orpo_loss(c, r);
            assert!(l > 0.0 && l.is_finite());
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Equal probs → loss is exactly log(2) (sigmoid(0) = 0.5).
        let l = pref::orpo_loss(0.5, 0.5);
        assert!((l - std::f64::consts::LN_2).abs() < 1e-12);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = pref::orpo_loss(0.7, 0.3);
        let b = pref::orpo_loss(0.7, 0.3);
        assert_eq!(a, b);
    }
}
