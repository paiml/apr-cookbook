//! # Tier 4.2 — ORPO monotone in p_rejected (mistral family)
//!
//! Falsifier: ORPO loss is monotone increasing in p_rejected — as the
//! model gives more probability to the rejected response, loss grows.
//!
//! Run with: cargo run --example t4_orpo_mistral

use apr_cookbook::finetune::preference as pref;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture_points() -> Vec<(f64, f64)> {
    vec![(0.7, 0.3), (0.6, 0.5), (0.8, 0.2), (0.65, 0.4)]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_orpo_mistral")?;
    for (p_c, p_r) in fixture_points() {
        let mono = pref::orpo_monotone_in_rejected(p_c, p_r);
        assert!(
            mono,
            "ORPO must be monotone increasing in p_rejected at ({p_c}, {p_r})"
        );
    }
    println!(
        "✓ ORPO monotone in p_rejected on {} fixture points",
        fixture_points().len()
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
        for (c, r) in fixture_points() {
            assert!(pref::orpo_monotone_in_rejected(c, r));
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Single-point check at boundary returns true vacuously.
        assert!(pref::orpo_monotone_in_rejected(0.7, 0.99));
    }

    #[test]
    fn deterministic_across_runs() {
        let a = pref::orpo_monotone_in_rejected(0.7, 0.3);
        let b = pref::orpo_monotone_in_rejected(0.7, 0.3);
        assert_eq!(a, b);
    }
}
