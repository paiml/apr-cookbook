//! # Tier 3.2 — Hyperband hyperopt (tabular-only)
//!
//! Falsifier: Hyperband with R=81, η=3 produces exactly 5 brackets
//! (s_max + 1 where s_max = ⌊log_3(81)⌋ = 4).
//!
//! Run with: cargo run --example t3_hyperopt_hyperband

use apr_cookbook::finetune::hyperopt;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const R_MAX: u32 = 81;
const ETA: u32 = 3;
const EXPECTED_BRACKETS: u32 = 5;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_hyperopt_hyperband")?;
    let n = hyperopt::hyperband_brackets(R_MAX, ETA);
    println!("✓ Hyperband R={} η={}: {} brackets", R_MAX, ETA, n);
    assert_eq!(
        n, EXPECTED_BRACKETS,
        "Hyperband(R={R_MAX}, η={ETA}) must yield {EXPECTED_BRACKETS} brackets"
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
        assert_eq!(hyperopt::hyperband_brackets(R_MAX, ETA), EXPECTED_BRACKETS);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // η=4 with R=81 yields s_max = floor(log_4(81)) = 3, so 4 brackets.
        assert_ne!(hyperopt::hyperband_brackets(R_MAX, 4), EXPECTED_BRACKETS);
    }

    #[test]
    fn deterministic_across_runs() {
        assert_eq!(
            hyperopt::hyperband_brackets(R_MAX, ETA),
            hyperopt::hyperband_brackets(R_MAX, ETA)
        );
    }
}
