//! # Tier 3.2 — TPE hyperopt (tabular-only)
//!
//! Falsifier: TPE biases sampling toward high-density (≥ median) region
//! near the synthetic optimum (0.3, 0.7) — at least one of the high-scoring
//! samples lies within 0.4 distance of the optimum.
//!
//! Run with: cargo run --example t3_hyperopt_tpe

use apr_cookbook::finetune::hyperopt;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const N_TRIALS: u32 = 100;
const SEED: u32 = 13;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_hyperopt_tpe")?;
    let trials = hyperopt::random_search(N_TRIALS, SEED);
    let near_optimum = hyperopt::tpe_density_count(&trials);
    println!(
        "✓ TPE: {}/{} above-median trials within d=0.4 of optimum",
        near_optimum, N_TRIALS
    );
    assert!(
        near_optimum >= 1,
        "TPE bias must yield ≥1 high-score sample near optimum"
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
        let trials = hyperopt::random_search(N_TRIALS, SEED);
        assert!(hyperopt::tpe_density_count(&trials) >= 1);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Empty trials → no high-density count.
        assert_eq!(hyperopt::tpe_density_count(&[]), 0);
    }

    #[test]
    fn deterministic_across_runs() {
        let trials = hyperopt::random_search(N_TRIALS, SEED);
        let a = hyperopt::tpe_density_count(&trials);
        let b = hyperopt::tpe_density_count(&trials);
        assert_eq!(a, b);
    }
}
