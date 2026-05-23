//! # Tier 3.2 — Random hyperopt (tabular-only)
//!
//! Falsifier: Random search produces exactly N trials and is deterministic
//! for a fixed seed.
//!
//! Run with: cargo run --example t3_hyperopt_random

use apr_cookbook::finetune::hyperopt;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const N_TRIALS: u32 = 30;
const SEED: u32 = 7;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_hyperopt_random")?;
    let trials = hyperopt::random_search(N_TRIALS, SEED);
    let best = hyperopt::best_score(&trials);
    println!(
        "✓ random: {} trials, best score = {:.4} (seed={})",
        trials.len(),
        best,
        SEED
    );
    assert_eq!(trials.len() as u32, N_TRIALS);
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
        let r1 = hyperopt::random_search(N_TRIALS, SEED);
        let r2 = hyperopt::random_search(N_TRIALS, SEED);
        assert_eq!(r1, r2);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        let r1 = hyperopt::random_search(N_TRIALS, SEED);
        let r2 = hyperopt::random_search(N_TRIALS, SEED + 1);
        assert_ne!(r1, r2);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = hyperopt::random_search(N_TRIALS, SEED);
        let b = hyperopt::random_search(N_TRIALS, SEED);
        assert_eq!(a, b);
    }
}
