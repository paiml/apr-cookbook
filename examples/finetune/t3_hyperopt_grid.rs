//! # Tier 3.2 — Grid hyperopt (tabular-only)
//!
//! Falsifier: Grid search trial count = ∏|axis| (exact cartesian product).
//!
//! Run with: cargo run --example t3_hyperopt_grid

use apr_cookbook::finetune::hyperopt;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const X_GRID: &[f64] = &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
const Y_GRID: &[f64] = &[0.0, 0.5, 0.7, 0.9];

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_hyperopt_grid")?;
    let trials = hyperopt::grid_search(X_GRID, Y_GRID);
    let best = hyperopt::best_score(&trials);
    println!(
        "✓ grid: {} trials ({}×{}), best score = {:.4}",
        trials.len(),
        X_GRID.len(),
        Y_GRID.len(),
        best
    );
    assert_eq!(trials.len(), X_GRID.len() * Y_GRID.len());
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
        let trials = hyperopt::grid_search(X_GRID, Y_GRID);
        assert_eq!(trials.len(), X_GRID.len() * Y_GRID.len());
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Empty axis collapses product to 0.
        let trials = hyperopt::grid_search(&[], Y_GRID);
        assert_eq!(trials.len(), 0);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = hyperopt::grid_search(X_GRID, Y_GRID);
        let b = hyperopt::grid_search(X_GRID, Y_GRID);
        assert_eq!(a, b);
    }
}
