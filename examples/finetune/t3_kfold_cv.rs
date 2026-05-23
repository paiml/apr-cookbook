//! # Tier 3.5 — k-fold cross-validation (tabular-only)
//!
//! Falsifier: 5-fold CV produces 5 disjoint validation sets covering 100%
//! of training data. Closed-form: union(val_k) = full set, pairwise empty
//! intersections.
//!
//! Run with: cargo run --example t3_kfold_cv

use apr_cookbook::finetune::multimodal as mm;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const N: usize = 50;
const K: u32 = 5;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_kfold_cv")?;
    let folds = mm::kfold_split(N, K);
    println!(
        "✓ k-fold (n={}, K={}): {} folds, each val size {}",
        N,
        K,
        folds.len(),
        folds[0].1.len()
    );
    assert!(
        mm::kfold_is_partition(N, K),
        "K folds must form a disjoint partition"
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
        assert!(mm::kfold_is_partition(N, K));
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // K=0 should not produce a valid partition — n=50 elements.
        // (Implementation returns 0 folds, so union has 0 elements ≠ 50).
        assert!(!mm::kfold_is_partition(N, 0));
    }

    #[test]
    fn deterministic_across_runs() {
        let a = mm::kfold_split(N, K);
        let b = mm::kfold_split(N, K);
        assert_eq!(a, b);
    }
}
