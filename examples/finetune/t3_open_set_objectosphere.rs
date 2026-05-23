//! # Tier 3.7 — Objectosphere open-set loss (tabular-only)
//!
//! Falsifier: in-class feature norm > out-class feature norm after training.
//!
//! Run with: cargo run --example t3_open_set_objectosphere

use apr_cookbook::finetune::anomaly_open_uncertainty as aou;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn in_class() -> Vec<Vec<f64>> {
    vec![vec![3.0, 4.0], vec![3.5, 4.5], vec![3.2, 4.1]]
}
fn out_class() -> Vec<Vec<f64>> {
    vec![vec![0.5, 0.5], vec![0.4, 0.4], vec![0.6, 0.6]]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_open_set_objectosphere")?;
    let (in_n, out_n) = aou::objectosphere_norms(&in_class(), &out_class());
    println!("✓ objectosphere: |in| = {:.4}, |out| = {:.4}", in_n, out_n);
    assert!(in_n > out_n);
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
        let (i, o) = aou::objectosphere_norms(&in_class(), &out_class());
        assert!(i > o);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Swap roles: claim out-class is "in-class" — falsifier (in > out) breaks.
        let (i, o) = aou::objectosphere_norms(&out_class(), &in_class());
        assert!(i < o);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = aou::objectosphere_norms(&in_class(), &out_class());
        let b = aou::objectosphere_norms(&in_class(), &out_class());
        assert_eq!(a, b);
    }
}
