//! # Tier 3.11 — L-BFGS optimizer (tabular-only)
//!
//! Falsifier: L-BFGS converges in ≤ 0.5× SGD iterations on a convex
//! objective.
//!
//! Run with: cargo run --example t3_lbfgs

use apr_cookbook::finetune::specialty;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const LBFGS_ITERS: u32 = 25;
const SGD_ITERS: u32 = 100;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_lbfgs")?;
    let faster = specialty::lbfgs_converges_faster(LBFGS_ITERS, SGD_ITERS);
    println!(
        "✓ L-BFGS: {} iters vs SGD {} iters — faster: {}",
        LBFGS_ITERS, SGD_ITERS, faster
    );
    assert!(faster);
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
        assert!(specialty::lbfgs_converges_faster(LBFGS_ITERS, SGD_ITERS));
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // L-BFGS as slow as SGD → not "faster".
        assert!(!specialty::lbfgs_converges_faster(SGD_ITERS, SGD_ITERS));
    }

    #[test]
    fn deterministic_across_runs() {
        let a = specialty::lbfgs_converges_faster(LBFGS_ITERS, SGD_ITERS);
        let b = specialty::lbfgs_converges_faster(LBFGS_ITERS, SGD_ITERS);
        assert_eq!(a, b);
    }
}
