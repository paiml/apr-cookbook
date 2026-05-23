//! # Tier 3.12 — FAMO multi-task balancing (tabular-only)
//!
//! Falsifier: FAMO multi-task balancing — per-task gradient norm is at most
//! 2× the median (no task dominates).
//!
//! Run with: cargo run --example t3_multitask_famo

use apr_cookbook::finetune::specialty;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const GRAD_NORMS: [f64; 5] = [0.4, 0.5, 0.45, 0.55, 0.6];

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_multitask_famo")?;
    let balanced = specialty::famo_balanced(&GRAD_NORMS);
    println!(
        "✓ FAMO grad norms {:?} → balanced = {}",
        GRAD_NORMS, balanced
    );
    assert!(balanced);
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
        assert!(specialty::famo_balanced(&GRAD_NORMS));
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // One task dominates → unbalanced.
        let bogus = [0.1, 0.1, 0.1, 5.0];
        assert!(!specialty::famo_balanced(&bogus));
    }

    #[test]
    fn deterministic_across_runs() {
        let a = specialty::famo_balanced(&GRAD_NORMS);
        let b = specialty::famo_balanced(&GRAD_NORMS);
        assert_eq!(a, b);
    }
}
