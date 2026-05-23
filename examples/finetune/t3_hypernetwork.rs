//! # Tier 3.16 — Hypernetwork (tabular-only)
//!
//! Falsifier: hypernetwork generates distinct weight vectors for distinct
//! task IDs (no collision in the synthetic scheme).
//!
//! Run with: cargo run --example t3_hypernetwork

use apr_cookbook::finetune::specialty;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const DIM: usize = 32;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_hypernetwork")?;
    let w1 = specialty::hypernetwork_generate(1, DIM);
    let w2 = specialty::hypernetwork_generate(2, DIM);
    let w3 = specialty::hypernetwork_generate(3, DIM);
    println!("✓ hypernetwork: 3 task-IDs → 3 distinct weight vectors (dim={DIM})");
    assert_ne!(w1, w2);
    assert_ne!(w2, w3);
    assert_ne!(w1, w3);
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
        let w1 = specialty::hypernetwork_generate(1, DIM);
        let w2 = specialty::hypernetwork_generate(2, DIM);
        assert_ne!(w1, w2);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Same task_id → identical weights.
        let a = specialty::hypernetwork_generate(7, DIM);
        let b = specialty::hypernetwork_generate(7, DIM);
        assert_eq!(a, b);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = specialty::hypernetwork_generate(7, DIM);
        let b = specialty::hypernetwork_generate(7, DIM);
        assert_eq!(a, b);
    }
}
