//! # Tier 1.5 — Smoke — Early stop
//!
//! Falsifier: early-stop on plateau halts within patience+1 epochs.
//!
//! Run with: cargo run --example t1_smoke_early_stop

use apr_cookbook::finetune::smoke;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_smoke_early_stop")?;
    let losses = vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.2, 0.21, 0.22, 0.23, 0.24];
    let patience = 2u32;
    let stopped = smoke::simulate_early_stop(&losses, patience);
    println!(
        "✓ early-stop on {}-epoch loss curve with patience={}: stopped at epoch {}",
        losses.len(),
        patience,
        stopped
    );
    let plateau_start = 5u32; // index where best=0.2 was last achieved
    assert!(
        stopped <= plateau_start + patience,
        "falsifier: should stop within patience+1, got {stopped}"
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
        let losses = vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.2, 0.21, 0.22, 0.23, 0.24];
        let stopped = smoke::simulate_early_stop(&losses, 2);
        assert!(stopped <= 7);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Strictly decreasing loss never triggers early stop — stopped equals last index
        let losses = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5];
        let stopped = smoke::simulate_early_stop(&losses, 2);
        assert_eq!(stopped, losses.len() as u32 - 1);
    }

    #[test]
    fn deterministic_across_runs() {
        let losses = vec![1.0, 0.8, 0.6, 0.5, 0.5, 0.5, 0.5];
        assert_eq!(
            smoke::simulate_early_stop(&losses, 2),
            smoke::simulate_early_stop(&losses, 2)
        );
    }
}
