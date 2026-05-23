//! # Tier 3.2 — ASHA hyperopt (tabular-only)
//!
//! Falsifier: ASHA early-prunes ≥ 50% of trials in the first rung at η=2
//! reduction factor. Closed-form: kept = ⌊n/η⌋, pruned = n - kept ≥ kept
//! when n ≥ 2.
//!
//! Run with: cargo run --example t3_hyperopt_asha

use apr_cookbook::finetune::hyperopt;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const N_INITIAL: u32 = 100;
const ETA: u32 = 2;
const N_RUNGS: u32 = 4;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_hyperopt_asha")?;
    let rungs = hyperopt::asha_rungs(N_INITIAL, ETA, N_RUNGS);
    println!(
        "✓ ASHA n={} η={}: rungs (kept, pruned) = {:?}",
        N_INITIAL, ETA, rungs
    );
    let (kept0, pruned0) = rungs[0];
    assert!(
        pruned0 >= kept0,
        "ASHA must prune ≥ 50% in first rung, got kept={kept0} pruned={pruned0}"
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
        let rungs = hyperopt::asha_rungs(N_INITIAL, ETA, N_RUNGS);
        let (kept, pruned) = rungs[0];
        assert!(pruned >= kept);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // η=1 means no pruning (every trial advances).
        let rungs = hyperopt::asha_rungs(N_INITIAL, 1, 1);
        let (kept, pruned) = rungs[0];
        assert!(kept >= pruned);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = hyperopt::asha_rungs(N_INITIAL, ETA, N_RUNGS);
        let b = hyperopt::asha_rungs(N_INITIAL, ETA, N_RUNGS);
        assert_eq!(a, b);
    }
}
