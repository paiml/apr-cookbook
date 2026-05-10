//! # Tier 3.5 — Multitask SFT (llama family)
//!
//! Falsifier: 3-task SFT with shared encoder — per-task loss decreases
//! independently after a step (no catastrophic forgetting on any task).
//!
//! Run with: cargo run --example t3_multimodal_multitask

use apr_cookbook::finetune::multimodal as mm;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const INITIAL_LOSSES: [f64; 3] = [1.5, 2.0, 0.8];
const WEIGHTS: [f64; 3] = [0.5, 0.3, 0.2];

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_multimodal_multitask")?;
    let step = mm::synthetic_multitask_step(&INITIAL_LOSSES, &WEIGHTS);
    println!(
        "✓ multitask: per-task before {:?} → after {:?} (total {:.4} → {:.4})",
        step.task_losses_before,
        step.task_losses_after,
        step.total_before(),
        step.total_after()
    );
    assert!(
        step.all_decreased(),
        "every task's loss must decrease (no catastrophic forgetting)"
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
        let s = mm::synthetic_multitask_step(&INITIAL_LOSSES, &WEIGHTS);
        assert!(s.all_decreased());
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // If a hypothetical step *doesn't* lower every task, all_decreased = false.
        let bogus = mm::MultitaskStep {
            task_losses_before: vec![1.0, 2.0, 3.0],
            task_losses_after: vec![0.5, 2.5, 2.5],
            weights: vec![1.0, 1.0, 1.0],
        };
        assert!(!bogus.all_decreased());
    }

    #[test]
    fn deterministic_across_runs() {
        let a = mm::synthetic_multitask_step(&INITIAL_LOSSES, &WEIGHTS);
        let b = mm::synthetic_multitask_step(&INITIAL_LOSSES, &WEIGHTS);
        assert_eq!(a, b);
    }
}
