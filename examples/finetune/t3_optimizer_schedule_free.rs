//! # Tier 3.10 — Schedule-free optimizer (tabular-only)
//!
//! Falsifier: schedule-free optimizer returns the base learning rate at every
//! step — no decay applied — distinct from cosine schedule which decays.
//!
//! Run with: cargo run --example t3_optimizer_schedule_free

use apr_cookbook::finetune::encoders_optimizers as enc;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const BASE_LR: f64 = 0.001;
const TOTAL_STEPS: u32 = 1000;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_optimizer_schedule_free")?;
    for step in [0_u32, 100, 500, 999] {
        let sf = enc::schedule_free_lr(BASE_LR, step, TOTAL_STEPS);
        let cos = enc::cosine_lr(BASE_LR, step, TOTAL_STEPS);
        assert_eq!(
            sf, BASE_LR,
            "schedule-free must return base_lr at step {step}"
        );
        let _ = cos;
    }
    println!("✓ schedule-free: lr = {BASE_LR} constant across all steps");
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
        for step in [0_u32, 100, 500, 999] {
            assert_eq!(enc::schedule_free_lr(BASE_LR, step, TOTAL_STEPS), BASE_LR);
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Cosine LR is NOT constant.
        let lr_start = enc::cosine_lr(BASE_LR, 0, TOTAL_STEPS);
        let lr_end = enc::cosine_lr(BASE_LR, TOTAL_STEPS, TOTAL_STEPS);
        assert!(lr_start > lr_end);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = enc::schedule_free_lr(BASE_LR, 100, TOTAL_STEPS);
        let b = enc::schedule_free_lr(BASE_LR, 100, TOTAL_STEPS);
        assert_eq!(a, b);
    }
}
