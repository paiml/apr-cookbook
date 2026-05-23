//! # Tier 1.5 — Smoke — Plan
//!
//! Falsifier: apr finetune --plan emits a plan but writes no checkpoint.
//!
//! Run with: cargo run --example t1_smoke_plan

use apr_cookbook::finetune::smoke;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_smoke_plan")?;
    let plan = smoke::plan_lora(7_000_000_000, 8, 32);
    println!(
        "✓ plan: method={}, trainable={}, ~{}MB, wrote_checkpoint={}",
        plan.method, plan.trainable_params, plan.estimated_memory_mb, plan.wrote_checkpoint
    );
    assert!(
        plan.is_plan_only(),
        "falsifier: plan must NOT write checkpoint"
    );
    assert!(
        plan.trainable_params > 0,
        "plan should report trainable param count"
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
        for r in [4u32, 8, 16, 32, 64] {
            let p = smoke::plan_lora(7_000_000_000, r, 32);
            assert!(p.is_plan_only());
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // If we constructed a PlanReport with wrote_checkpoint=true,
        // is_plan_only() should be false.
        let mut p = smoke::plan_lora(7_000_000_000, 8, 32);
        p.wrote_checkpoint = true;
        assert!(!p.is_plan_only());
    }

    #[test]
    fn deterministic_across_runs() {
        let a = smoke::plan_lora(7_000_000_000, 8, 32);
        let b = smoke::plan_lora(7_000_000_000, 8, 32);
        assert_eq!(a, b);
    }
}
