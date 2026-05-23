//! # Tier 1.5 — Smoke — Resume
//!
//! Falsifier: interrupted finetune resumes at the last persisted optimizer step.
//!
//! Run with: cargo run --example t1_smoke_resume

use apr_cookbook::finetune::smoke;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_smoke_resume")?;
    let r = smoke::simulate_resume(1000, 250);
    println!(
        "✓ resume: last_step={}, last_epoch={}, interrupted_at={:?}",
        r.last_step, r.last_epoch, r.interrupted_at
    );
    assert_eq!(r.last_step, 250);
    assert_eq!(r.interrupted_at, Some(250));
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
        for step in [50u32, 100, 250, 500, 999] {
            let r = smoke::simulate_resume(1000, step);
            assert_eq!(r.last_step, step);
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Resume at total_steps means no interruption — interrupted_at = None.
        let r = smoke::simulate_resume(1000, 1000);
        assert!(r.interrupted_at.is_none());
    }

    #[test]
    fn deterministic_across_runs() {
        let a = smoke::simulate_resume(1000, 250);
        let b = smoke::simulate_resume(1000, 250);
        assert_eq!(a, b);
    }
}
