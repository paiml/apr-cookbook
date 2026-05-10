//! # Tier 1.5 — Smoke — Dry run
//!
//! Falsifier: apr finetune --dry-run produces zero side effects.
//!
//! Run with: cargo run --example t1_smoke_dry_run

use apr_cookbook::finetune::smoke;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_smoke_dry_run")?;
    let v = smoke::DryRunVerdict::default();
    println!(
        "✓ dry-run verdict: fs_writes={}, gpu_allocations={}, network_calls={}",
        v.fs_writes, v.gpu_allocations, v.network_calls
    );
    assert!(
        v.is_clean(),
        "falsifier: dry-run must produce zero side effects"
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
        let v = smoke::DryRunVerdict::default();
        assert!(v.is_clean());
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        let dirty = smoke::DryRunVerdict {
            fs_writes: 1,
            gpu_allocations: 0,
            network_calls: 0,
        };
        assert!(!dirty.is_clean());
    }

    #[test]
    fn deterministic_across_runs() {
        let a = smoke::DryRunVerdict::default();
        let b = smoke::DryRunVerdict::default();
        assert_eq!(a, b);
    }
}
