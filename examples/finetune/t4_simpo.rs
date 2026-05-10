//! # Tier 4.9 — SimPO reference-free margin (qwen3 family)
//!
//! Falsifier: SimPO uses only chosen−rejected log-prob difference; no
//! reference-model forward passes required.
//!
//! Run with: cargo run --example t4_simpo

use apr_cookbook::finetune::online_alt as oa;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const LP_CHOSEN: f64 = 0.5;
const LP_REJECTED: f64 = -0.3;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_simpo")?;
    let m = oa::simpo_margin(LP_CHOSEN, LP_REJECTED);
    println!("✓ SimPO margin (no ref model): {:.4}", m);
    assert!((m - (LP_CHOSEN - LP_REJECTED)).abs() < 1e-12);
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
        let m = oa::simpo_margin(LP_CHOSEN, LP_REJECTED);
        assert!((m - 0.8).abs() < 1e-12);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Equal log-probs → margin = 0 (no preference signal).
        assert_eq!(oa::simpo_margin(0.5, 0.5), 0.0);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = oa::simpo_margin(LP_CHOSEN, LP_REJECTED);
        let b = oa::simpo_margin(LP_CHOSEN, LP_REJECTED);
        assert_eq!(a, b);
    }
}
