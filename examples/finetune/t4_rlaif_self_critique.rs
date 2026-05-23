//! # Tier 4.6 — RLAIF self-critique loop (gemma family)
//!
//! Falsifier: self-critique reduces unsafe-output rate without dropping
//! helpfulness more than 5pp.
//!
//! Run with: cargo run --example t4_rlaif_self_critique

use apr_cookbook::finetune::rlaif_reward as rr;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const PRE_UNSAFE: f64 = 0.30;
const POST_UNSAFE: f64 = 0.18;
const PRE_HELP: f64 = 0.80;
const POST_HELP: f64 = 0.78;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_rlaif_self_critique")?;
    let ok = rr::self_critique_balanced(PRE_UNSAFE, POST_UNSAFE, PRE_HELP, POST_HELP);
    println!(
        "✓ self-critique: unsafe {:.1}% → {:.1}%, helpful {:.1}% → {:.1}% (balanced: {})",
        PRE_UNSAFE * 100.0,
        POST_UNSAFE * 100.0,
        PRE_HELP * 100.0,
        POST_HELP * 100.0,
        ok
    );
    assert!(ok);
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
        assert!(rr::self_critique_balanced(
            PRE_UNSAFE,
            POST_UNSAFE,
            PRE_HELP,
            POST_HELP
        ));
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Helpfulness drops 10pp — falsifier breaks.
        assert!(!rr::self_critique_balanced(
            PRE_UNSAFE,
            POST_UNSAFE,
            0.80,
            0.70
        ));
    }

    #[test]
    fn deterministic_across_runs() {
        let a = rr::self_critique_balanced(PRE_UNSAFE, POST_UNSAFE, PRE_HELP, POST_HELP);
        let b = rr::self_critique_balanced(PRE_UNSAFE, POST_UNSAFE, PRE_HELP, POST_HELP);
        assert_eq!(a, b);
    }
}
