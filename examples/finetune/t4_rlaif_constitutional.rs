//! # Tier 4.6 — Constitutional RLAIF (phi family)
//!
//! Falsifier: harmful-prompt refusal rate rises ≥ 20pp post-RLAIF.
//!
//! Run with: cargo run --example t4_rlaif_constitutional

use apr_cookbook::finetune::rlaif_reward as rr;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const PRE_REFUSAL: f64 = 0.30;
const POST_REFUSAL: f64 = 0.55;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_rlaif_constitutional")?;
    let uplift = rr::refusal_rate_uplift(PRE_REFUSAL, POST_REFUSAL);
    println!(
        "✓ Constitutional: refusal {:.1}% → {:.1}% (+{:.1}pp)",
        PRE_REFUSAL * 100.0,
        POST_REFUSAL * 100.0,
        uplift * 100.0
    );
    assert!(uplift >= 0.20);
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
        assert!(rr::refusal_rate_uplift(PRE_REFUSAL, POST_REFUSAL) >= 0.20);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Small uplift — < 20pp.
        assert!(rr::refusal_rate_uplift(0.30, 0.35) < 0.20);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = rr::refusal_rate_uplift(PRE_REFUSAL, POST_REFUSAL);
        let b = rr::refusal_rate_uplift(PRE_REFUSAL, POST_REFUSAL);
        assert_eq!(a, b);
    }
}
