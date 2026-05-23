//! # Tier 4.8 — Nash-MD (phi family)
//!
//! Falsifier: Nash-MD policy converges to Nash equilibrium of preference game
//! — KL drift decays to ≤ ε after T steps.
//!
//! Run with: cargo run --example t4_nash_md

use apr_cookbook::finetune::online_alt as oa;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const INITIAL_KL: f64 = 0.5;
const DECAY: f64 = 0.1;
const T: u32 = 50;
const EPS: f64 = 0.005;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_nash_md")?;
    let kl = oa::nash_md_kl_drift(INITIAL_KL, DECAY, T);
    println!("✓ Nash-MD: KL drift after {T} steps = {:.6} (ε={EPS})", kl);
    assert!(kl <= EPS);
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
        assert!(oa::nash_md_kl_drift(INITIAL_KL, DECAY, T) <= EPS);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Zero decay → KL never shrinks.
        let kl = oa::nash_md_kl_drift(INITIAL_KL, 0.0, T);
        assert!(kl > EPS);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = oa::nash_md_kl_drift(INITIAL_KL, DECAY, T);
        let b = oa::nash_md_kl_drift(INITIAL_KL, DECAY, T);
        assert_eq!(a, b);
    }
}
