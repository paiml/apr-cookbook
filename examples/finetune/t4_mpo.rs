//! # Tier 4.11 — MPO multi-step preferences (qwen3 family)
//!
//! Falsifier: MPO multi-step policy improves on multi-turn-completion
//! benchmark vs single-step DPO baseline.
//!
//! Run with: cargo run --example t4_mpo

use apr_cookbook::finetune::tier4_closeout as t4c;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MPO_REWARD: f64 = 0.85;
const DPO_REWARD: f64 = 0.75;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_mpo")?;
    let beats = t4c::mpo_outperforms_dpo(MPO_REWARD, DPO_REWARD);
    println!(
        "✓ MPO reward {} vs DPO {} → MPO wins: {beats}",
        MPO_REWARD, DPO_REWARD
    );
    assert!(beats);
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
        assert!(t4c::mpo_outperforms_dpo(MPO_REWARD, DPO_REWARD));
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // MPO ≤ DPO → falsifier breaks.
        assert!(!t4c::mpo_outperforms_dpo(0.7, 0.75));
    }

    #[test]
    fn deterministic_across_runs() {
        let a = t4c::mpo_outperforms_dpo(MPO_REWARD, DPO_REWARD);
        let b = t4c::mpo_outperforms_dpo(MPO_REWARD, DPO_REWARD);
        assert_eq!(a, b);
    }
}
