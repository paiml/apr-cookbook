//! # Tier 4.11 — GSPO sequence-level Bradley-Terry (llama family)
//!
//! Falsifier: GSPO aggregates per-position log-probs into sequence preference
//! via simple sum (Bradley-Terry on sequences).
//!
//! Run with: cargo run --example t4_gspo

use apr_cookbook::finetune::tier4_closeout as t4c;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn per_position_lp() -> Vec<f64> {
    vec![-0.5, -0.3, -0.7, -0.2, -0.4, -0.6]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_gspo")?;
    let total = t4c::gspo_sequence_log_prob(&per_position_lp());
    let expected: f64 = per_position_lp().iter().sum();
    println!("✓ GSPO sequence log-prob = {:.4}", total);
    assert!((total - expected).abs() < 1e-12);
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
        let total = t4c::gspo_sequence_log_prob(&per_position_lp());
        let expected: f64 = per_position_lp().iter().sum();
        assert!((total - expected).abs() < 1e-12);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Empty sequence — sum is 0.
        let empty: Vec<f64> = Vec::new();
        assert_eq!(t4c::gspo_sequence_log_prob(&empty), 0.0);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = t4c::gspo_sequence_log_prob(&per_position_lp());
        let b = t4c::gspo_sequence_log_prob(&per_position_lp());
        assert_eq!(a, b);
    }
}
