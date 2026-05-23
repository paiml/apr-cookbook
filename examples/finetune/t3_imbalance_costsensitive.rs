//! # Tier 3.4 — Cost-sensitive learning for imbalance (tabular-only)
//!
//! Falsifier: asymmetric (10:1) cost matrix shifts the optimal decision
//! threshold below 0.5 so more positive predictions are made (favoring
//! recall over precision). Closed-form: t* = cost_fp / (cost_fp + cost_fn).
//!
//! Run with: cargo run --example t3_imbalance_costsensitive

use apr_cookbook::finetune::imbalance as imb;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const COST_FN: f64 = 10.0;
const COST_FP: f64 = 1.0;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_imbalance_costsensitive")?;
    let t = imb::cost_sensitive_threshold(COST_FN, COST_FP);
    println!(
        "✓ cost-sensitive threshold (cost_fn={} cost_fp={}): t* = {:.4}",
        COST_FN, COST_FP, t
    );
    assert!(t < 0.5, "asymmetric cost (10:1 FN:FP) must lower threshold");
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
        assert!(imb::cost_sensitive_threshold(COST_FN, COST_FP) < 0.5);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Symmetric costs → t* = 0.5.
        let t = imb::cost_sensitive_threshold(1.0, 1.0);
        assert!((t - 0.5).abs() < 1e-12);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = imb::cost_sensitive_threshold(COST_FN, COST_FP);
        let b = imb::cost_sensitive_threshold(COST_FN, COST_FP);
        assert_eq!(a, b);
    }
}
