//! # Tier 3.8 — MC-dropout uncertainty (tabular-only)
//!
//! Falsifier: predictive variance on OOD samples ≥ variance on in-distribution
//! by ≥ 50%.
//!
//! Run with: cargo run --example t3_uncertainty_mc_dropout

use apr_cookbook::finetune::anomaly_open_uncertainty as aou;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn in_dist() -> Vec<f64> {
    vec![1.0, 1.01, 0.99, 1.0, 1.02, 0.98, 1.01]
}
fn ood() -> Vec<f64> {
    vec![1.0, 1.5, 0.5, 1.3, 0.7, 1.4, 0.6]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_uncertainty_mc_dropout")?;
    let r = aou::mc_dropout_variance_ratio(&in_dist(), &ood());
    println!("✓ MC-dropout: var(OOD)/var(in-dist) = {:.4}", r);
    assert!(r >= 1.5);
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
        assert!(aou::mc_dropout_variance_ratio(&in_dist(), &ood()) >= 1.5);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Identical distributions → ratio ≈ 1.0.
        let r = aou::mc_dropout_variance_ratio(&in_dist(), &in_dist());
        assert!((r - 1.0).abs() < 0.01);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = aou::mc_dropout_variance_ratio(&in_dist(), &ood());
        let b = aou::mc_dropout_variance_ratio(&in_dist(), &ood());
        assert_eq!(a, b);
    }
}
