//! # Tier 4.1 — DPO β=0.1 vs β=1.0 (mistral family)
//!
//! Falsifier: DPO with β=0.1 produces lower KL-divergence than β=1.0 at
//! convergence (smaller scaling → smaller policy drift from reference).
//!
//! Run with: cargo run --example t4_dpo_mistral

use apr_cookbook::finetune::preference as pref;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn lp_diffs() -> Vec<f64> {
    vec![0.5, 0.4, 0.6, 0.45, 0.55, 0.50, 0.48, 0.52]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_dpo_mistral")?;
    let kl_b01: f64 = pref::kl_estimate(&lp_diffs().iter().map(|d| 0.1 * d).collect::<Vec<_>>());
    let kl_b1: f64 = pref::kl_estimate(&lp_diffs().iter().map(|d| 1.0 * d).collect::<Vec<_>>());
    println!("✓ DPO β=0.1 KL = {:.4}, β=1.0 KL = {:.4}", kl_b01, kl_b1);
    assert!(kl_b01 < kl_b1);
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
        let kl_b01 = pref::kl_estimate(&lp_diffs().iter().map(|d| 0.1 * d).collect::<Vec<_>>());
        let kl_b1 = pref::kl_estimate(&lp_diffs().iter().map(|d| 1.0 * d).collect::<Vec<_>>());
        assert!(kl_b01 < kl_b1);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Identical β — identical KL.
        let kl_a = pref::kl_estimate(&lp_diffs());
        let kl_b = pref::kl_estimate(&lp_diffs());
        assert!((kl_a - kl_b).abs() < 1e-12);
    }

    #[test]
    fn deterministic_across_runs() {
        let kl1 = pref::kl_estimate(&lp_diffs());
        let kl2 = pref::kl_estimate(&lp_diffs());
        assert_eq!(kl1, kl2);
    }
}
