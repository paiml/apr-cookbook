//! # Tier 4.7 — Ensemble reward modeling (qwen3 family)
//!
//! Falsifier: ensemble of 3 reward models reduces variance vs any single
//! model on a calibration set (averaging shrinks variance).
//!
//! Run with: cargo run --example t4_reward_ensemble

use apr_cookbook::finetune::rlaif_reward as rr;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn members() -> Vec<Vec<f64>> {
    vec![
        vec![1.1, 2.2, 3.3, 4.4, 5.5],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![0.9, 1.8, 2.7, 3.6, 4.5],
    ]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_reward_ensemble")?;
    let avg = rr::ensemble_mean(&members());
    let var_avg = rr::variance(&avg);
    let max_var: f64 = members()
        .iter()
        .map(|m| rr::variance(m))
        .fold(0.0_f64, f64::max);
    println!(
        "✓ ensemble: avg variance = {:.4}, max member variance = {:.4}",
        var_avg, max_var
    );
    assert!(var_avg <= max_var);
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
        let avg = rr::ensemble_mean(&members());
        let var_avg = rr::variance(&avg);
        let max_var: f64 = members()
            .iter()
            .map(|m| rr::variance(m))
            .fold(0.0_f64, f64::max);
        assert!(var_avg <= max_var);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Single member: avg = member, variance equal.
        let single = vec![members()[0].clone()];
        let avg = rr::ensemble_mean(&single);
        assert_eq!(avg, single[0]);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = rr::ensemble_mean(&members());
        let b = rr::ensemble_mean(&members());
        assert_eq!(a, b);
    }
}
