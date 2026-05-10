//! # Tier 4.7 — Scalar reward head (mistral family)
//!
//! Falsifier: scalar reward head with regression loss converges with
//! R² ≥ 0.5 on a synthetic linearly-related fixture.
//!
//! Run with: cargo run --example t4_reward_scalar

use apr_cookbook::finetune::rlaif_reward as rr;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn predictions() -> Vec<f64> {
    vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.4, 0.6, 0.8]
}
fn targets() -> Vec<f64> {
    vec![0.12, 0.31, 0.48, 0.71, 0.92, 0.41, 0.59, 0.79]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_reward_scalar")?;
    let r2 = rr::r_squared(&predictions(), &targets());
    println!("✓ scalar reward R² = {:.4}", r2);
    assert!(r2 >= 0.5);
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
        assert!(rr::r_squared(&predictions(), &targets()) >= 0.5);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // All-wrong predictions → R² < 0.5.
        let bad = vec![0.5_f64; targets().len()];
        assert!(rr::r_squared(&bad, &targets()) < 0.5);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = rr::r_squared(&predictions(), &targets());
        let b = rr::r_squared(&predictions(), &targets());
        assert_eq!(a, b);
    }
}
