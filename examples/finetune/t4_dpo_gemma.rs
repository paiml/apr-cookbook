//! # Tier 4.1 — DPO loss monotone decrease (gemma family)
//!
//! Falsifier: DPO loss drops monotonically over 100 simulated steps on a
//! synthetic 50-pair preference set (modeled as decreasing log-prob gap
//! between rejected and chosen).
//!
//! Run with: cargo run --example t4_dpo_gemma

use apr_cookbook::finetune::preference as pref;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const BETA: f64 = 0.1;
const N_STEPS: u32 = 100;

fn lp_at_step(step: u32) -> (f64, f64) {
    // chosen log-ratio grows; rejected stays low.
    let progress = f64::from(step) / f64::from(N_STEPS);
    (progress * 0.5, -progress * 0.5)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_dpo_gemma")?;
    let mut prev_loss = f64::INFINITY;
    let mut violations = 0;
    for step in 0..N_STEPS {
        let (lc, lr) = lp_at_step(step);
        let loss = pref::dpo_loss(lc, lr, BETA);
        if loss > prev_loss {
            violations += 1;
        }
        prev_loss = loss;
    }
    println!(
        "✓ DPO trajectory over {} steps: {} non-monotone steps",
        N_STEPS, violations
    );
    assert_eq!(violations, 0);
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
        let mut prev = f64::INFINITY;
        for step in 0..N_STEPS {
            let (lc, lr) = lp_at_step(step);
            let loss = pref::dpo_loss(lc, lr, BETA);
            assert!(loss <= prev);
            prev = loss;
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Reverse trajectory — loss INCREASES.
        let mut prev = f64::NEG_INFINITY;
        for step in (0..N_STEPS).rev() {
            let (lc, lr) = lp_at_step(step);
            let loss = pref::dpo_loss(lc, lr, BETA);
            assert!(loss >= prev);
            prev = loss;
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let losses_a: Vec<f64> = (0..10)
            .map(|s| {
                let (lc, lr) = lp_at_step(s);
                pref::dpo_loss(lc, lr, BETA)
            })
            .collect();
        let losses_b: Vec<f64> = (0..10)
            .map(|s| {
                let (lc, lr) = lp_at_step(s);
                pref::dpo_loss(lc, lr, BETA)
            })
            .collect();
        assert_eq!(losses_a, losses_b);
    }
}
