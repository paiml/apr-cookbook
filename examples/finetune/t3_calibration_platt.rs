//! # Tier 3.3 — Platt scaling calibration (tabular-only)
//!
//! Falsifier: Platt scaling produces sigmoid-shaped curve through (0.5, 0.5)
//! when intercept b = 0 (any slope a). Closed-form: sigmoid(a·0 + 0) = 0.5.
//!
//! Run with: cargo run --example t3_calibration_platt

use apr_cookbook::finetune::calibration as cal;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const A: f64 = 1.5;
const B: f64 = 0.0;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_calibration_platt")?;
    let p_at_zero = cal::platt_apply(0.0, A, B);
    println!("✓ Platt a={} b={}: P(0) = {:.4}", A, B, p_at_zero);
    assert!(
        (p_at_zero - 0.5).abs() < 1e-12,
        "Platt with b=0 must pass through (0, 0.5)"
    );
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
        for a in &[0.5, 1.0, 2.0, 5.0] {
            assert!((cal::platt_apply(0.0, *a, B) - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // b ≠ 0 → curve does NOT pass through (0, 0.5).
        let p = cal::platt_apply(0.0, A, 1.0);
        assert!((p - 0.5).abs() > 0.1);
    }

    #[test]
    fn deterministic_across_runs() {
        assert_eq!(cal::platt_apply(0.3, A, B), cal::platt_apply(0.3, A, B));
    }
}
