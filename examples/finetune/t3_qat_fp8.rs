//! # Tier 3.17 — QAT FP8 fake-quant (llama family)
//!
//! Falsifier: QAT FP8 fake-quant matches dequant→quant identity within ε.
//!
//! Run with: cargo run --example t3_qat_fp8

use apr_cookbook::finetune::tier3_closeout as t3c;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture() -> Vec<f64> {
    vec![0.5, 1.0, 1.5, 2.0, 3.0, 7.5, 100.0, 0.125, 0.25]
}
const TOL: f64 = 7.0;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_qat_fp8")?;
    let err = t3c::fp8_max_round_trip_error(&fixture());
    println!(
        "✓ FP8 fake-quant max round-trip error: {:.4} (tol {})",
        err, TOL
    );
    assert!(err < TOL);
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
        assert!(t3c::fp8_max_round_trip_error(&fixture()) < TOL);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Massive values exceed FP8 representable range → recovery error blows tol.
        let big = vec![1e6, 1e8, 1e10];
        let err = t3c::fp8_max_round_trip_error(&big);
        assert!(err > TOL);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = t3c::fp8_max_round_trip_error(&fixture());
        let b = t3c::fp8_max_round_trip_error(&fixture());
        assert_eq!(a, b);
    }
}
