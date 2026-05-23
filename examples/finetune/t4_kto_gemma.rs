//! # Tier 4.3 — KTO works on isolated samples (gemma family)
//!
//! Falsifier: KTO does not require paired preferences — a single positive
//! sample produces a finite, well-defined loss.
//!
//! Run with: cargo run --example t4_kto_gemma

use apr_cookbook::finetune::preference as pref;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const BETA: f64 = 0.1;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_kto_gemma")?;
    // 5 isolated positive samples, no rejected counterpart needed.
    let isolated_lp_diffs = [0.5_f64, 0.4, 0.6, 0.3, 0.5];
    for lp in isolated_lp_diffs {
        let l = pref::kto_loss(lp, BETA, true, 0.5);
        assert!(
            l > 0.0 && l.is_finite(),
            "isolated KTO loss at lp={lp} = {l}"
        );
    }
    println!(
        "✓ KTO: {} isolated positive samples → all finite losses",
        isolated_lp_diffs.len()
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
        for lp in [0.5_f64, 0.4, 0.6] {
            let l = pref::kto_loss(lp, BETA, true, 0.5);
            assert!(l > 0.0 && l.is_finite());
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // NaN log-prob diff propagates — well-formed KTO must not produce NaN.
        let l = pref::kto_loss(f64::NAN, BETA, true, 0.5);
        assert!(l.is_nan());
    }

    #[test]
    fn deterministic_across_runs() {
        let a = pref::kto_loss(0.5, BETA, true, 0.5);
        let b = pref::kto_loss(0.5, BETA, true, 0.5);
        assert_eq!(a, b);
    }
}
