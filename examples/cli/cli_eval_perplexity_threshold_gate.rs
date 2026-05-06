//! # apr eval — Perplexity Threshold Gate (spec H13)
//!
//! `apr eval --threshold <X>` enforces `PPL <= X` (default 20.0 per spec
//! H13). This recipe builds the gate and asserts the contract: NaN/inf
//! perplexity must NEVER pass (silent failure on training divergence is
//! the GH-186 footgun); negative threshold is a config error.
//!
//! Demonstrates the **EVAL.5** recipe for PMAT-103 (apr eval coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender spec H13 (PPL <= 20)
//!
//! Run with: cargo run --example cli_eval_perplexity_threshold_gate
//!
//! Added by PMAT-103 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PplVerdict {
    Pass { observed: f64, threshold: f64 },
    Fail { observed: f64, threshold: f64 },
    NonFiniteObserved,
    InvalidThreshold,
}

pub fn check_ppl(observed: f64, threshold: f64) -> PplVerdict {
    if !threshold.is_finite() || threshold <= 0.0 {
        return PplVerdict::InvalidThreshold;
    }
    if !observed.is_finite() || observed <= 0.0 {
        return PplVerdict::NonFiniteObserved;
    }
    if observed <= threshold {
        PplVerdict::Pass {
            observed,
            threshold,
        }
    } else {
        PplVerdict::Fail {
            observed,
            threshold,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_eval_perplexity_threshold_gate")?;

    for (label, ppl, threshold) in [
        ("happy", 12.5, 20.0),
        ("at-threshold", 20.0, 20.0),
        ("over-budget", 35.0, 20.0),
        ("nan ppl", f64::NAN, 20.0),
        ("inf ppl", f64::INFINITY, 20.0),
        ("zero threshold", 5.0, 0.0),
        ("negative threshold", 5.0, -1.0),
    ] {
        println!("{label:>20}  →  {:?}", check_ppl(ppl, threshold));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_below_threshold_passes() {
        let v = check_ppl(12.5, 20.0);
        assert!(matches!(v, PplVerdict::Pass { .. }));
    }

    #[test]
    fn boundary_at_exact_threshold_passes() {
        // PPL == threshold is conservative-pass (matches spec H13 = ≤).
        let v = check_ppl(20.0, 20.0);
        assert!(matches!(v, PplVerdict::Pass { .. }));
    }

    #[test]
    fn over_threshold_fails() {
        let v = check_ppl(20.001, 20.0);
        assert!(matches!(v, PplVerdict::Fail { .. }));
    }

    #[test]
    fn nan_observed_rejected() {
        // CRITICAL: a divergent training run produces NaN PPL — must NEVER pass.
        let v = check_ppl(f64::NAN, 20.0);
        assert_eq!(v, PplVerdict::NonFiniteObserved);
    }

    #[test]
    fn inf_observed_rejected() {
        let v = check_ppl(f64::INFINITY, 20.0);
        assert_eq!(v, PplVerdict::NonFiniteObserved);
    }

    #[test]
    fn zero_or_negative_observed_rejected() {
        // PPL is exp(loss); exp returns positive. PPL ≤ 0 is impossible.
        assert_eq!(check_ppl(0.0, 20.0), PplVerdict::NonFiniteObserved);
        assert_eq!(check_ppl(-1.0, 20.0), PplVerdict::NonFiniteObserved);
    }

    #[test]
    fn invalid_threshold_rejected() {
        assert_eq!(check_ppl(5.0, 0.0), PplVerdict::InvalidThreshold);
        assert_eq!(check_ppl(5.0, -1.0), PplVerdict::InvalidThreshold);
        assert_eq!(check_ppl(5.0, f64::NAN), PplVerdict::InvalidThreshold);
    }
}
