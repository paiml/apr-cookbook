//! # apr eval — pass@k Temperature Pairing (ALB-088)
//!
//! `apr eval --samples <K> --temperature <T>` computes pass@k. The
//! temperature MUST be ≥ some non-zero value (typically 0.8) when k > 1
//! — k samples at T=0 are identical (greedy) and pass@k collapses to
//! pass@1. This recipe enforces the pairing rule and prevents the silent
//! "looks like pass@k but isn't" failure mode.
//!
//! Demonstrates the **EVAL.7** recipe for PMAT-103 (apr eval coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender ALB-088 + Chen et al. (2021) HumanEval pass@k
//!
//! Run with: cargo run --example cli_eval_pass_at_k_temperature_pairing
//!
//! Added by PMAT-103 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MIN_TEMPERATURE_FOR_K_GT_1: f64 = 0.4;

#[derive(Debug, PartialEq)]
pub enum PairVerdict {
    Pass1Greedy, // k=1, T=0 — canonical greedy
    PassAtKWithSampling { k: u32, temperature: f64 },
    KGt1RequiresNonzeroTemperature { temperature: f64 },
    NegativeOrInvalidTemperature,
    KZeroRejected,
}

pub fn validate_pairing(k: u32, temperature: f64) -> PairVerdict {
    if k == 0 {
        return PairVerdict::KZeroRejected;
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return PairVerdict::NegativeOrInvalidTemperature;
    }
    if k == 1 && temperature == 0.0 {
        return PairVerdict::Pass1Greedy;
    }
    if k > 1 && temperature < MIN_TEMPERATURE_FOR_K_GT_1 {
        return PairVerdict::KGt1RequiresNonzeroTemperature { temperature };
    }
    PairVerdict::PassAtKWithSampling { k, temperature }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_eval_pass_at_k_temperature_pairing")?;

    let cases = [
        ("pass@1 greedy", 1, 0.0),
        ("pass@1 sample", 1, 0.8),
        ("pass@10 sample", 10, 0.8),
        ("pass@10 greedy (BAD)", 10, 0.0),
        ("pass@10 too cold (BAD)", 10, 0.2),
        ("k=0", 0, 0.8),
        ("nan temp", 1, f64::NAN),
        ("negative temp", 1, -0.1),
    ];
    for (label, k, t) in cases {
        println!(
            "{label:>25}  k={k} T={t:>5.2}  →  {:?}",
            validate_pairing(k, t)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairing_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn pass_1_greedy_canonical() {
        assert_eq!(validate_pairing(1, 0.0), PairVerdict::Pass1Greedy);
    }

    #[test]
    fn pass_1_with_temperature_classifies_as_sampling() {
        assert_eq!(
            validate_pairing(1, 0.8),
            PairVerdict::PassAtKWithSampling {
                k: 1,
                temperature: 0.8
            }
        );
    }

    #[test]
    fn k_gt_1_with_zero_temp_rejected() {
        // CRITICAL: k=10 at T=0 silently collapses to pass@1 — must reject.
        let v = validate_pairing(10, 0.0);
        assert!(matches!(
            v,
            PairVerdict::KGt1RequiresNonzeroTemperature { .. }
        ));
    }

    #[test]
    fn k_gt_1_with_too_cold_temp_rejected() {
        // Below the heuristic floor — too low to give meaningful diversity.
        let v = validate_pairing(10, 0.2);
        assert!(matches!(
            v,
            PairVerdict::KGt1RequiresNonzeroTemperature { .. }
        ));
    }

    #[test]
    fn k_gt_1_with_warm_temp_passes() {
        assert_eq!(
            validate_pairing(10, 0.8),
            PairVerdict::PassAtKWithSampling {
                k: 10,
                temperature: 0.8
            }
        );
    }

    #[test]
    fn k_zero_rejected() {
        assert_eq!(validate_pairing(0, 0.5), PairVerdict::KZeroRejected);
    }

    #[test]
    fn nan_temp_rejected() {
        assert_eq!(
            validate_pairing(1, f64::NAN),
            PairVerdict::NegativeOrInvalidTemperature
        );
    }

    #[test]
    fn negative_temp_rejected() {
        assert_eq!(
            validate_pairing(1, -0.1),
            PairVerdict::NegativeOrInvalidTemperature
        );
    }

    #[test]
    fn boundary_at_exactly_min_temp_passes() {
        // Conservative-pass at the floor.
        let v = validate_pairing(10, MIN_TEMPERATURE_FOR_K_GT_1);
        assert!(matches!(v, PairVerdict::PassAtKWithSampling { .. }));
    }
}
