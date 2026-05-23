//! # apr canary check — Pass/Fail Verdict
//!
//! `apr canary check <NAME>` runs the recorded prompt and compares
//! actual tokens against expected. This recipe builds the verdict
//! classifier with three classes: ExactMatch (every token matches),
//! WithinTolerance (some token disagreement but per-token logit drift
//! stays under tolerance), Diverged (a token disagreement coincides
//! with logit drift > tolerance).
//!
//! Demonstrates the **CANARY.5** recipe for PMAT-100 (apr canary check coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CANARY-002 + tolerance-band convention
//!
//! Run with: cargo run --example cli_canary_check_verdict
//!
//! Added by PMAT-100 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub enum CheckVerdict {
    ExactMatch,
    WithinTolerance {
        mismatches: usize,
    },
    Diverged {
        first_bad_position: usize,
        drift: f64,
    },
    LengthMismatch {
        expected_len: usize,
        actual_len: usize,
    },
}

pub fn check(
    expected: &[u32],
    actual: &[u32],
    per_token_drift: &[f64],
    tolerance: f64,
) -> CheckVerdict {
    if expected.len() != actual.len() {
        return CheckVerdict::LengthMismatch {
            expected_len: expected.len(),
            actual_len: actual.len(),
        };
    }
    let mut mismatches = 0;
    for (i, (e, a)) in expected.iter().zip(actual).enumerate() {
        if e != a {
            mismatches += 1;
            let drift = *per_token_drift.get(i).unwrap_or(&0.0);
            if drift > tolerance {
                return CheckVerdict::Diverged {
                    first_bad_position: i,
                    drift,
                };
            }
        }
    }
    if mismatches == 0 {
        CheckVerdict::ExactMatch
    } else {
        CheckVerdict::WithinTolerance { mismatches }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_canary_check_verdict")?;

    let expected = vec![19, 4, 12, 8];
    let actual_match = vec![19, 4, 12, 8];
    let actual_close = vec![19, 4, 12, 9]; // last token differs but drift small
    let actual_diverged = vec![19, 4, 99, 8]; // big mismatch, big drift

    let drift_close = vec![0.001, 0.001, 0.001, 0.05];
    let drift_diverged = vec![0.001, 0.001, 1.5, 0.001];

    println!(
        "exact:     {:?}",
        check(&expected, &actual_match, &drift_close, 0.1)
    );
    println!(
        "close:     {:?}",
        check(&expected, &actual_close, &drift_close, 0.1)
    );
    println!(
        "diverged:  {:?}",
        check(&expected, &actual_diverged, &drift_diverged, 0.1)
    );
    println!(
        "len mism:  {:?}",
        check(&expected, &actual_close[..2], &drift_close, 0.1)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identical_tokens_yield_exact_match() {
        let v = check(&[1, 2, 3], &[1, 2, 3], &[0.0, 0.0, 0.0], 0.1);
        assert_eq!(v, CheckVerdict::ExactMatch);
    }

    #[test]
    fn drift_within_tolerance_yields_within_tolerance() {
        let v = check(&[1, 2, 3], &[1, 2, 9], &[0.0, 0.0, 0.05], 0.1);
        assert_eq!(v, CheckVerdict::WithinTolerance { mismatches: 1 });
    }

    #[test]
    fn drift_above_tolerance_short_circuits_to_diverged() {
        let v = check(&[1, 2, 3], &[1, 99, 3], &[0.0, 1.5, 0.0], 0.1);
        if let CheckVerdict::Diverged {
            first_bad_position, ..
        } = v
        {
            assert_eq!(first_bad_position, 1);
        } else {
            panic!("expected Diverged");
        }
    }

    #[test]
    fn length_mismatch_short_circuits_before_per_token_check() {
        let v = check(&[1, 2, 3], &[1, 2], &[0.0, 0.0], 0.1);
        assert_eq!(
            v,
            CheckVerdict::LengthMismatch {
                expected_len: 3,
                actual_len: 2
            }
        );
    }

    #[test]
    fn missing_drift_treated_as_zero() {
        // If per_token_drift array is shorter than tokens, missing positions
        // default to 0.0 — test covers WithinTolerance result.
        let v = check(&[1, 2, 3], &[1, 2, 9], &[0.0], 0.1);
        assert_eq!(v, CheckVerdict::WithinTolerance { mismatches: 1 });
    }

    #[test]
    fn empty_expected_yields_exact_match_with_empty_actual() {
        let v = check(&[], &[], &[], 0.1);
        assert_eq!(v, CheckVerdict::ExactMatch);
    }
}
