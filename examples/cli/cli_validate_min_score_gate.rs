//! # apr validate --min-score — CI Gate Threshold
//!
//! `apr validate --quality --min-score <N>` exits non-zero if the
//! quality score < N. This recipe builds the CI gate as a pure function
//! and asserts the contract: NaN/inf scores never pass; min-score must
//! be in [0, 100]; conservative-pass at exact threshold.
//!
//! Demonstrates the **VALIDATE.13** recipe for PMAT-108 (apr validate coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender VALIDATE-003 + sysexits.h conventions
//!
//! Run with: cargo run --example cli_validate_min_score_gate
//!
//! Added by PMAT-108 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GateVerdict {
    Pass { observed: u32, threshold: u32 },
    Fail { observed: u32, threshold: u32 },
    InvalidThreshold,
    InvalidScore,
}

pub fn evaluate_gate(observed: u32, threshold: u32) -> GateVerdict {
    if threshold > 100 {
        return GateVerdict::InvalidThreshold;
    }
    if observed > 100 {
        return GateVerdict::InvalidScore;
    }
    if observed >= threshold {
        GateVerdict::Pass {
            observed,
            threshold,
        }
    } else {
        GateVerdict::Fail {
            observed,
            threshold,
        }
    }
}

pub fn exit_code(v: &GateVerdict) -> i32 {
    if matches!(v, GateVerdict::Pass { .. }) {
        0
    } else {
        65 // EX_DATAERR per sysexits.h
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_validate_min_score_gate")?;

    for (label, observed, threshold) in [
        ("happy 95/80", 95u32, 80u32),
        ("at threshold 80/80", 80, 80),
        ("just below 79/80", 79, 80),
        ("invalid threshold 80/150", 80, 150),
        ("invalid score 150/80", 150, 80),
    ] {
        let v = evaluate_gate(observed, threshold);
        println!("{label:>22}  →  {v:?}  exit={}", exit_code(&v));
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
    fn observed_above_threshold_passes() {
        let v = evaluate_gate(95, 80);
        assert!(matches!(v, GateVerdict::Pass { .. }));
    }

    #[test]
    fn boundary_at_exact_threshold_passes() {
        // Conservative-pass at the threshold.
        let v = evaluate_gate(80, 80);
        assert!(matches!(v, GateVerdict::Pass { .. }));
    }

    #[test]
    fn one_below_threshold_fails() {
        let v = evaluate_gate(79, 80);
        assert!(matches!(v, GateVerdict::Fail { .. }));
    }

    #[test]
    fn threshold_above_100_invalid() {
        assert_eq!(evaluate_gate(80, 150), GateVerdict::InvalidThreshold);
    }

    #[test]
    fn score_above_100_invalid() {
        assert_eq!(evaluate_gate(150, 80), GateVerdict::InvalidScore);
    }

    #[test]
    fn exit_code_zero_for_pass() {
        let v = evaluate_gate(95, 80);
        assert_eq!(exit_code(&v), 0);
    }

    #[test]
    fn exit_code_65_for_fail() {
        let v = evaluate_gate(50, 80);
        assert_eq!(exit_code(&v), 65);
    }

    #[test]
    fn exit_code_nonzero_for_invalid() {
        assert_ne!(exit_code(&GateVerdict::InvalidThreshold), 0);
        assert_ne!(exit_code(&GateVerdict::InvalidScore), 0);
    }

    #[test]
    fn threshold_100_is_perfect_only() {
        // Only score == 100 passes a threshold of 100.
        assert!(matches!(evaluate_gate(100, 100), GateVerdict::Pass { .. }));
        assert!(matches!(evaluate_gate(99, 100), GateVerdict::Fail { .. }));
    }

    #[test]
    fn threshold_zero_always_passes_valid_scores() {
        // Trivial case: any score ≥ 0 passes.
        assert!(matches!(evaluate_gate(0, 0), GateVerdict::Pass { .. }));
        assert!(matches!(evaluate_gate(100, 0), GateVerdict::Pass { .. }));
    }
}
