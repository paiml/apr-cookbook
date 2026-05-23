//! # apr cbtop — CI Threshold Gate
//!
//! `apr cbtop --ci --throughput <T> --brick-score <S>` runs the
//! ComputeBrick monitor in CI mode: any measurement below either
//! threshold produces exit code 1. This recipe models the gate as a
//! pure function so a CI pipeline can preview the verdict before
//! invoking the binary.
//!
//! Demonstrates the **CBTOP.3** recipe for PMAT-094 (apr cbtop coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CBTOP-CI-001 + sysexits.h conventions
//!
//! Run with: cargo run --example cli_cbtop_ci_threshold_gate
//!
//! Added by PMAT-094 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CbtopMeasurement {
    pub throughput_tps: f64,
    pub brick_score: f64, // 0..=100
}

#[derive(Debug, Clone, PartialEq)]
pub enum CiVerdict {
    Pass,
    BelowThroughput { observed: f64, required: f64 },
    BelowBrickScore { observed: f64, required: f64 },
    BelowBoth,
}

pub fn ci_gate(
    m: CbtopMeasurement,
    min_throughput: Option<f64>,
    min_brick: Option<f64>,
) -> CiVerdict {
    let bad_t = min_throughput.is_some_and(|t| m.throughput_tps < t);
    let bad_b = min_brick.is_some_and(|b| m.brick_score < b);
    match (bad_t, bad_b) {
        (true, true) => CiVerdict::BelowBoth,
        (true, false) => CiVerdict::BelowThroughput {
            observed: m.throughput_tps,
            required: min_throughput.unwrap(),
        },
        (false, true) => CiVerdict::BelowBrickScore {
            observed: m.brick_score,
            required: min_brick.unwrap(),
        },
        _ => CiVerdict::Pass,
    }
}

pub fn exit_code(v: &CiVerdict) -> i32 {
    i32::from(!matches!(v, CiVerdict::Pass))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_cbtop_ci_threshold_gate")?;

    let cases = [
        (
            "happy",
            CbtopMeasurement {
                throughput_tps: 85.0,
                brick_score: 92.0,
            },
        ),
        (
            "low tps",
            CbtopMeasurement {
                throughput_tps: 10.0,
                brick_score: 90.0,
            },
        ),
        (
            "low brick",
            CbtopMeasurement {
                throughput_tps: 80.0,
                brick_score: 50.0,
            },
        ),
        (
            "both bad",
            CbtopMeasurement {
                throughput_tps: 5.0,
                brick_score: 20.0,
            },
        ),
    ];
    for (label, m) in cases {
        let v = ci_gate(m, Some(50.0), Some(80.0));
        println!("{label:>10}  →  {v:?}  exit={}", exit_code(&v));
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
    fn happy_measurement_passes() {
        let m = CbtopMeasurement {
            throughput_tps: 100.0,
            brick_score: 95.0,
        };
        assert_eq!(ci_gate(m, Some(50.0), Some(80.0)), CiVerdict::Pass);
    }

    #[test]
    fn no_thresholds_always_passes() {
        // CI mode with no thresholds is effectively report-only.
        let m = CbtopMeasurement {
            throughput_tps: 0.0,
            brick_score: 0.0,
        };
        assert_eq!(ci_gate(m, None, None), CiVerdict::Pass);
    }

    #[test]
    fn below_throughput_only() {
        let m = CbtopMeasurement {
            throughput_tps: 10.0,
            brick_score: 95.0,
        };
        let v = ci_gate(m, Some(50.0), Some(80.0));
        assert!(matches!(v, CiVerdict::BelowThroughput { .. }));
    }

    #[test]
    fn below_both_collapses_to_below_both() {
        // Don't double-report — single combined verdict for double failure.
        let m = CbtopMeasurement {
            throughput_tps: 5.0,
            brick_score: 20.0,
        };
        assert_eq!(ci_gate(m, Some(50.0), Some(80.0)), CiVerdict::BelowBoth);
    }

    #[test]
    fn boundary_at_exact_threshold_passes() {
        // Conservative-pass at equality (matches CI gate convention).
        let m = CbtopMeasurement {
            throughput_tps: 50.0,
            brick_score: 80.0,
        };
        assert_eq!(ci_gate(m, Some(50.0), Some(80.0)), CiVerdict::Pass);
    }

    #[test]
    fn nonzero_exit_for_any_failure() {
        for v in [
            CiVerdict::BelowThroughput {
                observed: 1.0,
                required: 2.0,
            },
            CiVerdict::BelowBrickScore {
                observed: 1.0,
                required: 2.0,
            },
            CiVerdict::BelowBoth,
        ] {
            assert_ne!(exit_code(&v), 0, "verdict {v:?} must exit nonzero");
        }
    }
}
