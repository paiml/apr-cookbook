//! # apr compare-hf — Threshold Gate (CI Failure Semantics)
//!
//! `apr compare-hf` ships a CI-ready threshold gate: any per-tensor max-abs
//! delta above `--threshold` produces a non-zero exit code. This recipe
//! models the gate as a pure function so a CI pipeline can preview the
//! decision (PASS / FAIL with the offending tensor names) before invoking
//! the binary.
//!
//! Demonstrates the **CMPHF.2** recipe for PMAT-088 (apr compare-hf coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CMPHF-002 + IEEE 754-2019 (subnormal handling)
//!
//! Run with: cargo run --example cli_compare_hf_threshold_gate
//!
//! Added by PMAT-088 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
struct TensorDelta {
    name: &'static str,
    max_abs_delta: f64,
}

#[derive(Debug, PartialEq)]
enum GateVerdict<'a> {
    Pass,
    Fail { offenders: Vec<&'a TensorDelta> },
}

fn gate(deltas: &[TensorDelta], threshold: f64) -> GateVerdict<'_> {
    let offenders: Vec<&TensorDelta> = deltas
        .iter()
        .filter(|d| d.max_abs_delta > threshold)
        .collect();
    if offenders.is_empty() {
        GateVerdict::Pass
    } else {
        GateVerdict::Fail { offenders }
    }
}

fn exit_code(v: &GateVerdict<'_>) -> i32 {
    match v {
        GateVerdict::Pass => 0,
        GateVerdict::Fail { .. } => 65, // EX_DATAERR per sysexits.h
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_compare_hf_threshold_gate")?;

    let deltas = [
        TensorDelta {
            name: "model.embed_tokens.weight",
            max_abs_delta: 1.2e-7,
        },
        TensorDelta {
            name: "model.layers.0.q_proj.weight",
            max_abs_delta: 4.4e-6,
        },
        TensorDelta {
            name: "model.layers.0.k_proj.weight",
            max_abs_delta: 9.1e-4,
        },
        TensorDelta {
            name: "lm_head.weight",
            max_abs_delta: 2.0e-7,
        },
    ];

    for thr in [1e-3, 1e-5, 1e-7] {
        let v = gate(&deltas, thr);
        println!(
            "threshold={thr:e}  →  exit_code={}  verdict={v:?}",
            exit_code(&v)
        );
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
    fn all_below_threshold_passes() {
        let deltas = [
            TensorDelta {
                name: "a",
                max_abs_delta: 1e-9,
            },
            TensorDelta {
                name: "b",
                max_abs_delta: 1e-10,
            },
        ];
        let v = gate(&deltas, 1e-6);
        assert_eq!(v, GateVerdict::Pass);
        assert_eq!(exit_code(&v), 0);
    }

    #[test]
    fn one_above_threshold_fails() {
        let deltas = [
            TensorDelta {
                name: "good",
                max_abs_delta: 1e-9,
            },
            TensorDelta {
                name: "bad",
                max_abs_delta: 1e-3,
            },
        ];
        let v = gate(&deltas, 1e-6);
        match v {
            GateVerdict::Fail { offenders } => {
                assert_eq!(offenders.len(), 1);
                assert_eq!(offenders[0].name, "bad");
            }
            _ => panic!("expected Fail"),
        }
    }

    #[test]
    fn fail_returns_nonzero_exit_code() {
        // Without a non-zero exit code, CI silently passes broken model converts.
        let deltas = [TensorDelta {
            name: "x",
            max_abs_delta: 1.0,
        }];
        let v = gate(&deltas, 1e-6);
        assert_ne!(exit_code(&v), 0);
    }

    #[test]
    fn delta_equal_to_threshold_passes() {
        // Boundary: equality is conservative-pass so well-tuned thresholds
        // don't flip on numerically identical tensors.
        let deltas = [TensorDelta {
            name: "x",
            max_abs_delta: 1e-6,
        }];
        let v = gate(&deltas, 1e-6);
        assert_eq!(v, GateVerdict::Pass);
    }
}
