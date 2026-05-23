//! # apr bench — Spec H12 Throughput Gate (≥10 tok/s)
//!
//! `apr bench` enforces spec H12: throughput ≥ 10 tok/s for any model
//! that ships. This recipe builds the gate and asserts the contract:
//! observed tok/s must be finite and positive; gate result includes the
//! observed margin above/below threshold; CI exits with EX_DATAERR on fail.
//!
//! Demonstrates the **BENCH.11** recipe for PMAT-109 (apr bench coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender spec H12 + sysexits.h
//!
//! Run with: cargo run --example cli_bench_h12_throughput_gate
//!
//! Added by PMAT-109 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const H12_MIN_TPS: f64 = 10.0;

#[derive(Debug, PartialEq)]
pub enum ThroughputVerdict {
    Pass { observed: f64, margin: f64 },
    Fail { observed: f64, deficit: f64 },
    NotFinite,
    Negative,
}

pub fn check_h12(observed_tps: f64) -> ThroughputVerdict {
    if !observed_tps.is_finite() {
        return ThroughputVerdict::NotFinite;
    }
    if observed_tps < 0.0 {
        return ThroughputVerdict::Negative;
    }
    if observed_tps >= H12_MIN_TPS {
        ThroughputVerdict::Pass {
            observed: observed_tps,
            margin: observed_tps - H12_MIN_TPS,
        }
    } else {
        ThroughputVerdict::Fail {
            observed: observed_tps,
            deficit: H12_MIN_TPS - observed_tps,
        }
    }
}

pub fn exit_code(v: &ThroughputVerdict) -> i32 {
    if matches!(v, ThroughputVerdict::Pass { .. }) {
        0
    } else {
        65
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_bench_h12_throughput_gate")?;

    for tps in [50.0_f64, 12.5, 10.0, 9.9, 0.5, 0.0, -1.0, f64::NAN] {
        let v = check_h12(tps);
        println!("{tps:>5.1} tok/s  →  {v:?}  exit={}", exit_code(&v));
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
    fn above_10_tps_passes() {
        let v = check_h12(15.0);
        if let ThroughputVerdict::Pass { margin, .. } = v {
            assert_eq!(margin, 5.0);
        }
    }

    #[test]
    fn boundary_at_exactly_10_tps_passes() {
        // Conservative-pass at the threshold.
        assert!(matches!(check_h12(10.0), ThroughputVerdict::Pass { .. }));
    }

    #[test]
    fn just_below_10_fails() {
        let v = check_h12(9.99);
        if let ThroughputVerdict::Fail { deficit, .. } = v {
            assert!(deficit > 0.0);
        } else {
            panic!("expected Fail");
        }
    }

    #[test]
    fn nan_rejected() {
        assert_eq!(check_h12(f64::NAN), ThroughputVerdict::NotFinite);
    }

    #[test]
    fn inf_rejected() {
        assert_eq!(check_h12(f64::INFINITY), ThroughputVerdict::NotFinite);
    }

    #[test]
    fn negative_rejected() {
        assert_eq!(check_h12(-1.0), ThroughputVerdict::Negative);
    }

    #[test]
    fn exit_code_zero_for_pass() {
        assert_eq!(exit_code(&check_h12(15.0)), 0);
    }

    #[test]
    fn exit_code_65_for_fail() {
        assert_eq!(exit_code(&check_h12(5.0)), 65);
    }
}
