//! # Recipe: FP8 Lint — SM Capability Gate
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr fp8-lint --observation-file observation.json` (capability path)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the SM-capability decision tree inside `apr fp8-lint`. E4M3
//! has native hardware support starting on Hopper (SM 9.0). On SM 8.9
//! (Ada) FP8 is software-emulated through Transformer Engine fallbacks, and
//! older arches are flat-out unsupported. The lint must distinguish the
//! three states with three distinct verdicts: **pass**, **warn (emulated)**,
//! and **error (unsupported)**.
//!
//! ## Run Command
//! ```bash
//! cargo run --example fp8_lint_capability_gate
//! ```
//!
//! ## References
//! - NVIDIA Hopper Architecture Whitepaper (SM 9.0).
//! - NVIDIA Ada Lovelace Architecture Whitepaper (SM 8.9 emulation path).
//!
//! Added by PMAT-089 (expand-cookbooks followup — quantization lint coverage).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityVerdict {
    NativeSupported,                // SM >= 9.0
    EmulatedFallback(&'static str), // SM 8.9 — emulated, warn
    Unsupported(String),            // < 8.9 — hard error
}

pub fn check_capability(obs: &Value) -> CapabilityVerdict {
    let major = obs
        .pointer("/capability/sm_major")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let minor = obs
        .pointer("/capability/sm_minor")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let combined = major * 10 + minor;
    match combined {
        90.. => CapabilityVerdict::NativeSupported,
        89 => CapabilityVerdict::EmulatedFallback(
            "SM 8.9 uses Transformer Engine software emulation — slower, higher error",
        ),
        n => CapabilityVerdict::Unsupported(format!(
            "SM {major}.{minor} (= {n}) has no FP8 path; minimum is 8.9 emulated, 9.0 native"
        )),
    }
}

fn observation_with_sm(major: u64, minor: u64) -> Value {
    json!({
        "schema_version": 1,
        "format": "E4M3",
        "capability": { "sm_major": major, "sm_minor": minor },
        "frobenius_rel_err": 0.012,
        "saturation_count": 0,
        "scale_factor": 448.0
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("fp8_lint_capability_gate")?;

    for (major, minor, label) in [
        (9, 0, "H100"),
        (8, 9, "RTX 4090"),
        (8, 6, "RTX 3090"),
        (7, 5, "T4"),
    ] {
        let obs = observation_with_sm(major, minor);
        let v = check_capability(&obs);
        println!("{label:>10} (SM {major}.{minor}) → {v:?}");
    }

    ctx.record_string_metric("verdict", "matrix_printed");
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn hopper_sm_90_is_native() {
        assert_eq!(
            check_capability(&observation_with_sm(9, 0)),
            CapabilityVerdict::NativeSupported
        );
    }

    #[test]
    fn ada_sm_89_is_emulated_warn() {
        // SM 8.9 is the emulated fallback — pass-with-warn, not pass.
        assert!(matches!(
            check_capability(&observation_with_sm(8, 9)),
            CapabilityVerdict::EmulatedFallback(_)
        ));
    }

    #[test]
    fn ampere_sm_86_is_unsupported_error() {
        // RTX 3090 / A100 — no FP8 path at all, not even emulated.
        assert!(matches!(
            check_capability(&observation_with_sm(8, 6)),
            CapabilityVerdict::Unsupported(_)
        ));
    }

    #[test]
    fn turing_sm_75_is_unsupported_error() {
        assert!(matches!(
            check_capability(&observation_with_sm(7, 5)),
            CapabilityVerdict::Unsupported(_)
        ));
    }

    #[test]
    fn future_sm_100_is_native() {
        // Forward compatibility — Blackwell+ should pass without code change.
        assert_eq!(
            check_capability(&observation_with_sm(10, 0)),
            CapabilityVerdict::NativeSupported
        );
    }
}
