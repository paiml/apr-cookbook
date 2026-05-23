//! # Contracts-Macros Pre/Post-Condition Envelope
//!
//! Design-by-contract attaches preconditions (caller must satisfy) and
//! postconditions (callee guarantees) to a function. This recipe builds
//! the envelope as plain functions (so the test harness can exercise
//! both arms without the macro), then shows the `#[contract]` macro
//! wraps the same logic in production builds.
//!
//! Demonstrates the **CM.5** recipe for PMAT-122 (contracts-macros coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Meyer, B. (1992). Applying "Design by Contract". IEEE Computer 25(10).
//!
//! Run with: cargo run --example contracts_macros_pre_post_envelope
//!
//! Added by PMAT-122 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use provable_contracts_macros::contract;

#[derive(Debug, PartialEq)]
pub enum CheckVerdict {
    Ok,
    PreconditionFailed { reason: &'static str },
    PostconditionFailed { reason: &'static str },
}

pub fn pre_sqrt_input(x: f64) -> CheckVerdict {
    if !x.is_finite() {
        return CheckVerdict::PreconditionFailed {
            reason: "input must be finite",
        };
    }
    if x < 0.0 {
        return CheckVerdict::PreconditionFailed {
            reason: "input must be ≥ 0",
        };
    }
    CheckVerdict::Ok
}

pub fn post_sqrt_output(input: f64, output: f64) -> CheckVerdict {
    if !output.is_finite() {
        return CheckVerdict::PostconditionFailed {
            reason: "output must be finite",
        };
    }
    if output < 0.0 {
        return CheckVerdict::PostconditionFailed {
            reason: "output must be ≥ 0",
        };
    }
    let reconstructed = output * output;
    let tolerance = (input.abs() + 1.0) * 1e-9;
    if (reconstructed - input).abs() > tolerance {
        return CheckVerdict::PostconditionFailed {
            reason: "output² must equal input within tolerance",
        };
    }
    CheckVerdict::Ok
}

#[contract("test-sqrt-v1", equation = "checked_sqrt")]
pub fn checked_sqrt(x: f64) -> f64 {
    x.sqrt()
}

pub fn invoke_with_envelope(x: f64) -> std::result::Result<f64, CheckVerdict> {
    match pre_sqrt_input(x) {
        CheckVerdict::Ok => {}
        v => return Err(v),
    }
    let y = checked_sqrt(x);
    match post_sqrt_output(x, y) {
        CheckVerdict::Ok => Ok(y),
        v => Err(v),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_pre_post_envelope")?;

    for x in [4.0, 0.0, 100.0, -1.0, f64::NAN, f64::INFINITY] {
        println!("checked_sqrt({x}) → {:?}", invoke_with_envelope(x));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_input_passes_both_pre_and_post() {
        let r = invoke_with_envelope(4.0);
        assert!(r.is_ok());
        assert!((r.unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn negative_caught_by_precondition() {
        let r = invoke_with_envelope(-1.0);
        assert!(matches!(r, Err(CheckVerdict::PreconditionFailed { .. })));
    }

    #[test]
    fn nan_caught_by_precondition() {
        let r = invoke_with_envelope(f64::NAN);
        assert!(matches!(r, Err(CheckVerdict::PreconditionFailed { .. })));
    }

    #[test]
    fn pre_check_in_isolation() {
        assert_eq!(pre_sqrt_input(4.0), CheckVerdict::Ok);
        assert!(matches!(
            pre_sqrt_input(-1.0),
            CheckVerdict::PreconditionFailed { .. }
        ));
    }

    #[test]
    fn post_check_in_isolation() {
        assert_eq!(post_sqrt_output(4.0, 2.0), CheckVerdict::Ok);
        // sqrt(4) ≠ 5 → post fails
        assert!(matches!(
            post_sqrt_output(4.0, 5.0),
            CheckVerdict::PostconditionFailed { .. }
        ));
    }

    #[test]
    fn post_rejects_nan_output() {
        assert!(matches!(
            post_sqrt_output(4.0, f64::NAN),
            CheckVerdict::PostconditionFailed { .. }
        ));
    }

    #[test]
    fn zero_passes_both() {
        let r = invoke_with_envelope(0.0);
        assert_eq!(r.unwrap(), 0.0);
    }

    #[test]
    fn infinity_caught_by_precondition() {
        // Infinity is non-finite, so pre rejects.
        assert!(matches!(
            invoke_with_envelope(f64::INFINITY),
            Err(CheckVerdict::PreconditionFailed { .. })
        ));
    }
}
