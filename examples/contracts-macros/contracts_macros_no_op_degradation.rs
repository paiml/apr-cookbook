//! # Contracts-Macros No-Op Degradation
//!
//! When the consuming crate's `build.rs` does not export the
//! `CONTRACT_<NAME>_<EQ>` env vars, the `#[contract]` macro emits a
//! pass-through (no runtime check). This is by design: contract
//! checking is a build-time opt-in, not a hard runtime requirement.
//! This recipe codifies the predicate for "is contract checking
//! active for this build".
//!
//! Demonstrates the **CM.6** recipe for PMAT-122 (contracts-macros coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CM-001 + Rust proc-macro hygiene rules
//!
//! Run with: cargo run --example contracts_macros_no_op_degradation
//!
//! Added by PMAT-122 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use provable_contracts_macros::contract;

#[derive(Debug, PartialEq)]
pub enum ActivationVerdict {
    Active { contract_path: String },
    NoOpFallback,
    InvalidValue,
}

pub fn classify_env_value(v: Option<&str>) -> ActivationVerdict {
    match v {
        None | Some("") => ActivationVerdict::NoOpFallback,
        Some(s) if s == "0" || s.eq_ignore_ascii_case("false") => ActivationVerdict::NoOpFallback,
        Some("1") => ActivationVerdict::InvalidValue,
        Some(s) => ActivationVerdict::Active {
            contract_path: s.into(),
        },
    }
}

#[contract("test-passthrough-v1", equation = "noop")]
pub fn passthrough(x: i32) -> i32 {
    x
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_no_op_degradation")?;

    let cases = [
        None,
        Some(""),
        Some("0"),
        Some("FALSE"),
        Some("1"),
        Some("contracts/recipe-iiur-v1.yaml"),
    ];
    for c in cases {
        println!("{c:?}  →  {:?}", classify_env_value(c));
    }
    println!("passthrough(42) = {}", passthrough(42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn unset_env_falls_back_noop() {
        assert_eq!(classify_env_value(None), ActivationVerdict::NoOpFallback);
    }

    #[test]
    fn empty_string_falls_back_noop() {
        assert_eq!(
            classify_env_value(Some("")),
            ActivationVerdict::NoOpFallback
        );
    }

    #[test]
    fn explicit_zero_falls_back_noop() {
        assert_eq!(
            classify_env_value(Some("0")),
            ActivationVerdict::NoOpFallback
        );
    }

    #[test]
    fn case_insensitive_false_falls_back_noop() {
        assert_eq!(
            classify_env_value(Some("false")),
            ActivationVerdict::NoOpFallback
        );
        assert_eq!(
            classify_env_value(Some("FALSE")),
            ActivationVerdict::NoOpFallback
        );
        assert_eq!(
            classify_env_value(Some("False")),
            ActivationVerdict::NoOpFallback
        );
    }

    #[test]
    fn bare_one_invalid() {
        // "1" is ambiguous (boolean true vs path). The macro requires
        // an explicit contract path; reject "1" as a setup error.
        assert_eq!(
            classify_env_value(Some("1")),
            ActivationVerdict::InvalidValue
        );
    }

    #[test]
    fn explicit_path_activates() {
        let v = classify_env_value(Some("contracts/recipe-iiur-v1.yaml"));
        assert!(matches!(v, ActivationVerdict::Active { .. }));
    }

    #[test]
    fn passthrough_returns_input_unchanged() {
        // In no-op mode (cookbook build), the macro is a no-op.
        // The function body runs verbatim.
        for x in [-100, 0, 1, 42, 1000] {
            assert_eq!(passthrough(x), x);
        }
    }

    #[test]
    fn activation_carries_contract_path() {
        if let ActivationVerdict::Active { contract_path } =
            classify_env_value(Some("contracts/foo.yaml"))
        {
            assert_eq!(contract_path, "contracts/foo.yaml");
        } else {
            panic!("expected Active");
        }
    }
}
