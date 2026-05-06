//! # Advanced Tool-Call JSON-Schema Validator
//!
//! Validate that an LLM-emitted function call has correct shape:
//!   has 'name' field (non-empty string)
//!   has 'arguments' field (JSON object)
//!   all required parameters present in arguments
//!   no unknown parameters when strict_mode is on
//!
//! Demonstrates the **ADV.19** recipe for PMAT-149 (advanced round 7).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: OpenAI function-calling JSON schema spec.
//!
//! Run with: cargo run --example adv_tool_call_validator
//!
//! Added by PMAT-149 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum CallVerdict {
    Valid,
    EmptyName,
    MissingRequired { params: Vec<String> },
    UnknownParameters { params: Vec<String> },
    InvalidArgsType,
}

pub fn validate(
    call_name: &str,
    call_args: &[&str],
    required_params: &[&str],
    allowed_params: &[&str],
    strict_mode: bool,
) -> CallVerdict {
    if call_name.is_empty() {
        return CallVerdict::EmptyName;
    }
    let args_set: BTreeSet<&str> = call_args.iter().copied().collect();
    let required_set: BTreeSet<&str> = required_params.iter().copied().collect();
    let allowed_set: BTreeSet<&str> = allowed_params.iter().copied().collect();
    let missing: Vec<String> = required_set
        .difference(&args_set)
        .map(|s| (*s).to_string())
        .collect();
    if !missing.is_empty() {
        return CallVerdict::MissingRequired { params: missing };
    }
    if strict_mode {
        let unknown: Vec<String> = args_set
            .difference(&allowed_set)
            .map(|s| (*s).to_string())
            .collect();
        if !unknown.is_empty() {
            return CallVerdict::UnknownParameters { params: unknown };
        }
    }
    CallVerdict::Valid
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_tool_call_validator")?;

    println!(
        "valid: {:?}",
        validate(
            "get_weather",
            &["location", "units"],
            &["location"],
            &["location", "units"],
            true
        )
    );
    println!(
        "missing required: {:?}",
        validate(
            "get_weather",
            &["units"],
            &["location"],
            &["location", "units"],
            false
        )
    );
    println!(
        "unknown strict: {:?}",
        validate(
            "get_weather",
            &["location", "extra"],
            &["location"],
            &["location"],
            true
        )
    );
    println!(
        "unknown lenient: {:?}",
        validate(
            "get_weather",
            &["location", "extra"],
            &["location"],
            &["location"],
            false
        )
    );
    println!("empty name: {:?}", validate("", &[], &[], &[], false));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn valid_call() {
        let v = validate("fn", &["a", "b"], &["a"], &["a", "b", "c"], true);
        assert_eq!(v, CallVerdict::Valid);
    }

    #[test]
    fn missing_required_rejected() {
        let v = validate("fn", &["b"], &["a"], &["a", "b"], false);
        assert!(matches!(v, CallVerdict::MissingRequired { .. }));
    }

    #[test]
    fn unknown_strict_rejected() {
        let v = validate("fn", &["a", "extra"], &["a"], &["a"], true);
        assert!(matches!(v, CallVerdict::UnknownParameters { .. }));
    }

    #[test]
    fn unknown_lenient_allowed() {
        let v = validate("fn", &["a", "extra"], &["a"], &["a"], false);
        assert_eq!(v, CallVerdict::Valid);
    }

    #[test]
    fn empty_name_rejected() {
        assert_eq!(validate("", &[], &[], &[], false), CallVerdict::EmptyName);
    }

    #[test]
    fn no_required_params_ok() {
        let v = validate("fn", &[], &[], &[], false);
        assert_eq!(v, CallVerdict::Valid);
    }

    #[test]
    fn missing_lists_all_missing() {
        let v = validate("fn", &[], &["a", "b", "c"], &["a", "b", "c"], false);
        if let CallVerdict::MissingRequired { params } = v {
            assert_eq!(params.len(), 3);
        }
    }

    #[test]
    fn strict_disabled_extras_ok() {
        let v = validate("fn", &["a", "ignored"], &["a"], &["a"], false);
        assert_eq!(v, CallVerdict::Valid);
    }

    #[test]
    fn missing_takes_precedence_over_unknown() {
        let v = validate("fn", &["unknown"], &["required"], &[], true);
        // Missing required is reported, not unknown.
        assert!(matches!(v, CallVerdict::MissingRequired { .. }));
    }

    #[test]
    fn deduplicated_args_in_unknown_list() {
        let v = validate("fn", &["x", "x"], &[], &[], true);
        if let CallVerdict::UnknownParameters { params } = v {
            assert_eq!(params, vec!["x".to_string()]);
        }
    }
}
