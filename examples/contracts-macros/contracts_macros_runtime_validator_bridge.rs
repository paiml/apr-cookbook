//! # Contracts-Macros + Runtime Validator Bridge
//!
//! Demonstrates the bridge between the compile-time `#[contract]` proc-macro
//! and the runtime YAML validator (`provable_contracts`/`aprender-contracts`,
//! the cookbook already uses this in `tests/contracts.rs`):
//!
//! - The macro asserts at COMPILE time that an algorithm-binding exists for
//!   the annotated function.
//! - The runtime validator asserts that every YAML contract file under
//!   `contracts/` is well-formed and has all required obligations bound.
//!
//! This recipe loads one of the cookbook's own contract YAMLs, walks the
//! obligations, and shows what the macro's `CONTRACT_<NAME>_<EQ>` env-var
//! key WOULD look like for each obligation in that contract.
//!
//! Demonstrates the **CM.3** recipe per
//! `docs/specifications/expand-cookbooks/subcrate-coverage.md` —
//! the YAML→key→macro authoring loop.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender-contracts-macros + provable_contracts crates v0.31.2 (companion runtime + compile-time halves of the contract enforcement story)
//!
//! Run with: cargo run --example contracts_macros_runtime_validator_bridge
//!
//! Added by PMAT-084 (expand-cookbooks: aprender-contracts-macros coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use provable_contracts::schema::parse_contract;
use std::path::PathBuf;

fn make_env_key(contract: &str, equation: &str) -> String {
    let contract_part = contract.to_uppercase().replace(['-', '.'], "_");
    let equation_part = equation.to_uppercase().replace(['-', '.'], "_");
    format!("CONTRACT_{contract_part}_{equation_part}")
}

fn contracts_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("contracts")
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_runtime_validator_bridge")?;

    let contract_path = contracts_dir().join("recipe-iiur-v1.yaml");
    let contract = parse_contract(&contract_path).map_err(|e| {
        apr_cookbook::CookbookError::Validation(format!("parse_contract failed: {e}"))
    })?;

    let contract_name = "recipe-iiur-v1";
    println!(
        "Loaded {} (version {}); deriving #[contract] env-keys for each equation:",
        contract_name, contract.metadata.version
    );

    for equation_name in contract.equations.keys() {
        let key = make_env_key(contract_name, equation_name);
        println!("  equation={equation_name:25} → env-key={key}");
    }

    println!();
    println!("To bind these algorithmically at compile time, the consuming");
    println!("crate's build.rs sets each env var to \"implemented\", and the");
    println!("annotated function gains compile-time enforcement of the");
    println!("contract's preconditions/postconditions via macro expansion.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn contract_has_at_least_one_equation() {
        let contract = parse_contract(&contracts_dir().join("recipe-iiur-v1.yaml"))
            .expect("parse_contract must succeed");
        assert!(
            !contract.equations.is_empty(),
            "recipe-iiur-v1 should declare at least one equation"
        );
    }

    #[test]
    fn env_key_derivation_matches_macro() {
        // Round-trip check: derived keys match the proc-macro's internal
        // make_env_key behavior (covered by contracts_macros_env_key_convention.rs).
        assert_eq!(
            make_env_key("recipe-iiur-v1", "isolation"),
            "CONTRACT_RECIPE_IIUR_V1_ISOLATION"
        );
    }
}
