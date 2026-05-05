//! # Contracts-Macros Env Key Convention
//!
//! The `#[contract("name", equation = "eq")]` macro generates a lookup key
//! `CONTRACT_<NAME_UPPER>_<EQUATION_UPPER>` (with hyphens and dots replaced
//! by underscores) that the consuming crate's `build.rs` sets from its
//! `binding.yaml`. This recipe demonstrates the canonical key-naming
//! convention so build-script authors and macro users can reason about
//! the lookup independently of the macro firing.
//!
//! Demonstrates the **CM.2** recipe per
//! `docs/specifications/expand-cookbooks/subcrate-coverage.md` —
//! the env-key naming contract that the macro pivots on.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Findler, R. B. & Felleisen, M. (2002). Contracts for higher-order functions. ICFP. DOI: 10.1145/581478.581484
//!
//! Run with: cargo run --example contracts_macros_env_key_convention
//!
//! Added by PMAT-084 (expand-cookbooks: aprender-contracts-macros coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

/// Recreates the key-derivation logic the proc-macro applies internally
/// (per the `make_env_key` function in
/// `aprender-contracts-macros-0.31.2/src/lib.rs`). Hyphens and dots become
/// underscores; the key uppercases the contract + equation name.
fn make_env_key(contract: &str, equation: &str) -> String {
    let contract_part = contract.to_uppercase().replace(['-', '.'], "_");
    let equation_part = equation.to_uppercase().replace(['-', '.'], "_");
    format!("CONTRACT_{contract_part}_{equation_part}")
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_env_key_convention")?;

    let cases = [
        ("rmsnorm-kernel-v1", "rmsnorm"),
        ("attention-kernel-v1", "scaled_dot_product"),
        ("gated-delta-net-v1", "decay"),
        ("v1.0", "eq.1"),
    ];

    println!("Contract → env key derivation (proc-macro contract);");
    for (contract, equation) in &cases {
        let key = make_env_key(contract, equation);
        println!("  {contract:30} {equation:25} → {key}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convention_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn matches_macro_internal_test_cases() {
        // These three cases mirror the proc-macro's own internal test_make_env_key.
        // Drift here means the macro's key derivation has changed.
        assert_eq!(
            make_env_key("rmsnorm-kernel-v1", "rmsnorm"),
            "CONTRACT_RMSNORM_KERNEL_V1_RMSNORM"
        );
        assert_eq!(
            make_env_key("attention-kernel-v1", "scaled_dot_product"),
            "CONTRACT_ATTENTION_KERNEL_V1_SCALED_DOT_PRODUCT"
        );
        assert_eq!(
            make_env_key("gated-delta-net-v1", "decay"),
            "CONTRACT_GATED_DELTA_NET_V1_DECAY"
        );
    }

    #[test]
    fn dots_collapse_like_hyphens() {
        assert_eq!(make_env_key("v1.0", "eq.1"), "CONTRACT_V1_0_EQ_1");
    }
}
