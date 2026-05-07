//! # WASM Start Section Validate
//!
//! Validate a WASM module's start function: must take no parameters
//! and return no results, and reference an existing function index.
//! Returns categorical reason if invalid.
//!
//! Demonstrates the **WASM.X** recipe for PMAT-214 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly Core Specification §3.4.7 start function
//!  validation rules.
//!
//! Run with: cargo run --example wasm_start_section_validate
//!
//! Added by PMAT-214 (catalog 1549→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum StartVerdict {
    Valid,
    NoSuchFunction,
    HasParameters,
    HasResults,
    InvalidConfig,
}

/// Functions: (param_count, result_count). start_idx references this list.
pub fn validate(functions: &[(u32, u32)], start_idx: u32) -> StartVerdict {
    if functions.is_empty() {
        return StartVerdict::InvalidConfig;
    }
    let idx = start_idx as usize;
    if idx >= functions.len() {
        return StartVerdict::NoSuchFunction;
    }
    let (params, results) = functions[idx];
    if params != 0 {
        return StartVerdict::HasParameters;
    }
    if results != 0 {
        return StartVerdict::HasResults;
    }
    StartVerdict::Valid
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_start_section_validate")?;

    let funcs = [(0, 0), (1, 0), (0, 1)];
    println!("valid: {:?}", validate(&funcs, 0));
    println!("has-params: {:?}", validate(&funcs, 1));
    println!("has-results: {:?}", validate(&funcs, 2));
    println!("oob: {:?}", validate(&funcs, 5));
    println!("invalid: {:?}", validate(&[], 0));
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
    fn empty_functions_rejected() {
        assert_eq!(validate(&[], 0), StartVerdict::InvalidConfig);
    }

    #[test]
    fn no_such_function() {
        assert_eq!(validate(&[(0, 0)], 5), StartVerdict::NoSuchFunction);
    }

    #[test]
    fn valid_start() {
        assert_eq!(validate(&[(0, 0)], 0), StartVerdict::Valid);
    }

    #[test]
    fn has_parameters_rejected() {
        assert_eq!(validate(&[(1, 0)], 0), StartVerdict::HasParameters);
    }

    #[test]
    fn has_results_rejected() {
        assert_eq!(validate(&[(0, 1)], 0), StartVerdict::HasResults);
    }

    #[test]
    fn deterministic() {
        let r1 = validate(&[(0, 0)], 0);
        let r2 = validate(&[(0, 0)], 0);
        assert_eq!(r1, r2);
    }

    #[test]
    fn middle_function_can_be_start() {
        let funcs = [(1, 1), (0, 0), (2, 2)];
        assert_eq!(validate(&funcs, 1), StartVerdict::Valid);
    }

    #[test]
    fn many_functions_handled() {
        let funcs: Vec<(u32, u32)> = (0..30).map(|_| (0u32, 0u32)).collect();
        assert_eq!(validate(&funcs, 15), StartVerdict::Valid);
    }

    #[test]
    fn boundary_last_idx() {
        let funcs = [(0, 0), (0, 0)];
        assert_eq!(validate(&funcs, 1), StartVerdict::Valid);
    }

    #[test]
    fn boundary_first_idx() {
        assert_eq!(validate(&[(0, 0)], 0), StartVerdict::Valid);
    }

    #[test]
    fn high_param_count_rejected() {
        assert_eq!(validate(&[(100, 0)], 0), StartVerdict::HasParameters);
    }

    #[test]
    fn high_result_count_rejected() {
        assert_eq!(validate(&[(0, 100)], 0), StartVerdict::HasResults);
    }
}
