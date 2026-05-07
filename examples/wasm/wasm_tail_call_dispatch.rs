//! # WASM Tail Call Dispatch
//!
//! Validate `return_call` tail-call dispatch: target function's
//! signature must match caller's return signature; returns categorical
//! validation result.
//!
//! Demonstrates the **WASM.X** recipe for PMAT-214 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly tail-calls proposal (Phase 4); engine V8/SM
//!  return-call lowering.
//!
//! Run with: cargo run --example wasm_tail_call_dispatch
//!
//! Added by PMAT-214 (catalog 1549→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TailCallVerdict {
    Valid,
    SigMismatch,
    NoSuchTarget,
    InvalidConfig,
}

/// Functions: (params: Vec<&str>, results: Vec<&str>) where strings are valtype.
pub fn validate(
    functions: &[(Vec<&str>, Vec<&str>)],
    caller_idx: u32,
    callee_idx: u32,
) -> TailCallVerdict {
    if functions.is_empty() {
        return TailCallVerdict::InvalidConfig;
    }
    let cidx = caller_idx as usize;
    let calidx = callee_idx as usize;
    if cidx >= functions.len() || calidx >= functions.len() {
        return TailCallVerdict::NoSuchTarget;
    }
    let (_, caller_results) = &functions[cidx];
    let (_, callee_results) = &functions[calidx];
    if caller_results != callee_results {
        return TailCallVerdict::SigMismatch;
    }
    TailCallVerdict::Valid
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_tail_call_dispatch")?;

    let funcs = vec![
        (vec!["i32"], vec!["i32"]),
        (vec!["i64"], vec!["i32"]),
        (vec!["i32"], vec!["i64"]),
    ];
    println!("compatible: {:?}", validate(&funcs, 0, 1));
    println!("mismatch: {:?}", validate(&funcs, 0, 2));
    println!("oob: {:?}", validate(&funcs, 0, 5));
    println!("invalid: {:?}", validate(&[], 0, 0));
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
        assert_eq!(validate(&[], 0, 0), TailCallVerdict::InvalidConfig);
    }

    #[test]
    fn caller_oob() {
        let funcs = vec![(vec!["i32"], vec!["i32"])];
        assert_eq!(validate(&funcs, 5, 0), TailCallVerdict::NoSuchTarget);
    }

    #[test]
    fn callee_oob() {
        let funcs = vec![(vec!["i32"], vec!["i32"])];
        assert_eq!(validate(&funcs, 0, 5), TailCallVerdict::NoSuchTarget);
    }

    #[test]
    fn matching_results_valid() {
        let funcs = vec![(vec!["i32"], vec!["i32"]), (vec!["i64"], vec!["i32"])];
        assert_eq!(validate(&funcs, 0, 1), TailCallVerdict::Valid);
    }

    #[test]
    fn mismatched_results_invalid() {
        let funcs = vec![(vec!["i32"], vec!["i32"]), (vec!["i32"], vec!["i64"])];
        assert_eq!(validate(&funcs, 0, 1), TailCallVerdict::SigMismatch);
    }

    #[test]
    fn deterministic() {
        let funcs = vec![(vec!["i32"], vec!["i32"])];
        let r1 = validate(&funcs, 0, 0);
        let r2 = validate(&funcs, 0, 0);
        assert_eq!(r1, r2);
    }

    #[test]
    fn self_call_valid() {
        let funcs = vec![(vec!["i32"], vec!["i32"])];
        assert_eq!(validate(&funcs, 0, 0), TailCallVerdict::Valid);
    }

    #[test]
    fn empty_results_match_empty() {
        let funcs = vec![(vec!["i32"], vec![]), (vec!["i64"], vec![])];
        assert_eq!(validate(&funcs, 0, 1), TailCallVerdict::Valid);
    }

    #[test]
    fn multiple_results_match_required() {
        let funcs = vec![(vec![], vec!["i32", "i64"]), (vec![], vec!["i32", "i64"])];
        assert_eq!(validate(&funcs, 0, 1), TailCallVerdict::Valid);
    }

    #[test]
    fn multiple_results_mismatch_invalid() {
        let funcs = vec![(vec![], vec!["i32", "i64"]), (vec![], vec!["i64", "i32"])];
        assert_eq!(validate(&funcs, 0, 1), TailCallVerdict::SigMismatch);
    }

    #[test]
    fn many_functions_handled() {
        let funcs: Vec<(Vec<&str>, Vec<&str>)> =
            (0..30).map(|_| (vec!["i32"], vec!["i32"])).collect();
        assert_eq!(validate(&funcs, 0, 29), TailCallVerdict::Valid);
    }

    #[test]
    fn different_param_count_ok() {
        // Tail calls don't care about caller-callee params; only results.
        let funcs = vec![
            (vec!["i32"], vec!["i64"]),
            (vec!["i32", "i32", "i32"], vec!["i64"]),
        ];
        assert_eq!(validate(&funcs, 0, 1), TailCallVerdict::Valid);
    }
}
