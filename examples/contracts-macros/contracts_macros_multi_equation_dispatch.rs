//! # Contracts-Macros Multi-Equation Dispatch
//!
//! A single contract often has multiple named equations. The `#[contract]`
//! macro accepts an `equation = "..."` argument that selects which
//! equation's pre/postconditions are wired in for the annotated function.
//! This recipe demonstrates the dispatch + the build-time env-var key
//! convention `CONTRACT_<UPPER_NAME>_<UPPER_EQ>` that the macro reads.
//!
//! Demonstrates the **CM.4** recipe for PMAT-122 (contracts-macros coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Meyer, B. (1992). Applying "Design by Contract". IEEE Computer 25(10).
//!
//! Run with: cargo run --example contracts_macros_multi_equation_dispatch
//!
//! Added by PMAT-122 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use provable_contracts_macros::contract;

#[contract("test-arith-v1", equation = "add")]
fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[contract("test-arith-v1", equation = "sub")]
fn sub(a: i32, b: i32) -> i32 {
    a - b
}

#[contract("test-arith-v1", equation = "mul")]
fn mul(a: i32, b: i32) -> i32 {
    a * b
}

pub fn dispatch_op(op: &str, a: i32, b: i32) -> Option<i32> {
    match op {
        "add" => Some(add(a, b)),
        "sub" => Some(sub(a, b)),
        "mul" => Some(mul(a, b)),
        _ => None,
    }
}

pub fn env_key_for(contract_name: &str, equation: &str) -> String {
    let n = contract_name.replace('-', "_").to_ascii_uppercase();
    let e = equation.to_ascii_uppercase();
    format!("CONTRACT_{n}_{e}")
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_multi_equation_dispatch")?;

    for (op, a, b) in [("add", 2, 3), ("sub", 5, 2), ("mul", 4, 7), ("div", 0, 0)] {
        println!("{op}({a}, {b}) → {:?}", dispatch_op(op, a, b));
    }
    for eq in ["add", "sub", "mul"] {
        println!("{} → {}", eq, env_key_for("test-arith-v1", eq));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn add_works() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(dispatch_op("add", 2, 3), Some(5));
    }

    #[test]
    fn sub_works() {
        assert_eq!(sub(5, 2), 3);
        assert_eq!(dispatch_op("sub", 5, 2), Some(3));
    }

    #[test]
    fn mul_works() {
        assert_eq!(mul(4, 7), 28);
        assert_eq!(dispatch_op("mul", 4, 7), Some(28));
    }

    #[test]
    fn unknown_op_returns_none() {
        assert!(dispatch_op("div", 1, 1).is_none());
    }

    #[test]
    fn env_key_uppercases_and_underscores() {
        assert_eq!(
            env_key_for("test-arith-v1", "add"),
            "CONTRACT_TEST_ARITH_V1_ADD"
        );
    }

    #[test]
    fn env_key_handles_already_underscored() {
        assert_eq!(
            env_key_for("test_arith_v1", "sub"),
            "CONTRACT_TEST_ARITH_V1_SUB"
        );
    }

    #[test]
    fn env_key_uppercases_equation() {
        assert_eq!(env_key_for("c", "MyEq"), "CONTRACT_C_MYEQ");
    }
}
