//! # Contracts-Macros Attribute Basic
//!
//! Smallest possible use of the `#[contract]` proc-macro from
//! `provable_contracts_macros` (package: `aprender-contracts-macros`).
//! Annotates a `fn double(x: i32) -> i32` with a contract reference and
//! an equation name. The macro reads `CONTRACT_<NAME>_<EQ>` env vars set
//! by the consuming crate's `build.rs`; when absent (as here in the
//! cookbook), the macro gracefully degrades to a no-op pass-through.
//!
//! Demonstrates the **CM.1** recipe per
//! `docs/specifications/expand-cookbooks/subcrate-coverage.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Meyer, B. (1992). Applying "Design by Contract". IEEE Computer 25(10). DOI: 10.1109/2.161279
//!
//! Run with: cargo run --example contracts_macros_attribute_basic
//!
//! Added by PMAT-084 (expand-cookbooks: aprender-contracts-macros coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use provable_contracts_macros::contract;

#[contract("test-double-v1", equation = "double")]
fn double(x: i32) -> i32 {
    x * 2
}

#[contract("test-square-v1", equation = "square")]
fn square(x: i32) -> i32 {
    x * x
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_attribute_basic")?;
    let d = double(21);
    let s = square(7);
    println!("double(21) = {d} (contract: test-double-v1#double)");
    println!("square(7)  = {s} (contract: test-square-v1#square)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macros_compile_and_run() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn double_returns_double() {
        assert_eq!(double(5), 10);
        assert_eq!(double(0), 0);
        assert_eq!(double(-3), -6);
    }

    #[test]
    fn square_returns_square() {
        assert_eq!(square(4), 16);
        assert_eq!(square(0), 0);
        assert_eq!(square(-2), 4);
    }
}
