//! # Contracts-Macros Invariant Compose Chain
//!
//! Compose a chain of unary invariants `inv_a → inv_b → inv_c` and
//! verify the chain is type-compatible (each output type matches the
//! next input type). Returns first incompatible step.
//!
//! Demonstrates the **CMM.180** recipe for PMAT-217 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hoare-logic composition rule {P} c1 {Q}, {Q} c2 {R} ⇒
//!  {P} c1;c2 {R}; functional pipeline composition.
//!
//! Run with: cargo run --example contracts_macros_invariant_compose_chain
//!
//! Added by PMAT-217 (catalog 1576→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ComposeVerdict {
    Composable,
    Mismatch {
        step: u32,
        expected: String,
        actual: String,
    },
    InvalidConfig,
}

/// Steps: (input_type, output_type) — each step's output must match the next's input.
pub fn check(steps: &[(&str, &str)]) -> ComposeVerdict {
    if steps.len() < 2 {
        return ComposeVerdict::InvalidConfig;
    }
    for w in steps.windows(2).enumerate() {
        let (i, pair) = w;
        let prev_out = pair[0].1;
        let next_in = pair[1].0;
        if prev_out != next_in {
            return ComposeVerdict::Mismatch {
                step: (i + 1) as u32,
                expected: prev_out.to_string(),
                actual: next_in.to_string(),
            };
        }
    }
    ComposeVerdict::Composable
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_invariant_compose_chain")?;

    let chain = [("A", "B"), ("B", "C"), ("C", "D")];
    println!("ok: {:?}", check(&chain));
    let bad = [("A", "B"), ("X", "C")];
    println!("mismatch: {:?}", check(&bad));
    println!("invalid: {:?}", check(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[]), ComposeVerdict::InvalidConfig);
    }

    #[test]
    fn single_step_rejected() {
        assert_eq!(check(&[("A", "B")]), ComposeVerdict::InvalidConfig);
    }

    #[test]
    fn linear_chain_composable() {
        let v = check(&[("A", "B"), ("B", "C")]);
        assert_eq!(v, ComposeVerdict::Composable);
    }

    #[test]
    fn mismatch_at_step_one() {
        let v = check(&[("A", "B"), ("X", "C")]);
        assert_eq!(
            v,
            ComposeVerdict::Mismatch {
                step: 1,
                expected: "B".to_string(),
                actual: "X".to_string(),
            }
        );
    }

    #[test]
    fn mismatch_at_step_two() {
        let v = check(&[("A", "B"), ("B", "C"), ("X", "D")]);
        if let ComposeVerdict::Mismatch { step, .. } = v {
            assert_eq!(step, 2);
        }
    }

    #[test]
    fn long_chain_composable() {
        let v = check(&[("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]);
        assert_eq!(v, ComposeVerdict::Composable);
    }

    #[test]
    fn deterministic() {
        let r1 = check(&[("A", "B"), ("B", "C")]);
        let r2 = check(&[("A", "B"), ("B", "C")]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn case_sensitive_types() {
        let v = check(&[("A", "B"), ("b", "C")]);
        assert!(matches!(v, ComposeVerdict::Mismatch { .. }));
    }

    #[test]
    fn unicode_type_supported() {
        let v = check(&[("café", "résumé"), ("résumé", "naïve")]);
        assert_eq!(v, ComposeVerdict::Composable);
    }

    #[test]
    fn many_steps_handled() {
        let chain: Vec<(&str, &str)> = (0..30).map(|_| ("X", "X")).collect();
        let v = check(&chain);
        assert_eq!(v, ComposeVerdict::Composable);
    }

    #[test]
    fn self_loop_composable() {
        let v = check(&[("A", "A"), ("A", "A")]);
        assert_eq!(v, ComposeVerdict::Composable);
    }

    #[test]
    fn empty_string_type_treated_as_value() {
        let v = check(&[("A", ""), ("", "B")]);
        assert_eq!(v, ComposeVerdict::Composable);
    }
}
