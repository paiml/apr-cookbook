//! # Analysis — Contract Algorithm-Binding Pattern
//!
//! aprender 0.31.x Unreleased shipped a record sweep that flipped 150+
//! provable contracts from `unbound` → `PARTIAL_ALGORITHM_LEVEL` by tying
//! each falsifier to a concrete, executable algorithm reference. This
//! recipe demonstrates the authoring pattern: take an algorithm, identify
//! the falsifiable invariant, write a contract YAML stub that binds the
//! algorithm to its falsifier.
//!
//! Demonstrates the **AN+.2** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CHANGELOG Unreleased "150+ provable contracts flipped" + Hoare (1969). An Axiomatic Basis for Computer Programming. CACM 12(10). DOI: 10.1145/363235.363259
//!
//! Run with: cargo run --example analysis_contract_algorithm_binding_pattern
//!
//! Added by PMAT-086 (expand-cookbooks: Tier 4 authoring patterns).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

/// The algorithm under contract: standard softmax. Invariants:
/// - All outputs in [0, 1]
/// - Output sum equals 1.0 (normalized)
/// - Stable under input shift (softmax(x + c) == softmax(x))
fn softmax(x: &[f32]) -> Vec<f32> {
    let max_x = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = x.iter().map(|v| (v - max_x).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|v| v / sum).collect()
}

/// Falsifier 1: outputs sum to 1.0 (within tolerance).
fn falsify_softmax_sums_to_one(x: &[f32]) -> bool {
    let out = softmax(x);
    let sum: f32 = out.iter().sum();
    (sum - 1.0).abs() < 1e-6
}

/// Falsifier 2: outputs all in [0, 1].
fn falsify_softmax_in_unit_interval(x: &[f32]) -> bool {
    softmax(x).iter().all(|v| (0.0..=1.0).contains(v))
}

/// Falsifier 3: shift invariance.
fn falsify_softmax_shift_invariant(x: &[f32], shift: f32) -> bool {
    let a = softmax(x);
    let shifted: Vec<f32> = x.iter().map(|v| v + shift).collect();
    let b = softmax(&shifted);
    a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-5)
}

const CONTRACT_YAML_STUB: &str = "\
# Hand-authored binding stub for the softmax-kernel-v1 contract.
# Bind the three falsifiers to the algorithm reference. After committing
# this YAML, build.rs sets CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX=implemented
# and #[contract] enforces the preconditions/postconditions at compile time.
contract:
  name: softmax-kernel-v1
  status: PARTIAL_ALGORITHM_LEVEL
  algorithm:
    fn: softmax
    file: examples/analysis/analysis_contract_algorithm_binding_pattern.rs
  falsifiers:
    - falsify_softmax_sums_to_one
    - falsify_softmax_in_unit_interval
    - falsify_softmax_shift_invariant
";

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("analysis_contract_algorithm_binding_pattern")?;
    let x = [1.0f32, 2.0, 3.0];
    println!("softmax({:?}) = {:?}", x, softmax(&x));
    println!("\nfalsifiers:");
    println!("  sums_to_one:       {}", falsify_softmax_sums_to_one(&x));
    println!(
        "  in_unit_interval:  {}",
        falsify_softmax_in_unit_interval(&x)
    );
    println!(
        "  shift_invariant:   {}",
        falsify_softmax_shift_invariant(&x, 100.0)
    );
    println!("\ncontract YAML stub (commit to contracts/softmax-kernel-v1.yaml):");
    println!("{CONTRACT_YAML_STUB}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn falsifier_sums_to_one() {
        assert!(falsify_softmax_sums_to_one(&[0.5, 1.5, -2.0]));
    }

    #[test]
    fn falsifier_in_unit_interval() {
        assert!(falsify_softmax_in_unit_interval(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn falsifier_shift_invariant_holds_for_large_shift() {
        // Numerical-stable softmax should be invariant under additive shift.
        assert!(falsify_softmax_shift_invariant(&[1.0, 2.0, 3.0], 1000.0));
    }
}
