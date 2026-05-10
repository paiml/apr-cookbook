//! # Tier 3.18 — Sample packing (Axolotl) (cross-family)
//!
//! Falsifier: sample packing reduces batched-token-padding by ≥ 50% (i.e.
//! useful-token ratio is ≥ 1.5× of naive batching's ratio) on a
//! length-heterogeneous fixture.
//!
//! Run with: cargo run --example t3_sample_packing

use apr_cookbook::finetune::tier3_closeout as t3c;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture() -> Vec<u32> {
    vec![100, 20, 15, 10, 5, 8, 6, 12, 18, 25]
}
const BIN_SIZE: u32 = 200;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_sample_packing")?;
    let lens = fixture();
    let max_len = *lens.iter().max().unwrap();
    let naive = t3c::naive_batching_useful_ratio(&lens, max_len);
    let packed = t3c::packed_useful_ratio(&lens, BIN_SIZE);
    println!(
        "✓ sample packing: naive ratio = {:.3}, packed = {:.3} ({:.1}× ratio)",
        naive,
        packed,
        packed / naive
    );
    assert!(packed > naive * 1.5);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn falsifier_holds_on_fixture() {
        let lens = fixture();
        let n = t3c::naive_batching_useful_ratio(&lens, *lens.iter().max().unwrap());
        let p = t3c::packed_useful_ratio(&lens, BIN_SIZE);
        assert!(p > n * 1.5);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Uniform-length fixture — packed and naive ratios converge to ~1.0.
        let uniform = vec![100_u32; 8];
        let n = t3c::naive_batching_useful_ratio(&uniform, 100);
        let p = t3c::packed_useful_ratio(&uniform, 100);
        assert!((p - n).abs() < 0.01);
    }

    #[test]
    fn deterministic_across_runs() {
        let lens = fixture();
        let a = t3c::packed_useful_ratio(&lens, BIN_SIZE);
        let b = t3c::packed_useful_ratio(&lens, BIN_SIZE);
        assert_eq!(a, b);
    }
}
