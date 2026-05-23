//! # Tier 3.17 — QAT MXFP4 microscaling (qwen3 family)
//!
//! Falsifier: MXFP4 4-bit microscaling — 32-element block with shared 8-bit
//! exponent yields ≈ 4.25 bits/element exactly per OCP spec.
//!
//! Run with: cargo run --example t3_qat_mxfp4

use apr_cookbook::finetune::tier3_closeout as t3c;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const BLOCK_SIZE: u32 = 32;
const EXPECTED_BPE: f64 = 4.25;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_qat_mxfp4")?;
    let bpe = t3c::mxfp4_bits_per_element(BLOCK_SIZE);
    println!(
        "✓ MXFP4 block_size={}: {:.3} bits/elem (target {})",
        BLOCK_SIZE, bpe, EXPECTED_BPE
    );
    assert!((bpe - EXPECTED_BPE).abs() < 0.05);
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
        let bpe = t3c::mxfp4_bits_per_element(BLOCK_SIZE);
        assert!((bpe - EXPECTED_BPE).abs() < 0.05);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // block_size=4 → (4*4 + 8) / 4 = 6.0 bits/elem (much higher).
        let bpe = t3c::mxfp4_bits_per_element(4);
        assert!((bpe - EXPECTED_BPE).abs() > 0.5);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = t3c::mxfp4_bits_per_element(BLOCK_SIZE);
        let b = t3c::mxfp4_bits_per_element(BLOCK_SIZE);
        assert_eq!(a, b);
    }
}
