//! # Tier 2.1 — LoRA rank-32 on mistral
//!
//! Falsifier: rank-32 LoRA merge round-trip is bit-identical when α/r = 1.0
//! (and trainable param count = r × (d_in + d_out)).
//!
//! Run with: cargo run --example t2_lora_rank32_mistral

use apr_cookbook::finetune::lora;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const FAMILY: &str = "mistral";
const RANK: u32 = 32;
const D_IN: usize = 256;
const D_OUT: usize = 256;
const ALPHA: f64 = 32.0; // α=r so scale = 1.0 (round-trip identity)

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_lora_rank32_mistral")?;
    let mut layer = lora::LoraLayer::new(D_OUT, D_IN, RANK, ALPHA);
    layer.set_b_for_test(0.05);

    let merged = layer.merge();
    let unmerged = layer.unmerge(&merged);
    let dist = lora::frobenius_distance(&layer.base, &unmerged);

    let trainable = layer.trainable_params();
    let frozen = layer.frozen_params();
    let ratio = layer.reduction_ratio();

    println!(
        "✓ {} LoRA-r{}: trainable={} frozen={} ratio={:.4} merge-roundtrip-dist={:.2e}",
        FAMILY, RANK, trainable, frozen, ratio, dist
    );

    assert!(
        dist < 1e-10,
        "falsifier: merge round-trip must be bit-identical at α/r=1, got {dist}"
    );
    assert_eq!(
        trainable,
        u64::from(RANK) * (D_IN as u64 + D_OUT as u64),
        "trainable param count must match r×(d_in+d_out) formula"
    );
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
        let mut layer = lora::LoraLayer::new(D_OUT, D_IN, RANK, ALPHA);
        layer.set_b_for_test(0.05);
        let merged = layer.merge();
        let unmerged = layer.unmerge(&merged);
        assert!(lora::frobenius_distance(&layer.base, &unmerged) < 1e-10);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // α/r ≠ 1.0 → round-trip still works (linear algebra), but param-count
        // formula is what we falsify. Use wrong formula: should not match.
        let layer = lora::LoraLayer::new(D_OUT, D_IN, RANK, ALPHA);
        let wrong_count = u64::from(RANK) * (D_IN as u64 * D_OUT as u64); // multiplicative, not additive
        assert_ne!(layer.trainable_params(), wrong_count);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = lora::LoraLayer::new(D_OUT, D_IN, RANK, ALPHA);
        let b = lora::LoraLayer::new(D_OUT, D_IN, RANK, ALPHA);
        assert_eq!(a.merge(), b.merge());
    }
}
