//! # Tier 2.5 — Vector-Bank LoRA (V-Bank LoRA) — llama family
//!
//! Falsifier: Vector-Bank LoRA: shared bank of N basis vectors compresses
//! LoRA storage by ≥ 5× vs standard rank-r LoRA on a 64-layer × rank-16 ×
//! 4096×4096 base.
//!
//! Run with: cargo run --example t2_vblora

use apr_cookbook::finetune::peft_variants as peft;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const N_LAYERS: u64 = 64;
const D_OUT: u64 = 4096;
const D_IN: u64 = 4096;
const RANK: u64 = 16;
const BANK: u64 = 128;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_vblora")?;
    let ratio = peft::vblora_compression_ratio(N_LAYERS, D_IN, D_OUT, RANK, BANK);
    println!(
        "✓ V-Bank LoRA: bank={} layers={} rank={} compression={:.2}×",
        BANK, N_LAYERS, RANK, ratio
    );
    assert!(
        ratio >= 5.0,
        "V-Bank LoRA must compress ≥ 5× vs standard, got {ratio}"
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
        let ratio = peft::vblora_compression_ratio(N_LAYERS, D_IN, D_OUT, RANK, BANK);
        assert!(ratio >= 5.0);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // bank ≥ n_layers · rank: V-Bank degenerates to standard LoRA, ratio → ~1.
        let huge_bank = N_LAYERS * RANK;
        let ratio = peft::vblora_compression_ratio(N_LAYERS, D_IN, D_OUT, RANK, huge_bank);
        assert!(ratio < 5.0);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = peft::vblora_compression_ratio(N_LAYERS, D_IN, D_OUT, RANK, BANK);
        let b = peft::vblora_compression_ratio(N_LAYERS, D_IN, D_OUT, RANK, BANK);
        assert_eq!(a, b);
    }
}
