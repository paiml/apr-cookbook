//! # Tier 2.7 — LoRA on AQLM 2-bit base (llama family)
//!
//! Falsifier: AQLM 2-bit base + LoRA storage ratio is bounded ≤ 0.3× FP16,
//! so post-merge perplexity-tolerance budget (≤ FP16 + 10%) is feasible
//! given that 2-bit recovery error per block ≤ |max|/1.5.
//!
//! Run with: cargo run --example t2_lora_aqlm

use apr_cookbook::finetune::quantized_base as q;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const BASE_PARAMS: u64 = 7_000_000_000;
const LORA_PARAMS: u64 = 64_000_000;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_lora_aqlm")?;
    let r = q::aqlm_storage_ratio(BASE_PARAMS, LORA_PARAMS);
    println!(
        "✓ AQLM 2-bit + LoRA: storage ratio = {:.4}× of FP16 baseline",
        r
    );
    assert!(r < 0.3, "AQLM 2-bit storage must be ≤ 0.3× FP16, got {r}");
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
        assert!(q::aqlm_storage_ratio(BASE_PARAMS, LORA_PARAMS) < 0.3);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Huge LoRA → ratio approaches 1.0+.
        let huge_lora = BASE_PARAMS / 2;
        let r = q::aqlm_storage_ratio(BASE_PARAMS, huge_lora);
        assert!(r > 0.3);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = q::aqlm_storage_ratio(BASE_PARAMS, LORA_PARAMS);
        let b = q::aqlm_storage_ratio(BASE_PARAMS, LORA_PARAMS);
        assert_eq!(a, b);
    }
}
