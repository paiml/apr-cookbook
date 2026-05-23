//! # Tier 3.18 — FSDP-LoRA (llama family, GPU)
//!
//! Falsifier: FSDP-LoRA per-GPU memory shard at world_size=8 ≤ 0.3× of
//! single-GPU baseline (closed-form sharding of the base; LoRA replicated).
//!
//! Run with: cargo run --example t3_fsdp_lora

use apr_cookbook::finetune::tier3_closeout as t3c;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const TOTAL_PARAMS: u64 = 7_000_000_000;
const LORA_PARAMS: u64 = 64_000_000;
const WORLD_SIZE: u32 = 8;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_fsdp_lora")?;
    let r = t3c::fsdp_per_gpu_ratio(TOTAL_PARAMS, WORLD_SIZE, LORA_PARAMS);
    println!(
        "✓ FSDP-LoRA at world_size={}: per-GPU shard = {:.4}× of baseline",
        WORLD_SIZE, r
    );
    assert!(r < 0.3);
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
        assert!(t3c::fsdp_per_gpu_ratio(TOTAL_PARAMS, WORLD_SIZE, LORA_PARAMS) < 0.3);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // World size 1 — no sharding, ratio = 1.0.
        let r = t3c::fsdp_per_gpu_ratio(TOTAL_PARAMS, 1, LORA_PARAMS);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = t3c::fsdp_per_gpu_ratio(TOTAL_PARAMS, WORLD_SIZE, LORA_PARAMS);
        let b = t3c::fsdp_per_gpu_ratio(TOTAL_PARAMS, WORLD_SIZE, LORA_PARAMS);
        assert_eq!(a, b);
    }
}
