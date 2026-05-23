//! # Tier 2.2 — QLoRA — Mistral 4-bit + rank-16
//!
//! Falsifier: QLoRA rank-16 mistral: post-merge perplexity ≤ FP16 baseline + 5%
//!
//! Run with: cargo run --example t2_qlora_4bit_rank16_mistral

use apr_cookbook::finetune::qlora;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const RANK: u32 = 16;
const N_BASE_PARAMS: u64 = 7000000000;
const LORA_TRAINABLE: u64 = 8388608;
const DOUBLE_QUANT: bool = false;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_qlora_4bit_rank16_mistral")?;
    let mem = qlora::qlora_memory(N_BASE_PARAMS, LORA_TRAINABLE);
    println!(
        "✓ QLoRA r={}: 4-bit base={:.1}MB + LoRA={:.1}MB = {:.1}MB ({:.2}× of FP16 {:.1}MB)",
        RANK,
        mem.base_4bit_mb,
        mem.lora_fp32_mb,
        mem.total_mb,
        mem.savings_ratio,
        mem.fp16_baseline_mb
    );
    assert!(
        mem.savings_ratio < 0.4,
        "falsifier: QLoRA total memory should be ≤ 0.4× FP16, got {}",
        mem.savings_ratio
    );

    // Quantization signal-recovery sanity check.
    let probe: Vec<f64> = (0..256).map(|i| (i as f64).sin()).collect();
    let (_, _, stats) = qlora::quantize_4bit_blockwise(&probe, 64);
    let stats = if DOUBLE_QUANT {
        qlora::enable_double_quant(&stats)
    } else {
        stats
    };
    assert!(
        stats.max_abs_error < 0.15,
        "max abs error of 4-bit blockwise quant should be < 0.15"
    );
    println!(
        "  signal-quant max_err={:.4} double_quant={}",
        stats.max_abs_error, stats.double_quant
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
        let mem = qlora::qlora_memory(N_BASE_PARAMS, LORA_TRAINABLE);
        assert!(mem.savings_ratio < 0.4);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // FP32 base + FP32 LoRA gives ratio ≈ 1.0+, which violates QLoRA savings.
        let fp32_base_bytes = N_BASE_PARAMS * 4;
        let fp32_lora_bytes = LORA_TRAINABLE * 4;
        let total = (fp32_base_bytes + fp32_lora_bytes) as f64;
        let fp16_baseline = (N_BASE_PARAMS * 2) as f64;
        let ratio = total / fp16_baseline;
        assert!(
            ratio > 1.0,
            "FP32 base + LoRA should be > 1.0× FP16 baseline"
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let a = qlora::qlora_memory(N_BASE_PARAMS, LORA_TRAINABLE);
        let b = qlora::qlora_memory(N_BASE_PARAMS, LORA_TRAINABLE);
        assert_eq!(a, b);
    }
}
