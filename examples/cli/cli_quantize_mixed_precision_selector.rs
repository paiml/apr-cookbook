//! # apr quantize --mixed — Per-Layer Precision Selector
//!
//! `apr quantize --scheme mixed` selects per-layer precision: keep
//! attention output proj in FP16 (sensitive), drop FFN to Int4 (robust),
//! quantize embeddings to Int8. This recipe builds the per-layer
//! decision rule and the size estimate.
//!
//! Demonstrates the **QUANT.4** recipe for PMAT-112 (apr quantize coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender QUANT-001 + Dettmers et al. 2023 (LLM.int8/QLoRA)
//!
//! Run with: cargo run --example cli_quantize_mixed_precision_selector
//!
//! Added by PMAT-112 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum LayerKind {
    Embedding,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionOut,
    FfnGate,
    FfnUp,
    FfnDown,
    LayerNorm,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Precision {
    Fp32,
    Fp16,
    Int8,
    Int4,
}

impl Precision {
    pub fn bits(self) -> u8 {
        match self {
            Precision::Fp32 => 32,
            Precision::Fp16 => 16,
            Precision::Int8 => 8,
            Precision::Int4 => 4,
        }
    }
}

pub fn select_precision(layer: LayerKind) -> Precision {
    match layer {
        // Norms must stay FP32 — small + numerically sensitive.
        LayerKind::LayerNorm => Precision::Fp32,
        // Output projection of attention dominates accuracy loss when quantized.
        LayerKind::AttentionOut => Precision::Fp16,
        // Embeddings tolerate Int8 well in practice.
        LayerKind::Embedding => Precision::Int8,
        // FFN matrices are the largest and most quantization-tolerant.
        LayerKind::FfnGate | LayerKind::FfnUp | LayerKind::FfnDown => Precision::Int4,
        // Attention Q/K/V matrices: balanced choice is Int8.
        LayerKind::AttentionQ | LayerKind::AttentionK | LayerKind::AttentionV => Precision::Int8,
    }
}

pub fn estimate_size_bits(layer_param_count: u64, layer: LayerKind) -> u64 {
    layer_param_count * select_precision(layer).bits() as u64
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_quantize_mixed_precision_selector")?;

    let layers = [
        LayerKind::LayerNorm,
        LayerKind::Embedding,
        LayerKind::AttentionQ,
        LayerKind::AttentionOut,
        LayerKind::FfnGate,
    ];
    for l in layers {
        println!(
            "{l:?}  →  {:?}  ({} bits)",
            select_precision(l),
            select_precision(l).bits()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn layer_norm_stays_fp32() {
        // LayerNorms are tiny but numerically critical.
        assert_eq!(select_precision(LayerKind::LayerNorm), Precision::Fp32);
    }

    #[test]
    fn attention_out_stays_fp16() {
        assert_eq!(select_precision(LayerKind::AttentionOut), Precision::Fp16);
    }

    #[test]
    fn ffn_drops_to_int4() {
        // FFN dominates parameter count; Int4 is the standard for QLoRA-style.
        for l in [LayerKind::FfnGate, LayerKind::FfnUp, LayerKind::FfnDown] {
            assert_eq!(select_precision(l), Precision::Int4);
        }
    }

    #[test]
    fn embeddings_use_int8() {
        assert_eq!(select_precision(LayerKind::Embedding), Precision::Int8);
    }

    #[test]
    fn attention_qkv_use_int8() {
        for l in [
            LayerKind::AttentionQ,
            LayerKind::AttentionK,
            LayerKind::AttentionV,
        ] {
            assert_eq!(select_precision(l), Precision::Int8);
        }
    }

    #[test]
    fn precision_bits_correct() {
        assert_eq!(Precision::Fp32.bits(), 32);
        assert_eq!(Precision::Fp16.bits(), 16);
        assert_eq!(Precision::Int8.bits(), 8);
        assert_eq!(Precision::Int4.bits(), 4);
    }

    #[test]
    fn size_estimate_proportional_to_bits() {
        // Same param count: Int4 should be 8x smaller than Fp32.
        let n = 1_000_000;
        let fp32 = estimate_size_bits(n, LayerKind::LayerNorm);
        let int4 = estimate_size_bits(n, LayerKind::FfnUp);
        assert_eq!(fp32 / int4, 8);
    }
}
