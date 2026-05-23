//! # apr ptx-map — Prefill vs Decode Kernel Variants
//!
//! `apr ptx-map <FILE> [--prefill]` switches between the two kernel
//! regimes: prefill (batched, GEMM-bound, large block sizes) vs decode
//! (single-token, GEMV-bound, latency-bound). The kernel selected for a
//! given (layer, regime) pair is deterministic given the model dtype and
//! block-size policy. This recipe documents and tests the dispatch table.
//!
//! Demonstrates the **PTXMAP.5** recipe for PMAT-096 (apr ptx-map coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PTX-MAP-003 + GEMV/GEMM kernel-class spec
//!
//! Run with: cargo run --example cli_ptx_map_prefill_vs_decode
//!
//! Added by PMAT-096 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    FP16,
    Q4K,
    Q5K,
    Q8_0,
}

pub fn pick_kernel(quant: Quant, regime: Regime) -> &'static str {
    match (quant, regime) {
        (Quant::FP16, Regime::Prefill) => "FP16Gemm",
        (Quant::FP16, Regime::Decode) => "FP16Gemv",
        (Quant::Q4K, Regime::Prefill) => "Q4KGemm",
        (Quant::Q4K, Regime::Decode) => "Q4KGemv",
        (Quant::Q5K, Regime::Prefill) => "Q5KGemm",
        (Quant::Q5K, Regime::Decode) => "Q5KGemv",
        (Quant::Q8_0, Regime::Prefill) => "Q8_0Gemm",
        (Quant::Q8_0, Regime::Decode) => "Q8_0Gemv",
    }
}

pub fn is_gemm_kernel(name: &str) -> bool {
    name.ends_with("Gemm")
}

pub fn is_gemv_kernel(name: &str) -> bool {
    name.ends_with("Gemv")
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_ptx_map_prefill_vs_decode")?;

    println!("Quant   Prefill              Decode");
    for q in [Quant::FP16, Quant::Q4K, Quant::Q5K, Quant::Q8_0] {
        println!(
            "{q:>6?}  {:>20}  {:>20}",
            pick_kernel(q, Regime::Prefill),
            pick_kernel(q, Regime::Decode)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_decode_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn every_quant_has_both_regimes() {
        for q in [Quant::FP16, Quant::Q4K, Quant::Q5K, Quant::Q8_0] {
            let p = pick_kernel(q, Regime::Prefill);
            let d = pick_kernel(q, Regime::Decode);
            assert!(!p.is_empty());
            assert!(!d.is_empty());
            assert_ne!(p, d, "prefill and decode must dispatch different kernels");
        }
    }

    #[test]
    fn prefill_kernels_are_gemm() {
        // Critical: prefill is batched matmul → GEMM, never GEMV.
        for q in [Quant::FP16, Quant::Q4K, Quant::Q5K, Quant::Q8_0] {
            let k = pick_kernel(q, Regime::Prefill);
            assert!(is_gemm_kernel(k), "prefill kernel {k} is not Gemm");
        }
    }

    #[test]
    fn decode_kernels_are_gemv() {
        // Decode is single-token (1×k @ k×n) → GEMV, never GEMM.
        for q in [Quant::FP16, Quant::Q4K, Quant::Q5K, Quant::Q8_0] {
            let k = pick_kernel(q, Regime::Decode);
            assert!(is_gemv_kernel(k), "decode kernel {k} is not Gemv");
        }
    }

    #[test]
    fn quant_prefix_preserved_in_kernel_name() {
        // Kernel naming must keep the quant prefix readable for logs/profiles.
        assert!(pick_kernel(Quant::Q4K, Regime::Decode).starts_with("Q4K"));
        assert!(pick_kernel(Quant::Q8_0, Regime::Prefill).starts_with("Q8_0"));
    }

    #[test]
    fn classifier_is_mutually_exclusive() {
        // A kernel is Gemm xor Gemv, not both.
        for q in [Quant::FP16, Quant::Q4K] {
            for r in [Regime::Prefill, Regime::Decode] {
                let k = pick_kernel(q, r);
                assert_ne!(is_gemm_kernel(k), is_gemv_kernel(k));
            }
        }
    }
}
