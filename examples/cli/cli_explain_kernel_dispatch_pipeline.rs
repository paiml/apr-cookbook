//! # apr explain — `--kernel` Dispatch Pipeline Description
//!
//! `apr explain --kernel <ARCH>` describes the kernel-dispatch pipeline
//! for a given architecture: which kernel runs at each layer slot
//! (embed → attention → ffn → norm → unembed). This recipe builds the
//! per-architecture dispatch table and asserts the contract: every layer
//! slot has a non-empty kernel name, and the output is deterministic per
//! (architecture, dtype) pair.
//!
//! Demonstrates the **EXPLAIN.5** recipe for PMAT-099 (apr explain coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PTX-MAP-003 + per-arch kernel-class spec
//!
//! Run with: cargo run --example cli_explain_kernel_dispatch_pipeline
//!
//! Added by PMAT-099 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerSlot {
    Embed,
    AttentionQkv,
    AttentionOut,
    FfnGateUp,
    FfnDown,
    Norm,
    Unembed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchSlot {
    pub slot: LayerSlot,
    pub kernel: &'static str,
}

pub fn dispatch_pipeline(arch: &str, dtype: &str) -> Vec<DispatchSlot> {
    let prefix = match dtype {
        "fp16" => "FP16",
        "bf16" => "BF16",
        "q4_k" => "Q4K",
        "q5_k" => "Q5K",
        _ => "FP32",
    };
    let attn_q = match arch {
        "whisper" => format!("{prefix}WhisperAttn"),
        _ => format!("{prefix}LlamaAttn"),
    };
    let ffn = match arch {
        "qwen2" | "qwen3" | "llama" | "mistral" | "gemma" => format!("{prefix}SwiGLU"),
        "bert" | "gpt2" => format!("{prefix}Gelu"),
        _ => format!("{prefix}Mlp"),
    };
    vec![
        DispatchSlot {
            slot: LayerSlot::Embed,
            kernel: Box::leak(format!("{prefix}Embed").into_boxed_str()),
        },
        DispatchSlot {
            slot: LayerSlot::AttentionQkv,
            kernel: Box::leak(attn_q.into_boxed_str()),
        },
        DispatchSlot {
            slot: LayerSlot::AttentionOut,
            kernel: Box::leak(format!("{prefix}AttnOut").into_boxed_str()),
        },
        DispatchSlot {
            slot: LayerSlot::FfnGateUp,
            kernel: Box::leak(ffn.into_boxed_str()),
        },
        DispatchSlot {
            slot: LayerSlot::FfnDown,
            kernel: Box::leak(format!("{prefix}FfnDown").into_boxed_str()),
        },
        DispatchSlot {
            slot: LayerSlot::Norm,
            kernel: Box::leak(format!("{prefix}RmsNorm").into_boxed_str()),
        },
        DispatchSlot {
            slot: LayerSlot::Unembed,
            kernel: Box::leak(format!("{prefix}Unembed").into_boxed_str()),
        },
    ]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_explain_kernel_dispatch_pipeline")?;

    for (arch, dtype) in [("qwen3", "bf16"), ("whisper", "fp16"), ("bert", "fp32")] {
        println!("=== {arch} / {dtype} ===");
        for s in dispatch_pipeline(arch, dtype) {
            println!("  {:?} → {}", s.slot, s.kernel);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn pipeline_has_seven_slots() {
        let p = dispatch_pipeline("qwen3", "bf16");
        assert_eq!(p.len(), 7);
    }

    #[test]
    fn every_slot_has_nonempty_kernel() {
        let p = dispatch_pipeline("llama", "fp16");
        for s in p {
            assert!(!s.kernel.is_empty(), "empty kernel for {:?}", s.slot);
        }
    }

    #[test]
    fn dispatch_is_deterministic_per_arch_dtype() {
        let a = dispatch_pipeline("qwen3", "bf16");
        let b = dispatch_pipeline("qwen3", "bf16");
        assert_eq!(a.len(), b.len());
        for (sa, sb) in a.iter().zip(b.iter()) {
            assert_eq!(sa.slot, sb.slot);
            assert_eq!(sa.kernel, sb.kernel);
        }
    }

    #[test]
    fn whisper_uses_whisper_attn() {
        let p = dispatch_pipeline("whisper", "fp16");
        let attn = p
            .iter()
            .find(|s| s.slot == LayerSlot::AttentionQkv)
            .unwrap();
        assert!(attn.kernel.contains("Whisper"));
    }

    #[test]
    fn qwen_uses_swiglu_ffn() {
        let p = dispatch_pipeline("qwen3", "bf16");
        let ffn = p.iter().find(|s| s.slot == LayerSlot::FfnGateUp).unwrap();
        assert!(ffn.kernel.contains("SwiGLU"));
    }

    #[test]
    fn bert_uses_gelu_ffn() {
        let p = dispatch_pipeline("bert", "fp32");
        let ffn = p.iter().find(|s| s.slot == LayerSlot::FfnGateUp).unwrap();
        assert!(ffn.kernel.contains("Gelu"));
    }

    #[test]
    fn dtype_prefix_propagates_to_every_slot() {
        let p = dispatch_pipeline("llama", "q4_k");
        for s in p {
            assert!(s.kernel.starts_with("Q4K"), "missing Q4K prefix on {s:?}");
        }
    }
}
