//! # apr gpu — OOM Recovery Advisor
//!
//! When `apr` workflows hit a CUDA OOM, the operator wants concrete next
//! actions. This recipe builds the advisor as a pure function: given
//! (model_size_b, batch, context, dtype, current_vram), return a
//! ranked list of mitigations (reduce batch, switch to int8, enable
//! gradient checkpointing, fall back to CPU, …).
//!
//! Demonstrates the **GPU.12** recipe for PMAT-107 (apr gpu coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CRUX-F-13 (OOM postmortem) + remediation playbook
//!
//! Run with: cargo run --example cli_gpu_oom_recovery_advisor
//!
//! Added by PMAT-107 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mitigation {
    ReduceBatch { from: u32, to: u32 },
    SwitchToInt8,
    EnableGradientCheckpointing,
    EnableCpuOffload,
    UseLora,
    DefaultToCpu,
}

#[derive(Debug, Clone, Copy)]
pub struct OomContext {
    pub model_b_params: f64,
    pub current_batch: u32,
    pub context_tokens: u32,
    pub dtype_bytes: u32,
    pub vram_total_gb: f64,
}

pub fn rank_mitigations(ctx: OomContext) -> Vec<Mitigation> {
    let mut out = Vec::new();
    // First mitigation: halve batch (always cheap).
    if ctx.current_batch > 1 {
        out.push(Mitigation::ReduceBatch {
            from: ctx.current_batch,
            to: ctx.current_batch / 2,
        });
    }
    // Second: switch dtype to int8 if currently fp16/bf16.
    if ctx.dtype_bytes >= 2 {
        out.push(Mitigation::SwitchToInt8);
    }
    // Third: gradient checkpointing (saves ~30% activation memory).
    out.push(Mitigation::EnableGradientCheckpointing);
    // Fourth: CPU offload (slow but fits anything).
    out.push(Mitigation::EnableCpuOffload);
    // Fifth: LoRA — only when finetuning.
    if ctx.model_b_params * 2.0 > ctx.vram_total_gb {
        out.push(Mitigation::UseLora);
    }
    // Last resort: CPU.
    out.push(Mitigation::DefaultToCpu);
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_gpu_oom_recovery_advisor")?;

    let scenarios = [
        (
            "7B bf16 batch 32 on 24GB",
            OomContext {
                model_b_params: 7.0,
                current_batch: 32,
                context_tokens: 4096,
                dtype_bytes: 2,
                vram_total_gb: 24.0,
            },
        ),
        (
            "70B fp16 batch 1 on 80GB",
            OomContext {
                model_b_params: 70.0,
                current_batch: 1,
                context_tokens: 4096,
                dtype_bytes: 2,
                vram_total_gb: 80.0,
            },
        ),
    ];
    for (label, ctx) in scenarios {
        println!("{label}:");
        for m in rank_mitigations(ctx) {
            println!("  → {m:?}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ctx() -> OomContext {
        OomContext {
            model_b_params: 7.0,
            current_batch: 32,
            context_tokens: 4096,
            dtype_bytes: 2,
            vram_total_gb: 24.0,
        }
    }

    #[test]
    fn advisor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn batch_reduction_appears_first_when_batch_gt_1() {
        let m = rank_mitigations(sample_ctx());
        assert!(matches!(m[0], Mitigation::ReduceBatch { .. }));
    }

    #[test]
    fn batch_1_skips_batch_reduction() {
        let mut ctx = sample_ctx();
        ctx.current_batch = 1;
        let m = rank_mitigations(ctx);
        assert!(!m
            .iter()
            .any(|x| matches!(x, Mitigation::ReduceBatch { .. })));
    }

    #[test]
    fn always_includes_cpu_fallback_last() {
        let m = rank_mitigations(sample_ctx());
        assert_eq!(m.last(), Some(&Mitigation::DefaultToCpu));
    }

    #[test]
    fn always_includes_gradient_checkpointing() {
        let m = rank_mitigations(sample_ctx());
        assert!(m.contains(&Mitigation::EnableGradientCheckpointing));
    }

    #[test]
    fn lora_suggested_when_model_too_big_for_vram() {
        // 70B × 2 = 140GB > 24GB → suggest LoRA.
        let mut ctx = sample_ctx();
        ctx.model_b_params = 70.0;
        let m = rank_mitigations(ctx);
        assert!(m.contains(&Mitigation::UseLora));
    }

    #[test]
    fn lora_not_suggested_when_model_fits() {
        // 1B × 2 = 2GB ≤ 24GB → no LoRA needed.
        let mut ctx = sample_ctx();
        ctx.model_b_params = 1.0;
        let m = rank_mitigations(ctx);
        assert!(!m.contains(&Mitigation::UseLora));
    }

    #[test]
    fn int8_skipped_for_already_int_dtype() {
        let mut ctx = sample_ctx();
        ctx.dtype_bytes = 1; // already int8
        let m = rank_mitigations(ctx);
        assert!(!m.contains(&Mitigation::SwitchToInt8));
    }
}
