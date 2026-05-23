//! # apr serve plan — VRAM Capacity Estimator
//!
//! `apr serve plan <FILE> --max-batch <B> --max-context <C>` estimates
//! the inference capacity envelope: weights memory + KV cache + activation
//! peak. This recipe builds the calculator and asserts the contract: KV
//! cache scales O(batch × context × n_layers × hidden), activation peak
//! is per-batch (not per-token), final answer fits in u64 without overflow.
//!
//! Demonstrates the **SERVE-PLAN.4** recipe for PMAT-105 (apr serve coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SERVE-001 + KV-cache memory model
//!
//! Run with: cargo run --example cli_serve_plan_capacity_estimator
//!
//! Added by PMAT-105 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy)]
pub struct ModelTopo {
    pub weights_bytes: u64,
    pub n_layers: u32,
    pub hidden_size: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub dtype_bytes: u32, // 2 for fp16/bf16, 4 for fp32
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CapacityPlan {
    pub weights_bytes: u64,
    pub kv_cache_bytes: u64,
    pub activation_peak_bytes: u64,
    pub total_bytes: u64,
}

pub fn estimate_capacity(topo: ModelTopo, max_batch: u32, max_context: u32) -> CapacityPlan {
    // KV cache = batch × context × n_layers × n_kv_heads × head_dim × 2 (K + V) × dtype_bytes
    let kv_cache = u64::from(max_batch)
        * u64::from(max_context)
        * u64::from(topo.n_layers)
        * u64::from(topo.n_kv_heads)
        * u64::from(topo.head_dim)
        * 2
        * u64::from(topo.dtype_bytes);
    // Activation peak: roughly batch × hidden × dtype × small constant for transformer FF.
    let activation =
        u64::from(max_batch) * u64::from(topo.hidden_size) * u64::from(topo.dtype_bytes) * 4;
    let total = topo.weights_bytes + kv_cache + activation;
    CapacityPlan {
        weights_bytes: topo.weights_bytes,
        kv_cache_bytes: kv_cache,
        activation_peak_bytes: activation,
        total_bytes: total,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_serve_plan_capacity_estimator")?;

    let qwen3_7b = ModelTopo {
        weights_bytes: 7_000_000_000 * 2, // bf16
        n_layers: 28,
        hidden_size: 3584,
        n_kv_heads: 4,
        head_dim: 128,
        dtype_bytes: 2,
    };

    for (label, batch, ctx) in [
        ("decode 1×8K", 1u32, 8192u32),
        ("decode 4×8K", 4, 8192),
        ("batch 32×4K", 32, 4096),
        ("128K context", 1, 131_072),
    ] {
        let plan = estimate_capacity(qwen3_7b, batch, ctx);
        println!(
            "{label:>15}: weights={} GB, kv={} GB, act={} GB, total={} GB",
            plan.weights_bytes / 1_000_000_000,
            plan.kv_cache_bytes / 1_000_000_000,
            plan.activation_peak_bytes / 1_000_000_000,
            plan.total_bytes / 1_000_000_000
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_topo() -> ModelTopo {
        ModelTopo {
            weights_bytes: 1_000_000_000,
            n_layers: 16,
            hidden_size: 1024,
            n_kv_heads: 8,
            head_dim: 64,
            dtype_bytes: 2,
        }
    }

    #[test]
    fn estimator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn weights_bytes_unchanged() {
        let plan = estimate_capacity(sample_topo(), 1, 1024);
        assert_eq!(plan.weights_bytes, sample_topo().weights_bytes);
    }

    #[test]
    fn kv_cache_scales_linearly_with_batch() {
        let p1 = estimate_capacity(sample_topo(), 1, 1024);
        let p2 = estimate_capacity(sample_topo(), 2, 1024);
        assert_eq!(p2.kv_cache_bytes, p1.kv_cache_bytes * 2);
    }

    #[test]
    fn kv_cache_scales_linearly_with_context() {
        let p1 = estimate_capacity(sample_topo(), 1, 1024);
        let p2 = estimate_capacity(sample_topo(), 1, 2048);
        assert_eq!(p2.kv_cache_bytes, p1.kv_cache_bytes * 2);
    }

    #[test]
    fn fp32_doubles_kv_vs_bf16() {
        let mut topo = sample_topo();
        topo.dtype_bytes = 4;
        let plan = estimate_capacity(topo, 1, 1024);
        topo.dtype_bytes = 2;
        let baseline = estimate_capacity(topo, 1, 1024);
        assert_eq!(plan.kv_cache_bytes, baseline.kv_cache_bytes * 2);
    }

    #[test]
    fn total_equals_sum_of_components() {
        let plan = estimate_capacity(sample_topo(), 4, 8192);
        assert_eq!(
            plan.total_bytes,
            plan.weights_bytes + plan.kv_cache_bytes + plan.activation_peak_bytes
        );
    }

    #[test]
    fn large_batch_does_not_overflow() {
        // u32 × u32 × u32 fits in u64 with widening.
        let plan = estimate_capacity(sample_topo(), u32::MAX / 2, 1);
        assert!(plan.total_bytes > 0);
    }

    #[test]
    fn zero_batch_yields_zero_kv_and_activation() {
        let plan = estimate_capacity(sample_topo(), 0, 1024);
        assert_eq!(plan.kv_cache_bytes, 0);
        assert_eq!(plan.activation_peak_bytes, 0);
    }
}
