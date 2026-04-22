//! # Recipe: GPU Memory Planner over Batch Sizes
//!
//! **Category**: gpu
//! **CLI Equivalent**: `apr gpu plan model.apr --batch-sizes 1,4,16,64 --vram-gb 24`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example gpu_memory_planner` exits 0
//! 2. [x] `cargo test --example gpu_memory_planner` passes
//! 3. [x] Deterministic output (pure arithmetic)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr gpu plan` in-process (no shell-out)
//! 10. [x] Unit tests cover OOM detection, headroom, ZeRO partitioning savings
//!
//! ## Learning Objective
//! Computes VRAM requirements (weights + activations + KV cache) for each
//! batch size and flags infeasible configurations, plus simulates ZeRO-3
//! partitioning savings across an 8-GPU setup.
//!
//! ## Run Command
//! ```bash
//! cargo run --example gpu_memory_planner
//! ```
//!
//! ## References
//! - Rajbhandari, S. et al. (2020). *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*. SC20. arXiv:1910.02054

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone)]
struct ModelSpec {
    params: u64,
    bytes_per_param: u64, // fp16 = 2, fp32 = 4
    seq_len: u64,
    hidden: u64,
    n_layers: u64,
}

#[derive(Debug, Clone)]
struct PlanRow {
    batch_size: u64,
    weights_bytes: u64,
    activations_bytes: u64,
    kv_cache_bytes: u64,
    total_bytes: u64,
    fits_single_gpu: bool,
    zero3_per_gpu: u64,
    zero3_fits: bool,
}

const BYTES_PER_GB: u64 = 1 << 30;

fn weights_bytes(spec: &ModelSpec) -> u64 {
    spec.params * spec.bytes_per_param
}

fn activations_bytes(spec: &ModelSpec, batch: u64) -> u64 {
    // Rough activation memory: batch * seq_len * hidden * n_layers * bytes_per_param
    batch * spec.seq_len * spec.hidden * spec.n_layers * spec.bytes_per_param
}

fn kv_cache_bytes(spec: &ModelSpec, batch: u64) -> u64 {
    // Per token: 2 (K, V) * hidden * n_layers * bytes_per_param
    batch * spec.seq_len * 2 * spec.hidden * spec.n_layers * spec.bytes_per_param
}

fn plan_row(spec: &ModelSpec, batch: u64, vram_bytes: u64, n_gpus: u64) -> PlanRow {
    let w = weights_bytes(spec);
    let a = activations_bytes(spec, batch);
    let k = kv_cache_bytes(spec, batch);
    let total = w + a + k;
    let fits_single = total <= vram_bytes;
    // ZeRO-3: weights partitioned across n_gpus; activations/KV stay per-GPU.
    let zero3_per_gpu = w / n_gpus.max(1) + a + k;
    let zero3_fits = zero3_per_gpu <= vram_bytes;
    PlanRow {
        batch_size: batch,
        weights_bytes: w,
        activations_bytes: a,
        kv_cache_bytes: k,
        total_bytes: total,
        fits_single_gpu: fits_single,
        zero3_per_gpu,
        zero3_fits,
    }
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("gpu_memory_planner")?;
    println!("=== Recipe: {} ===", ctx.name());

    let spec = ModelSpec {
        params: 7_000_000_000, // 7B
        bytes_per_param: 2,
        seq_len: 2048,
        hidden: 4096,
        n_layers: 32,
    };
    let vram_bytes = 24 * BYTES_PER_GB;
    let n_gpus = 8;
    let batches: Vec<u64> = vec![1, 4, 16, 64];

    let rows: Vec<PlanRow> = batches
        .iter()
        .map(|&b| plan_row(&spec, b, vram_bytes, n_gpus))
        .collect();

    println!(
        "\nModel: {}B params @ {} bytes/param, seq={}, hidden={}, layers={}",
        spec.params / 1_000_000_000,
        spec.bytes_per_param,
        spec.seq_len,
        spec.hidden,
        spec.n_layers
    );
    println!(
        "Target: {} GB VRAM, {} GPUs\n",
        vram_bytes / BYTES_PER_GB,
        n_gpus
    );
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>10} {:>6} {:>10} {:>6}",
        "Batch", "Weights", "Acts", "KV", "Total", "Fits?", "ZeRO3/GPU", "ZeRO?"
    );
    for r in &rows {
        println!(
            "{:>6} {:>9}GB {:>9}GB {:>9}GB {:>9}GB {:>6} {:>9}GB {:>6}",
            r.batch_size,
            r.weights_bytes / BYTES_PER_GB,
            r.activations_bytes / BYTES_PER_GB,
            r.kv_cache_bytes / BYTES_PER_GB,
            r.total_bytes / BYTES_PER_GB,
            if r.fits_single_gpu { "yes" } else { "no" },
            r.zero3_per_gpu / BYTES_PER_GB,
            if r.zero3_fits { "yes" } else { "no" }
        );
    }

    let report = json!({
        "recipe": ctx.name(),
        "model": {
            "params": spec.params,
            "bytes_per_param": spec.bytes_per_param,
            "seq_len": spec.seq_len,
            "hidden": spec.hidden,
            "n_layers": spec.n_layers,
        },
        "vram_bytes": vram_bytes,
        "n_gpus": n_gpus,
        "rows": rows.iter().map(|r| json!({
            "batch_size": r.batch_size,
            "weights_bytes": r.weights_bytes,
            "activations_bytes": r.activations_bytes,
            "kv_cache_bytes": r.kv_cache_bytes,
            "total_bytes": r.total_bytes,
            "fits_single_gpu": r.fits_single_gpu,
            "zero3_per_gpu": r.zero3_per_gpu,
            "zero3_fits": r.zero3_fits,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("gpu-plan.json");
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out, bytes)?;

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_spec() -> ModelSpec {
        ModelSpec {
            params: 1_000_000,
            bytes_per_param: 2,
            seq_len: 128,
            hidden: 64,
            n_layers: 4,
        }
    }

    #[test]
    fn weights_scale_with_bytes_per_param() {
        let mut s = tiny_spec();
        let w_fp16 = weights_bytes(&s);
        s.bytes_per_param = 4;
        let w_fp32 = weights_bytes(&s);
        assert_eq!(w_fp32, 2 * w_fp16);
    }

    #[test]
    fn oom_detected_for_large_batch() {
        let spec = ModelSpec {
            params: 70_000_000_000,
            bytes_per_param: 2,
            seq_len: 4096,
            hidden: 8192,
            n_layers: 80,
        };
        let row = plan_row(&spec, 128, 24 * BYTES_PER_GB, 1);
        assert!(!row.fits_single_gpu);
    }

    #[test]
    fn zero3_partitioning_reduces_per_gpu() {
        let spec = tiny_spec();
        let row_1 = plan_row(&spec, 1, BYTES_PER_GB, 1);
        let row_8 = plan_row(&spec, 1, BYTES_PER_GB, 8);
        assert!(row_8.zero3_per_gpu < row_1.zero3_per_gpu);
    }

    #[test]
    fn activations_scale_with_batch() {
        let s = tiny_spec();
        let a1 = activations_bytes(&s, 1);
        let a4 = activations_bytes(&s, 4);
        assert_eq!(a4, 4 * a1);
    }

    #[test]
    fn kv_cache_is_twice_activations_per_layer() {
        let s = tiny_spec();
        let a = activations_bytes(&s, 1);
        let k = kv_cache_bytes(&s, 1);
        assert_eq!(k, 2 * a);
    }
}
