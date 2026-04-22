//! # Recipe: Memory Profiling with Per-Layer Breakdown
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr profile --mode memory --breakdown layers model.apr`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example profile_memory_layers` exits 0
//! 2. [x] `cargo test --example profile_memory_layers` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr profile --mode memory` in-process (no shell-out)
//! 10. [x] Unit tests cover weight bytes, activation bytes, total, tail-layer spotting
//!
//! ## Learning Objective
//! Demonstrates per-layer memory profiling: weights, activations, and KV-cache
//! bytes. Reports hottest layers by total bytes and by tail-variance (the
//! Dean/Barroso "tail at scale" lens). Matches `apr profile --mode memory`.
//!
//! ## Run Command
//! ```bash
//! cargo run --example profile_memory_layers
//! ```
//!
//! ## References
//! - Dean, J. & Barroso, L. A. (2013). *The Tail at Scale*. CACM. DOI: 10.1145/2408776.2408794

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSpec {
    pub name: String,
    pub d_model: usize,
    pub d_ff: usize,
    pub seq_len: usize,
    pub dtype_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct LayerMemory {
    pub name: String,
    pub weight_bytes: u64,
    pub activation_bytes: u64,
    pub kv_cache_bytes: u64,
    pub total_bytes: u64,
}

pub fn profile_layer(spec: &LayerSpec) -> LayerMemory {
    let attn_weights = 4u64 * (spec.d_model as u64).pow(2);
    let ffn_weights = 2u64 * (spec.d_model as u64) * (spec.d_ff as u64);
    let weight_bytes = (attn_weights + ffn_weights) * spec.dtype_bytes as u64;
    let activation_bytes =
        (spec.seq_len as u64) * (spec.d_model as u64) * spec.dtype_bytes as u64 * 3;
    let kv_cache_bytes = 2u64 * spec.seq_len as u64 * spec.d_model as u64 * spec.dtype_bytes as u64;
    LayerMemory {
        name: spec.name.clone(),
        weight_bytes,
        activation_bytes,
        kv_cache_bytes,
        total_bytes: weight_bytes + activation_bytes + kv_cache_bytes,
    }
}

pub fn tail_variance(layers: &[LayerMemory]) -> f64 {
    if layers.is_empty() {
        return 0.0;
    }
    let mean = layers.iter().map(|l| l.total_bytes as f64).sum::<f64>() / layers.len() as f64;
    let var = layers
        .iter()
        .map(|l| {
            let d = l.total_bytes as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / layers.len() as f64;
    var.sqrt()
}

pub fn hottest_layer(layers: &[LayerMemory]) -> Option<&LayerMemory> {
    layers.iter().max_by_key(|l| l.total_bytes)
}

fn build_layers() -> Vec<LayerSpec> {
    (0..12)
        .map(|i| LayerSpec {
            name: format!("layer.{}", i),
            d_model: 768,
            d_ff: 3072,
            seq_len: 1024,
            dtype_bytes: 2, // fp16
        })
        .collect()
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("profile_memory_layers")?;
    println!("=== Recipe: {} ===", ctx.name());

    let specs = build_layers();
    let mems: Vec<LayerMemory> = specs.iter().map(profile_layer).collect();
    let total: u64 = mems.iter().map(|l| l.total_bytes).sum();
    let sigma = tail_variance(&mems);
    let hottest = hottest_layer(&mems);

    println!(
        "{:<12} {:>14} {:>14} {:>14} {:>14}",
        "layer", "weight_bytes", "activations", "kv_cache", "total"
    );
    for m in &mems {
        println!(
            "{:<12} {:>14} {:>14} {:>14} {:>14}",
            m.name, m.weight_bytes, m.activation_bytes, m.kv_cache_bytes, m.total_bytes
        );
    }
    if let Some(h) = hottest {
        println!("Hottest: {} ({} bytes)", h.name, h.total_bytes);
    }
    println!("Total: {} bytes; tail-sigma: {:.2}", total, sigma);

    let report = json!({
        "recipe": ctx.name(),
        "total_bytes": total,
        "tail_sigma": sigma,
        "hottest": hottest.map(|h| h.name.clone()),
        "layers": mems.iter().map(|m| json!({
            "name": m.name,
            "weight_bytes": m.weight_bytes,
            "activation_bytes": m.activation_bytes,
            "kv_cache_bytes": m.kv_cache_bytes,
            "total_bytes": m.total_bytes,
        })).collect::<Vec<_>>(),
    });
    let path = ctx.path("profile-memory-layers.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    ctx.record_metric("total_bytes", total as i64);
    ctx.record_float_metric("tail_sigma", sigma);
    ctx.record_metric("n_layers", mems.len() as i64);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(d: usize, ff: usize, seq: usize, b: usize) -> LayerSpec {
        LayerSpec {
            name: "x".into(),
            d_model: d,
            d_ff: ff,
            seq_len: seq,
            dtype_bytes: b,
        }
    }

    #[test]
    fn weight_bytes_scale_with_d_squared() {
        let a = profile_layer(&spec(128, 256, 64, 2));
        let b = profile_layer(&spec(256, 512, 64, 2));
        assert!(b.weight_bytes > 3 * a.weight_bytes);
    }

    #[test]
    fn activation_bytes_scale_with_seq() {
        let a = profile_layer(&spec(128, 256, 32, 2));
        let b = profile_layer(&spec(128, 256, 128, 2));
        assert_eq!(b.activation_bytes, 4 * a.activation_bytes);
    }

    #[test]
    fn dtype_bytes_double_for_fp32() {
        let fp16 = profile_layer(&spec(128, 256, 64, 2));
        let fp32 = profile_layer(&spec(128, 256, 64, 4));
        assert_eq!(fp32.total_bytes, 2 * fp16.total_bytes);
    }

    #[test]
    fn tail_variance_zero_for_identical_layers() {
        let mems = (0..3)
            .map(|i| LayerMemory {
                name: format!("l{}", i),
                weight_bytes: 10,
                activation_bytes: 10,
                kv_cache_bytes: 10,
                total_bytes: 30,
            })
            .collect::<Vec<_>>();
        assert_eq!(tail_variance(&mems), 0.0);
    }

    #[test]
    fn hottest_none_on_empty() {
        assert!(hottest_layer(&[]).is_none());
    }
}
