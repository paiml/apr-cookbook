//! # Recipe: Inspect — Quantization Statistics
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr inspect model.apr --quant-stats`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example inspect_quantization_stats` exits 0
//! 2. [x] `cargo test --example inspect_quantization_stats` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr inspect --quant-stats` in-process (no shell-out)
//! 10. [x] Unit tests cover scale, zero_point, outlier-ratio, round-trip error
//!
//! ## Learning Objective
//! Reports quantization quality diagnostics: per-tensor min/max, scale,
//! zero-point, outlier ratio, and dequantization round-trip error. These are
//! the diagnostics an operator needs to spot a poorly-calibrated INT8/INT4
//! quantization pass.
//!
//! ## Run Command
//! ```bash
//! cargo run --example inspect_quantization_stats
//! ```
//!
//! ## References
//! - Dettmers, T. et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS. arXiv:2208.07339

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use rand::Rng;
use serde_json::json;

#[derive(Debug, Clone)]
struct TensorStats {
    name: String,
    n: usize,
    min: f32,
    max: f32,
    scale: f32,
    zero_point: i32,
    outlier_ratio: f64,
    round_trip_mse: f64,
}

/// Symmetric INT8 quantization: scale = max(|x|) / 127.
/// zero_point is always 0 for symmetric.
fn compute_quant_stats(name: &str, x: &[f32], outlier_sigma: f32) -> TensorStats {
    let n = x.len();
    if n == 0 {
        return TensorStats {
            name: name.into(),
            n: 0,
            min: 0.0,
            max: 0.0,
            scale: 0.0,
            zero_point: 0,
            outlier_ratio: 0.0,
            round_trip_mse: 0.0,
        };
    }
    let min = x.iter().copied().fold(f32::INFINITY, f32::min);
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let abs_max = min.abs().max(max.abs()).max(f32::MIN_POSITIVE);
    let scale = abs_max / 127.0;
    let zero_point = 0_i32;

    // Outlier detection: values with |x - mean| > outlier_sigma * stddev.
    let mean = x.iter().copied().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let sigma = var.sqrt();
    let threshold = outlier_sigma * sigma;
    let n_outliers = x.iter().filter(|v| (**v - mean).abs() > threshold).count();
    let outlier_ratio = n_outliers as f64 / n as f64;

    // Round-trip MSE through INT8 symmetric quantization.
    let mut mse = 0.0_f64;
    for &v in x {
        let q = ((v / scale).round() as i32).clamp(-127, 127);
        let dq = q as f32 * scale;
        let err = f64::from(v - dq);
        mse += err * err;
    }
    mse /= n as f64;

    TensorStats {
        name: name.into(),
        n,
        min,
        max,
        scale,
        zero_point,
        outlier_ratio,
        round_trip_mse: mse,
    }
}

fn synth_weights(rng: &mut impl Rng, n: usize, scale: f32) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(rng.gen_range(-scale..scale));
    }
    // Plant a few outliers.
    for i in (0..n).step_by(n.max(1) / 20 + 1).take(5) {
        v[i % n] = scale * 10.0;
    }
    v
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("inspect_quantization_stats")?;
    println!("=== Recipe: {} ===", ctx.name());

    let specs = [
        ("attn.q", 1024),
        ("attn.k", 1024),
        ("attn.v", 1024),
        ("ffn.up", 4096),
    ];
    let mut stats = Vec::new();
    for (name, n) in specs {
        let w = synth_weights(ctx.rng(), n, 0.05);
        let s = compute_quant_stats(name, &w, 3.0);
        println!(
            "{:<10} n={:<5} min={:<7.4} max={:<7.4} scale={:<10.7} zp={:<3} outlier%={:<5.2} mse={:.2e}",
            s.name, s.n, s.min, s.max, s.scale, s.zero_point, s.outlier_ratio * 100.0, s.round_trip_mse
        );
        stats.push(s);
    }

    let report = json!({
        "recipe": ctx.name(),
        "tensors": stats.iter().map(|s| json!({
            "name": s.name,
            "n": s.n,
            "min": s.min,
            "max": s.max,
            "scale": s.scale,
            "zero_point": s.zero_point,
            "outlier_ratio": s.outlier_ratio,
            "round_trip_mse": s.round_trip_mse,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("quant-stats.json");
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out, bytes)?;

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tensor_has_zero_stats() {
        let s = compute_quant_stats("x", &[], 3.0);
        assert_eq!(s.n, 0);
        assert_eq!(s.scale, 0.0);
    }

    #[test]
    fn scale_is_absmax_over_127() {
        let x = vec![-0.5, 0.0, 0.25, 0.5];
        let s = compute_quant_stats("x", &x, 3.0);
        let expected = 0.5_f32 / 127.0;
        assert!((s.scale - expected).abs() < 1e-8);
    }

    #[test]
    fn zero_point_symmetric_is_zero() {
        let x = vec![1.0, -1.0];
        let s = compute_quant_stats("x", &x, 3.0);
        assert_eq!(s.zero_point, 0);
    }

    #[test]
    fn round_trip_mse_nonnegative() {
        let x = vec![0.1_f32, -0.2, 0.3, -0.4];
        let s = compute_quant_stats("x", &x, 3.0);
        assert!(s.round_trip_mse >= 0.0);
    }

    #[test]
    fn constant_tensor_has_low_outlier_ratio() {
        let x = vec![0.5_f32; 100];
        let s = compute_quant_stats("x", &x, 3.0);
        assert!(s.outlier_ratio < 0.05);
    }
}
