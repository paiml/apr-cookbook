//! # Recipe: Per-Tensor Statistics Histogram
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr tensors model.apr --stats`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example analysis_tensors_stats` exits 0
//! 2. [x] `cargo test --example analysis_tensors_stats` passes
//! 3. [x] Deterministic output (same seed → same mean/std/sparsity)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] NaN count is accurate (explicit injection test)
//!
//! ## Learning Objective
//! Demonstrates per-tensor weight statistics — mean, std, min, max, NaN count,
//! and sparsity — on a synthetic 10-tensor model with mixed FP32/FP16/INT8
//! dtypes and varied shapes. Reveals the health-check pattern Glorot & Bengio
//! first catalogued in 2010: weights drifting toward zero mean with ~1/√n
//! variance is a signature of a properly initialised feed-forward layer.
//!
//! ## Run Command
//! ```bash
//! cargo run --example analysis_tensors_stats
//! ```
//!
//! ## Format Variants
//! ```bash
//! apr tensors model.apr          --stats  # APR native
//! apr tensors model.gguf         --stats  # GGUF
//! apr tensors model.safetensors  --stats  # HF SafeTensors
//! ```
//!
//! ## References
//! - Glorot, X. & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks*. AISTATS. URL: proceedings.mlr.press/v9/glorot10a.html

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use rand::Rng;

/// Tensor element dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    FP32,
    FP16,
    INT8,
}

impl DType {
    pub fn label(self) -> &'static str {
        match self {
            DType::FP32 => "fp32",
            DType::FP16 => "fp16",
            DType::INT8 => "int8",
        }
    }

    pub fn element_bytes(self) -> usize {
        match self {
            DType::FP32 => 4,
            DType::FP16 => 2,
            DType::INT8 => 1,
        }
    }
}

/// Descriptor for one synthetic tensor.
#[derive(Debug, Clone)]
pub struct SyntheticTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub values: Vec<f32>,
}

impl SyntheticTensor {
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.element_bytes()
    }
}

/// Computed statistics for a single tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub nan_count: usize,
    pub sparsity_pct: f32,
}

/// Compute per-tensor statistics, ignoring NaN for mean/std/min/max and
/// reporting sparsity as the fraction of exactly-zero elements.
#[must_use]
pub fn compute_stats(values: &[f32]) -> TensorStats {
    if values.is_empty() {
        return TensorStats {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            nan_count: 0,
            sparsity_pct: 100.0,
        };
    }

    let nan_count = values.iter().filter(|v| v.is_nan()).count();
    let finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();

    if finite.is_empty() {
        return TensorStats {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            nan_count,
            sparsity_pct: 0.0,
        };
    }

    let n = finite.len() as f32;
    let mean = finite.iter().sum::<f32>() / n;
    let var = finite.iter().map(|v| (*v - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let min = finite.iter().copied().fold(f32::INFINITY, f32::min);
    let max = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let zeros = values.iter().filter(|v| **v == 0.0).count();
    let sparsity_pct = 100.0 * zeros as f32 / values.len() as f32;

    TensorStats {
        mean,
        std,
        min,
        max,
        nan_count,
        sparsity_pct,
    }
}

/// Build 10 synthetic tensors with varied dtypes and shapes.
pub fn build_tensors(rng: &mut impl Rng) -> Vec<SyntheticTensor> {
    fn gaussian(rng: &mut impl Rng, shape: &[usize], scale: f32) -> Vec<f32> {
        let n: usize = shape.iter().product();
        (0..n)
            .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
            .collect()
    }

    let mut ts = Vec::with_capacity(10);
    ts.push(SyntheticTensor {
        name: "embed_tokens.weight".into(),
        values: gaussian(rng, &[64, 16], 0.02),
        shape: vec![64, 16],
        dtype: DType::FP16,
    });
    ts.push(SyntheticTensor {
        name: "layers.0.attn.q_proj.weight".into(),
        values: gaussian(rng, &[16, 16], 0.1),
        shape: vec![16, 16],
        dtype: DType::FP32,
    });
    ts.push(SyntheticTensor {
        name: "layers.0.attn.k_proj.weight".into(),
        values: gaussian(rng, &[16, 16], 0.1),
        shape: vec![16, 16],
        dtype: DType::FP32,
    });
    ts.push(SyntheticTensor {
        name: "layers.0.attn.v_proj.weight".into(),
        values: gaussian(rng, &[16, 16], 0.1),
        shape: vec![16, 16],
        dtype: DType::FP32,
    });
    ts.push(SyntheticTensor {
        name: "layers.0.mlp.gate_proj.weight".into(),
        values: gaussian(rng, &[32, 16], 0.05),
        shape: vec![32, 16],
        dtype: DType::FP32,
    });
    // Deliberately sparse tensor — demonstrates the sparsity column.
    let mut sparse = gaussian(rng, &[32, 16], 0.05);
    for (i, v) in sparse.iter_mut().enumerate() {
        if i % 3 == 0 {
            *v = 0.0;
        }
    }
    ts.push(SyntheticTensor {
        name: "layers.0.mlp.up_proj.weight".into(),
        values: sparse,
        shape: vec![32, 16],
        dtype: DType::FP32,
    });
    ts.push(SyntheticTensor {
        name: "layers.0.mlp.down_proj.weight".into(),
        values: gaussian(rng, &[16, 32], 0.05),
        shape: vec![16, 32],
        dtype: DType::INT8,
    });
    ts.push(SyntheticTensor {
        name: "layers.0.norm.weight".into(),
        values: vec![1.0; 16],
        shape: vec![16],
        dtype: DType::FP32,
    });
    ts.push(SyntheticTensor {
        name: "lm_head.weight".into(),
        values: gaussian(rng, &[64, 16], 0.02),
        shape: vec![64, 16],
        dtype: DType::INT8,
    });
    ts.push(SyntheticTensor {
        name: "output_norm.weight".into(),
        values: gaussian(rng, &[16], 0.01),
        shape: vec![16],
        dtype: DType::FP32,
    });
    ts
}

/// Render an aligned plain-text table with per-tensor stats.
#[must_use]
pub fn render_table(tensors: &[SyntheticTensor], stats: &[TensorStats]) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "  {:<32} {:<10} {:<5}   {:>8}   {:>8}   {:>8}   {:>8}   {:>4}   {:>6}\n",
        "name", "shape", "dtype", "mean", "std", "min", "max", "nan", "spars%"
    ));
    s.push_str(&format!("  {}\n", "-".repeat(112)));
    for (t, st) in tensors.iter().zip(stats.iter()) {
        let shape_str = format!("{:?}", t.shape);
        s.push_str(&format!(
            "  {:<32} {:<10} {:<5}   {:>8.4}   {:>8.4}   {:>8.4}   {:>8.4}   {:>4}   {:>5.1}%\n",
            t.name,
            shape_str,
            t.dtype.label(),
            st.mean,
            st.std,
            st.min,
            st.max,
            st.nan_count,
            st.sparsity_pct,
        ));
    }
    s
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("analysis_tensors_stats")?;
    println!("=== Recipe: {} ===\n", ctx.name());

    // --- Section 1: Build 10 synthetic tensors ---------------------------
    let tensors = build_tensors(ctx.rng());
    println!(
        "Built {} synthetic tensors across 3 dtypes.\n",
        tensors.len()
    );

    // --- Section 2: Compute per-tensor statistics ------------------------
    let stats: Vec<TensorStats> = tensors.iter().map(|t| compute_stats(&t.values)).collect();

    // --- Section 3: Render aligned table --------------------------------
    println!("--- Per-Tensor Statistics (apr tensors --stats) ---\n");
    let table = render_table(&tensors, &stats);
    print!("{table}");
    println!();

    // --- Section 4: Dtype breakdown -------------------------------------
    let (mut fp32_bytes, mut fp16_bytes, mut int8_bytes) = (0usize, 0usize, 0usize);
    for t in &tensors {
        match t.dtype {
            DType::FP32 => fp32_bytes += t.size_bytes(),
            DType::FP16 => fp16_bytes += t.size_bytes(),
            DType::INT8 => int8_bytes += t.size_bytes(),
        }
    }
    println!("--- Dtype Breakdown ---");
    println!("  fp32: {} bytes", fp32_bytes);
    println!("  fp16: {} bytes", fp16_bytes);
    println!("  int8: {} bytes", int8_bytes);
    println!();

    // --- Section 5: Persist JSON summary --------------------------------
    let summary = serde_json::json!({
        "schema_version": 1,
        "sub": "tensors",
        "tensors": tensors.iter().zip(stats.iter()).map(|(t, st)| serde_json::json!({
            "name": t.name,
            "shape": t.shape,
            "dtype": t.dtype.label(),
            "mean": st.mean,
            "std": st.std,
            "min": st.min,
            "max": st.max,
            "nan_count": st.nan_count,
            "sparsity_pct": st.sparsity_pct,
        })).collect::<Vec<_>>(),
    });
    let summary_path = ctx.path("tensors_stats.json");
    std::fs::write(
        &summary_path,
        serde_json::to_vec_pretty(&summary)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;
    println!("Wrote {}", summary_path.display());

    // --- Section 6: Metrics --------------------------------------------
    let total_nans: usize = stats.iter().map(|s| s.nan_count).sum();
    let avg_sparsity: f32 = stats.iter().map(|s| s.sparsity_pct).sum::<f32>() / stats.len() as f32;
    ctx.record_metric("tensors_total", tensors.len() as i64);
    ctx.record_metric("nan_total", total_nans as i64);
    ctx.record_float_metric("sparsity_avg_pct", f64::from(avg_sparsity));

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn test_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_build_ten_tensors() {
        let mut rng = test_rng();
        let ts = build_tensors(&mut rng);
        assert_eq!(ts.len(), 10);
    }

    #[test]
    fn test_dtype_element_bytes() {
        assert_eq!(DType::FP32.element_bytes(), 4);
        assert_eq!(DType::FP16.element_bytes(), 2);
        assert_eq!(DType::INT8.element_bytes(), 1);
    }

    #[test]
    fn test_compute_stats_mean_std_min_max() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let st = compute_stats(&data);
        assert!((st.mean - 3.0).abs() < 1e-5);
        assert!((st.min - 1.0).abs() < 1e-5);
        assert!((st.max - 5.0).abs() < 1e-5);
        assert_eq!(st.nan_count, 0);
    }

    #[test]
    fn test_compute_stats_counts_nans() {
        let data = vec![1.0, f32::NAN, 3.0, f32::NAN, f32::NAN];
        let st = compute_stats(&data);
        assert_eq!(st.nan_count, 3);
    }

    #[test]
    fn test_compute_stats_sparsity() {
        let data = vec![0.0, 0.0, 0.0, 1.0];
        let st = compute_stats(&data);
        assert!((st.sparsity_pct - 75.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_stats_empty() {
        let st = compute_stats(&[]);
        assert_eq!(st.nan_count, 0);
        assert!((st.sparsity_pct - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_table_includes_header_and_each_tensor() {
        let mut rng = test_rng();
        let ts = build_tensors(&mut rng);
        let sts: Vec<_> = ts.iter().map(|t| compute_stats(&t.values)).collect();
        let table = render_table(&ts, &sts);
        assert!(table.contains("name"));
        for t in &ts {
            assert!(table.contains(&t.name), "table missing tensor {}", t.name);
        }
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let ts1 = build_tensors(&mut test_rng());
        let ts2 = build_tensors(&mut test_rng());
        assert_eq!(ts1.len(), ts2.len());
        for (a, b) in ts1.iter().zip(ts2.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.shape, b.shape);
            assert_eq!(a.values.len(), b.values.len());
            for (va, vb) in a.values.iter().zip(b.values.iter()) {
                assert!((va - vb).abs() < 1e-6);
            }
        }
    }
}
