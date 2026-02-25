//! # APR Tensor Listing
//!
//! CLI equivalent: `apr tensors model.apr [--stats]`
//!
//! Lists all tensors in a model file with shape, dtype, size, and optional
//! statistics (mean, std, min, max, NaN count, sparsity). Prints a compact
//! table sorted by size (largest first) with a total summary and dtype
//! breakdown.

use apr_cookbook::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Data type for a tensor's elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DType {
    FP32,
    FP16,
    INT8,
}

impl DType {
    /// Bytes consumed by a single element of this dtype.
    fn element_bytes(self) -> usize {
        match self {
            DType::FP32 => 4,
            DType::FP16 => 2,
            DType::INT8 => 1,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::FP32 => f.write_str("FP32"),
            DType::FP16 => f.write_str("FP16"),
            DType::INT8 => f.write_str("INT8"),
        }
    }
}

/// Metadata and weight data for a single tensor.
#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    shape: Vec<usize>,
    dtype: DType,
    data: Vec<f32>,
}

impl TensorInfo {
    fn param_count(&self) -> usize {
        self.shape.iter().product()
    }

    fn size_bytes(&self) -> usize {
        self.param_count() * self.dtype.element_bytes()
    }
}

/// Statistics computed over a tensor's weight data.
#[derive(Debug, Clone)]
struct TensorStats {
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    nan_count: usize,
    sparsity_pct: f64,
}

// ---------------------------------------------------------------------------
// Tensor generation
// ---------------------------------------------------------------------------

/// Maximum number of sample elements stored in `data` for statistics.
/// Keeps memory bounded regardless of tensor shape.
const MAX_SAMPLE: usize = 4096;

/// Build a single tensor with deterministic random weights.
///
/// The `shape` records the full logical dimensions (used for param_count /
/// size_bytes), while `data` stores at most `MAX_SAMPLE` representative
/// elements so that stats can be computed without allocating gigabytes.
fn make_tensor(rng: &mut impl Rng, name: &str, shape: Vec<usize>, dtype: DType) -> TensorInfo {
    let n = shape.iter().product::<usize>().min(MAX_SAMPLE);
    let data: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect();
    TensorInfo {
        name: name.to_string(),
        shape,
        dtype,
        data,
    }
}

/// Build the 15-tensor, 2-layer transformer model.
fn build_model(rng: &mut impl Rng) -> Vec<TensorInfo> {
    let mut tensors = Vec::with_capacity(15);

    // Embedding
    tensors.push(make_tensor(
        rng,
        "embed_tokens.weight",
        vec![32000, 4096],
        DType::FP16,
    ));

    // Layer 0 — attention
    for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
        let name = format!("layers.0.self_attn.{proj}.weight");
        tensors.push(make_tensor(rng, &name, vec![4096, 4096], DType::FP32));
    }
    // Layer 0 — MLP
    tensors.push(make_tensor(
        rng,
        "layers.0.mlp.gate_proj.weight",
        vec![11008, 4096],
        DType::FP32,
    ));
    tensors.push(make_tensor(
        rng,
        "layers.0.mlp.up_proj.weight",
        vec![11008, 4096],
        DType::FP32,
    ));
    tensors.push(make_tensor(
        rng,
        "layers.0.mlp.down_proj.weight",
        vec![4096, 11008],
        DType::FP32,
    ));

    // Layer 1 — attention
    for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
        let name = format!("layers.1.self_attn.{proj}.weight");
        tensors.push(make_tensor(rng, &name, vec![4096, 4096], DType::FP16));
    }
    // Layer 1 — MLP (gate only, INT8 quantized)
    tensors.push(make_tensor(
        rng,
        "layers.1.mlp.gate_proj.weight",
        vec![11008, 4096],
        DType::INT8,
    ));

    // RMS norm
    tensors.push(make_tensor(rng, "norm.weight", vec![4096], DType::FP32));

    // LM head
    tensors.push(make_tensor(
        rng,
        "lm_head.weight",
        vec![32000, 4096],
        DType::INT8,
    ));

    tensors
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Compute statistics for a tensor's weight data.
fn compute_stats(data: &[f32]) -> TensorStats {
    if data.is_empty() {
        return TensorStats {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            nan_count: 0,
            sparsity_pct: 100.0,
        };
    }

    let mut nan_count = 0usize;
    let mut zero_count = 0usize;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;

    for &v in data {
        let vf = f64::from(v);
        if v.is_nan() {
            nan_count += 1;
            continue;
        }
        if v == 0.0 {
            zero_count += 1;
        }
        sum += vf;
        sum_sq += vf * vf;
        if vf < lo {
            lo = vf;
        }
        if vf > hi {
            hi = vf;
        }
    }

    let valid = data.len() - nan_count;
    let mean = if valid > 0 { sum / valid as f64 } else { 0.0 };
    let variance = if valid > 1 {
        (sum_sq / valid as f64) - (mean * mean)
    } else {
        0.0
    };
    let std = variance.max(0.0).sqrt();
    let sparsity_pct = if data.is_empty() {
        0.0
    } else {
        (zero_count as f64 / data.len() as f64) * 100.0
    };

    TensorStats {
        mean,
        std,
        min: if lo.is_infinite() { 0.0 } else { lo },
        max: if hi.is_infinite() { 0.0 } else { hi },
        nan_count,
        sparsity_pct,
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

fn format_size(bytes: usize) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn format_params(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{n}")
    }
}

fn format_shape(shape: &[usize]) -> String {
    shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

/// Build a dtype breakdown from a tensor list.
fn dtype_breakdown(tensors: &[TensorInfo]) -> HashMap<DType, (usize, usize)> {
    let mut map: HashMap<DType, (usize, usize)> = HashMap::new();
    for t in tensors {
        let entry = map.entry(t.dtype).or_insert((0, 0));
        entry.0 += t.param_count();
        entry.1 += t.size_bytes();
    }
    map
}

// ---------------------------------------------------------------------------
// Table printing
// ---------------------------------------------------------------------------

fn print_tensor_table(tensors: &[TensorInfo], show_stats: bool) {
    if show_stats {
        println!(
            "{:<42} {:>16} {:>6} {:>12} {:>10}  {:>8} {:>8} {:>10} {:>10} {:>5} {:>8}",
            "Name",
            "Shape",
            "DType",
            "Params",
            "Size",
            "Mean",
            "Std",
            "Min",
            "Max",
            "NaN",
            "Sparse%",
        );
        println!("{}", "-".repeat(142));
    } else {
        println!(
            "{:<42} {:>16} {:>6} {:>12} {:>10}",
            "Name", "Shape", "DType", "Params", "Size",
        );
        println!("{}", "-".repeat(90));
    }

    for t in tensors {
        let shape_str = format_shape(&t.shape);
        let params_str = format_params(t.param_count());
        let size_str = format_size(t.size_bytes());

        if show_stats {
            let stats = compute_stats(&t.data);
            println!(
                "{:<42} {:>16} {:>6} {:>12} {:>10}  {:>8.4} {:>8.4} {:>10.4} {:>10.4} {:>5} {:>7.2}%",
                t.name, shape_str, t.dtype, params_str, size_str,
                stats.mean, stats.std, stats.min, stats.max,
                stats.nan_count, stats.sparsity_pct,
            );
        } else {
            println!(
                "{:<42} {:>16} {:>6} {:>12} {:>10}",
                t.name, shape_str, t.dtype, params_str, size_str,
            );
        }
    }
}

fn print_summary(tensors: &[TensorInfo]) {
    let total_params: usize = tensors.iter().map(TensorInfo::param_count).sum();
    let total_bytes: usize = tensors.iter().map(TensorInfo::size_bytes).sum();

    println!("\n--- Summary ---");
    println!("  Tensors:      {}", tensors.len());
    println!(
        "  Total params: {} ({})",
        total_params,
        format_params(total_params)
    );
    println!("  Total size:   {}", format_size(total_bytes));

    println!("\n--- DType Breakdown ---");
    let breakdown = dtype_breakdown(tensors);
    let mut entries: Vec<_> = breakdown.iter().collect();
    entries.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    for (dtype, (params, bytes)) in &entries {
        let pct = if total_bytes > 0 {
            (*bytes as f64 / total_bytes as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "  {:<6} {:>12} params  {:>12}  ({:.1}%)",
            dtype,
            format_params(*params),
            format_size(*bytes),
            pct,
        );
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("analysis_tensors")?;

    println!("=== APR Tensor Listing ===\n");

    // --- Section 1: Build synthetic 2-layer transformer ---
    let mut tensors = build_model(ctx.rng());
    println!(
        "Built synthetic 2-layer transformer with {} tensors.\n",
        tensors.len()
    );

    // --- Section 2: Sort by size (largest first) ---
    tensors.sort_by_key(|t| std::cmp::Reverse(t.size_bytes()));

    // --- Section 3: Compact table (no stats) ---
    println!("--- Tensor Table ---");
    print_tensor_table(&tensors, false);

    // --- Section 4: Table with stats ---
    println!("\n--- Tensor Table (--stats) ---");
    print_tensor_table(&tensors, true);

    // --- Section 5: Summary and dtype breakdown ---
    print_summary(&tensors);

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small model with the same 15-tensor structure but tiny dims
    /// so tests finish in milliseconds.
    fn small_model() -> Vec<TensorInfo> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut ts = Vec::with_capacity(15);

        ts.push(make_tensor(
            &mut rng,
            "embed_tokens.weight",
            vec![32, 8],
            DType::FP16,
        ));
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let name = format!("layers.0.self_attn.{proj}.weight");
            ts.push(make_tensor(&mut rng, &name, vec![8, 8], DType::FP32));
        }
        ts.push(make_tensor(
            &mut rng,
            "layers.0.mlp.gate_proj.weight",
            vec![16, 8],
            DType::FP32,
        ));
        ts.push(make_tensor(
            &mut rng,
            "layers.0.mlp.up_proj.weight",
            vec![16, 8],
            DType::FP32,
        ));
        ts.push(make_tensor(
            &mut rng,
            "layers.0.mlp.down_proj.weight",
            vec![8, 16],
            DType::FP32,
        ));
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let name = format!("layers.1.self_attn.{proj}.weight");
            ts.push(make_tensor(&mut rng, &name, vec![8, 8], DType::FP16));
        }
        ts.push(make_tensor(
            &mut rng,
            "layers.1.mlp.gate_proj.weight",
            vec![16, 8],
            DType::INT8,
        ));
        ts.push(make_tensor(&mut rng, "norm.weight", vec![8], DType::FP32));
        ts.push(make_tensor(
            &mut rng,
            "lm_head.weight",
            vec![32, 8],
            DType::INT8,
        ));
        ts
    }

    #[test]
    fn test_model_has_15_tensors() {
        let tensors = small_model();
        assert_eq!(tensors.len(), 15);
    }

    #[test]
    fn test_dtype_element_bytes() {
        assert_eq!(DType::FP32.element_bytes(), 4);
        assert_eq!(DType::FP16.element_bytes(), 2);
        assert_eq!(DType::INT8.element_bytes(), 1);
    }

    #[test]
    fn test_param_count_2d() {
        let t = TensorInfo {
            name: "w".to_string(),
            shape: vec![100, 200],
            dtype: DType::FP32,
            data: vec![],
        };
        assert_eq!(t.param_count(), 20_000);
    }

    #[test]
    fn test_size_bytes_fp16() {
        let t = TensorInfo {
            name: "w".to_string(),
            shape: vec![1000, 512],
            dtype: DType::FP16,
            data: vec![],
        };
        assert_eq!(t.size_bytes(), 1000 * 512 * 2);
    }

    #[test]
    fn test_compute_stats_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_stats(&data);
        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
        assert_eq!(stats.nan_count, 0);
        assert!((stats.sparsity_pct - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats_empty() {
        let stats = compute_stats(&[]);
        assert!((stats.mean - 0.0).abs() < 1e-6);
        assert!((stats.std - 0.0).abs() < 1e-6);
        assert_eq!(stats.nan_count, 0);
        assert!((stats.sparsity_pct - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats_with_nans() {
        let data = vec![1.0, f32::NAN, 3.0, f32::NAN, 5.0];
        let stats = compute_stats(&data);
        assert_eq!(stats.nan_count, 2);
        assert!((stats.mean - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats_sparsity() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 2.0];
        let stats = compute_stats(&data);
        assert!((stats.sparsity_pct - 60.0).abs() < 1e-6);
    }

    #[test]
    fn test_dtype_breakdown_totals() {
        let tensors = small_model();
        let breakdown = dtype_breakdown(&tensors);
        let total_bytes: usize = breakdown.values().map(|(_, b)| b).sum();
        let expected: usize = tensors.iter().map(TensorInfo::size_bytes).sum();
        assert_eq!(total_bytes, expected);
    }

    #[test]
    fn test_sort_by_size_descending() {
        let mut tensors = small_model();
        tensors.sort_by_key(|t| std::cmp::Reverse(t.size_bytes()));
        for pair in tensors.windows(2) {
            assert!(
                pair[0].size_bytes() >= pair[1].size_bytes(),
                "{} ({}) should be >= {} ({})",
                pair[0].name,
                pair[0].size_bytes(),
                pair[1].name,
                pair[1].size_bytes(),
            );
        }
    }
}
