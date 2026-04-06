#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use rand::Rng;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Per-tensor comparison result between APR and HF formats.
#[derive(Debug, Clone)]
pub struct TensorComparison {
    pub name_apr: String,
    pub name_hf: String,
    pub shape: Vec<usize>,
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
    pub cosine_similarity: f64,
    pub l2_distance: f64,
    pub passed: bool,
}

/// Aggregated comparison report across all tensor pairs.
#[derive(Debug, Clone)]
pub struct CompareReport {
    pub comparisons: Vec<TensorComparison>,
    pub threshold: f64,
    pub overall_passed: bool,
}

/// A named tensor with shape and float data, format-agnostic.
#[derive(Debug, Clone)]
pub struct NamedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Mapping entry from HF tensor name to APR tensor name.
#[derive(Debug, Clone)]
pub struct NameMapping {
    pub hf_name: String,
    pub apr_name: String,
}

// ---------------------------------------------------------------------------
// Name mapping: HuggingFace <-> APR conventions
// ---------------------------------------------------------------------------

/// Standard HF-to-APR tensor name mappings for a transformer model.
pub fn build_name_mappings(n_layers: usize) -> Vec<NameMapping> {
    let mut mappings = Vec::new();

    // Embedding layer
    mappings.push(NameMapping {
        hf_name: "model.embed_tokens.weight".into(),
        apr_name: "embed_tokens.weight".into(),
    });

    // Per-layer mappings
    for layer in 0..n_layers {
        let hf_attn = &[
            ("self_attn.q_proj.weight", "attention.query.weight"),
            ("self_attn.k_proj.weight", "attention.key.weight"),
            ("self_attn.v_proj.weight", "attention.value.weight"),
            ("self_attn.o_proj.weight", "attention.output.weight"),
        ];
        for &(hf_suffix, apr_suffix) in hf_attn {
            mappings.push(NameMapping {
                hf_name: format!("model.layers.{layer}.{hf_suffix}"),
                apr_name: format!("layers.{layer}.{apr_suffix}"),
            });
        }

        let hf_mlp = &[
            ("mlp.gate_proj.weight", "ffn.gate.weight"),
            ("mlp.up_proj.weight", "ffn.up.weight"),
            ("mlp.down_proj.weight", "ffn.down.weight"),
        ];
        for &(hf_suffix, apr_suffix) in hf_mlp {
            mappings.push(NameMapping {
                hf_name: format!("model.layers.{layer}.{hf_suffix}"),
                apr_name: format!("layers.{layer}.{apr_suffix}"),
            });
        }
    }

    // LM head
    mappings.push(NameMapping {
        hf_name: "lm_head.weight".into(),
        apr_name: "lm_head.weight".into(),
    });

    mappings
}

// ---------------------------------------------------------------------------
// Tensor generation
// ---------------------------------------------------------------------------

/// Generate a deterministic tensor set in APR naming convention.
pub fn generate_apr_tensors(
    rng: &mut impl Rng,
    mappings: &[NameMapping],
    dim: usize,
) -> Vec<NamedTensor> {
    mappings
        .iter()
        .map(|m| {
            let shape = tensor_shape_for_name(&m.apr_name, dim);
            let n_elems = shape.iter().product::<usize>();
            let data: Vec<f32> = (0..n_elems).map(|_| rng.gen_range(-1.0..1.0)).collect();
            NamedTensor {
                name: m.apr_name.clone(),
                shape,
                data,
            }
        })
        .collect()
}

/// Generate a matching HF tensor set from APR tensors (bit-for-bit identical).
pub fn generate_hf_tensors_from_apr(
    apr_tensors: &[NamedTensor],
    mappings: &[NameMapping],
) -> Vec<NamedTensor> {
    mappings
        .iter()
        .filter_map(|m| {
            let apr_tensor = apr_tensors.iter().find(|t| t.name == m.apr_name)?;
            Some(NamedTensor {
                name: m.hf_name.clone(),
                shape: apr_tensor.shape.clone(),
                data: apr_tensor.data.clone(),
            })
        })
        .collect()
}

/// Inject a controlled mismatch into one tensor's data.
pub fn inject_mismatch(tensors: &mut [NamedTensor], target_name: &str, magnitude: f32) {
    for tensor in tensors.iter_mut() {
        if tensor.name == target_name {
            for val in &mut tensor.data {
                *val += magnitude;
            }
            return;
        }
    }
}

/// Determine a realistic shape for a tensor given its name and model dimension.
pub fn tensor_shape_for_name(name: &str, dim: usize) -> Vec<usize> {
    if name.contains("embed_tokens") || name.contains("lm_head") {
        vec![128, dim] // vocab_size x dim
    } else if name.contains("gate") || name.contains("up") {
        vec![dim * 2, dim] // intermediate x dim
    } else if name.contains("down") {
        vec![dim, dim * 2] // dim x intermediate
    } else {
        vec![dim, dim] // attention projections
    }
}

// ---------------------------------------------------------------------------
// Comparison logic
// ---------------------------------------------------------------------------

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for i in 0..n {
        let va = f64::from(a[i]);
        let vb = f64::from(b[i]);
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    (dot / denom).clamp(-1.0, 1.0)
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut sum: f64 = 0.0;
    for i in 0..n {
        let d = f64::from(a[i]) - f64::from(b[i]);
        sum += d * d;
    }
    sum.sqrt()
}

pub fn max_abs_error(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut max_e: f64 = 0.0;
    for i in 0..n {
        let d = (f64::from(a[i]) - f64::from(b[i])).abs();
        if d > max_e {
            max_e = d;
        }
    }
    max_e
}

pub fn mean_abs_error(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum: f64 = 0.0;
    for i in 0..n {
        sum += (f64::from(a[i]) - f64::from(b[i])).abs();
    }
    sum / n as f64
}

/// Compare a single tensor pair and return metrics.
pub fn compare_tensor_pair(
    apr: &NamedTensor,
    hf: &NamedTensor,
    threshold: f64,
) -> TensorComparison {
    let max_err = max_abs_error(&apr.data, &hf.data);
    let mean_err = mean_abs_error(&apr.data, &hf.data);
    let cosine = cosine_similarity(&apr.data, &hf.data);
    let l2 = l2_distance(&apr.data, &hf.data);
    let passed = max_err <= threshold;

    TensorComparison {
        name_apr: apr.name.clone(),
        name_hf: hf.name.clone(),
        shape: apr.shape.clone(),
        max_abs_error: max_err,
        mean_abs_error: mean_err,
        cosine_similarity: cosine,
        l2_distance: l2,
        passed,
    }
}

/// Run the full comparison between APR and HF tensor sets.
pub fn compare_models(
    apr_tensors: &[NamedTensor],
    hf_tensors: &[NamedTensor],
    mappings: &[NameMapping],
    threshold: f64,
) -> CompareReport {
    let mut comparisons = Vec::new();

    for mapping in mappings {
        let apr = apr_tensors.iter().find(|t| t.name == mapping.apr_name);
        let hf = hf_tensors.iter().find(|t| t.name == mapping.hf_name);

        if let (Some(a), Some(h)) = (apr, hf) {
            comparisons.push(compare_tensor_pair(a, h, threshold));
        }
    }

    let overall_passed = comparisons.iter().all(|c| c.passed);

    CompareReport {
        comparisons,
        threshold,
        overall_passed,
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

pub fn print_report(report: &CompareReport) {
    println!(
        "\n{:<40} {:<14} {:>10} {:>10} {:>8}",
        "Tensor (APR)", "Shape", "MaxErr", "Cosine", "Verdict"
    );
    println!("{}", "-".repeat(90));
    for c in &report.comparisons {
        let shape_str = format!("{:?}", c.shape);
        let verdict = if c.passed { "PASS" } else { "FAIL" };
        println!(
            "{:<40} {:<14} {:>10.2e} {:>10.8} {:>8}",
            c.name_apr, shape_str, c.max_abs_error, c.cosine_similarity, verdict
        );
    }

    let pass_count = report.comparisons.iter().filter(|c| c.passed).count();
    let total = report.comparisons.len();
    let overall = if report.overall_passed {
        "PASS"
    } else {
        "FAIL"
    };
    println!(
        "\nOverall: {overall} ({pass_count}/{total} tensors within threshold {:.0e})",
        report.threshold
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
