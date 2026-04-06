//! # APR vs HuggingFace SafeTensors Comparison
//! **CLI Equivalent**: `apr compare-hf`
//!
//! CLI equivalent: `apr compare_hf model.apr --repo my-org/my-model --threshold 1e-5`
//!
//! Performs bit-for-bit tensor comparison between a local APR model and
//! HuggingFace SafeTensors weights. Maps HF naming conventions to APR naming,
//! then computes per-tensor metrics: max absolute error, mean absolute error,
//! cosine similarity, and L2 distance. Reports PASS/FAIL per tensor and overall.

use apr_cookbook::prelude::*;
use rand::Rng;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Per-tensor comparison result between APR and HF formats.
#[derive(Debug, Clone)]
struct TensorComparison {
    name_apr: String,
    name_hf: String,
    shape: Vec<usize>,
    max_abs_error: f64,
    mean_abs_error: f64,
    cosine_similarity: f64,
    l2_distance: f64,
    passed: bool,
}

/// Aggregated comparison report across all tensor pairs.
#[derive(Debug, Clone)]
struct CompareReport {
    comparisons: Vec<TensorComparison>,
    threshold: f64,
    overall_passed: bool,
}

/// A named tensor with shape and float data, format-agnostic.
#[derive(Debug, Clone)]
struct NamedTensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// Mapping entry from HF tensor name to APR tensor name.
#[derive(Debug, Clone)]
struct NameMapping {
    hf_name: String,
    apr_name: String,
}

// ---------------------------------------------------------------------------
// Name mapping: HuggingFace <-> APR conventions
// ---------------------------------------------------------------------------

/// Standard HF-to-APR tensor name mappings for a transformer model.
fn build_name_mappings(n_layers: usize) -> Vec<NameMapping> {
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
fn generate_apr_tensors(
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
fn generate_hf_tensors_from_apr(
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
fn inject_mismatch(tensors: &mut [NamedTensor], target_name: &str, magnitude: f32) {
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
fn tensor_shape_for_name(name: &str, dim: usize) -> Vec<usize> {
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

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
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

fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut sum: f64 = 0.0;
    for i in 0..n {
        let d = f64::from(a[i]) - f64::from(b[i]);
        sum += d * d;
    }
    sum.sqrt()
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f64 {
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

fn mean_abs_error(a: &[f32], b: &[f32]) -> f64 {
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
fn compare_tensor_pair(apr: &NamedTensor, hf: &NamedTensor, threshold: f64) -> TensorComparison {
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
fn compare_models(
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

fn print_report(report: &CompareReport) {
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

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("analysis_compare_hf")?;

    println!("=== APR vs HuggingFace SafeTensors Comparison ===\n");

    // --- Section 1: Build name mappings for a 2-layer transformer ---
    println!("--- Section 1: Building Name Mappings ---");
    let n_layers = 2;
    let dim = 32;
    let mappings = build_name_mappings(n_layers);
    println!(
        "Mapped {} tensor pairs across {n_layers} layers\n",
        mappings.len()
    );

    for m in &mappings {
        println!("  HF:  {:<55} -> APR: {}", m.hf_name, m.apr_name);
    }

    // --- Section 2: Generate synthetic APR and HF tensor sets ---
    println!("\n--- Section 2: Generating Tensor Sets ---");
    let apr_tensors = generate_apr_tensors(ctx.rng(), &mappings, dim);
    let hf_tensors = generate_hf_tensors_from_apr(&apr_tensors, &mappings);

    let total_params: usize = apr_tensors.iter().map(|t| t.data.len()).sum();
    println!(
        "APR tensors: {} tensors, {} total params",
        apr_tensors.len(),
        total_params
    );
    println!(
        "HF  tensors: {} tensors, {} total params",
        hf_tensors.len(),
        total_params
    );

    // --- Section 3: Bit-for-bit comparison (identical data) ---
    println!("\n--- Section 3: Bit-for-Bit Comparison (Identical) ---");
    let threshold = 1e-5;
    let report = compare_models(&apr_tensors, &hf_tensors, &mappings, threshold);
    print_report(&report);
    assert!(
        report.overall_passed,
        "Identical tensor data must produce PASS"
    );

    // --- Section 4: Comparison with injected mismatch ---
    println!("\n--- Section 4: Comparison with Injected Mismatch ---");
    let mismatch_target_hf = "model.layers.1.mlp.gate_proj.weight";
    let mismatch_magnitude = 0.01_f32;
    let mut hf_tensors_corrupted = hf_tensors.clone();
    inject_mismatch(
        &mut hf_tensors_corrupted,
        mismatch_target_hf,
        mismatch_magnitude,
    );
    println!("Injected mismatch: {mismatch_target_hf} += {mismatch_magnitude}");

    let report_mismatch = compare_models(&apr_tensors, &hf_tensors_corrupted, &mappings, threshold);
    print_report(&report_mismatch);
    assert!(
        !report_mismatch.overall_passed,
        "Mismatch injection must cause FAIL"
    );

    // --- Section 5: Per-tensor detail for failed tensors ---
    println!("\n--- Section 5: Failed Tensor Details ---");
    for c in report_mismatch.comparisons.iter().filter(|c| !c.passed) {
        println!("  APR name:     {}", c.name_apr);
        println!("  HF name:      {}", c.name_hf);
        println!("  Shape:        {:?}", c.shape);
        println!("  Max abs err:  {:.2e}", c.max_abs_error);
        println!("  Mean abs err: {:.2e}", c.mean_abs_error);
        println!("  Cosine sim:   {:.8}", c.cosine_similarity);
        println!("  L2 distance:  {:.6}", c.l2_distance);
        println!("  Threshold:    {:.0e}", report_mismatch.threshold);
    }

    // --- Section 6: Summary statistics ---
    println!("\n--- Section 6: Summary ---");
    let passed_clean = report.comparisons.iter().filter(|c| c.passed).count();
    let passed_corrupt = report_mismatch
        .comparisons
        .iter()
        .filter(|c| c.passed)
        .count();
    println!(
        "Clean comparison:   {}/{} PASS",
        passed_clean,
        report.comparisons.len()
    );
    println!(
        "Corrupt comparison: {}/{} PASS",
        passed_corrupt,
        report_mismatch.comparisons.len()
    );

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Vec<NameMapping>, Vec<NamedTensor>, Vec<NamedTensor>) {
        let mut ctx = RecipeContext::new("test_compare_hf").expect("ctx");
        let mappings = build_name_mappings(2);
        let apr = generate_apr_tensors(ctx.rng(), &mappings, 16);
        let hf = generate_hf_tensors_from_apr(&apr, &mappings);
        (mappings, apr, hf)
    }

    #[test]
    fn test_identical_tensors_all_pass() {
        let (mappings, apr, hf) = setup();
        let report = compare_models(&apr, &hf, &mappings, 1e-5);
        assert!(report.overall_passed);
        for c in &report.comparisons {
            assert!(c.passed);
        }
    }

    #[test]
    fn test_identical_tensors_zero_error() {
        let (mappings, apr, hf) = setup();
        let report = compare_models(&apr, &hf, &mappings, 1e-5);
        for c in &report.comparisons {
            assert!(c.max_abs_error < 1e-10, "max_abs_error={}", c.max_abs_error);
            assert!(
                c.mean_abs_error < 1e-10,
                "mean_abs_error={}",
                c.mean_abs_error
            );
            assert!(c.l2_distance < 1e-10, "l2_distance={}", c.l2_distance);
        }
    }

    #[test]
    fn test_identical_tensors_cosine_one() {
        let (mappings, apr, hf) = setup();
        let report = compare_models(&apr, &hf, &mappings, 1e-5);
        for c in &report.comparisons {
            assert!(
                (c.cosine_similarity - 1.0).abs() < 1e-6,
                "cosine={} for {}",
                c.cosine_similarity,
                c.name_apr
            );
        }
    }

    #[test]
    fn test_mismatch_injection_causes_failure() {
        let (mappings, apr, hf) = setup();
        let mut hf_bad = hf;
        inject_mismatch(&mut hf_bad, "model.layers.0.self_attn.q_proj.weight", 0.1);
        let report = compare_models(&apr, &hf_bad, &mappings, 1e-5);
        assert!(!report.overall_passed);
    }

    #[test]
    fn test_small_mismatch_within_threshold_passes() {
        let (mappings, apr, hf) = setup();
        let mut hf_nudged = hf;
        inject_mismatch(
            &mut hf_nudged,
            "model.layers.0.self_attn.q_proj.weight",
            1e-7,
        );
        let report = compare_models(&apr, &hf_nudged, &mappings, 1e-5);
        assert!(report.overall_passed);
    }

    #[test]
    fn test_name_mapping_count() {
        let mappings = build_name_mappings(2);
        // 1 embed + (4 attn + 3 mlp) * 2 layers + 1 lm_head = 16
        assert_eq!(mappings.len(), 16);
    }

    #[test]
    fn test_name_mapping_hf_prefix() {
        let mappings = build_name_mappings(1);
        let layer_mappings: Vec<_> = mappings
            .iter()
            .filter(|m| m.hf_name.starts_with("model.layers."))
            .collect();
        assert_eq!(layer_mappings.len(), 7); // 4 attn + 3 mlp
    }

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let cs = cosine_similarity(&a, &a);
        assert!((cs - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let cs = cosine_similarity(&a, &b);
        assert!(cs.abs() < 1e-6);
    }

    #[test]
    fn test_report_threshold_respected() {
        let (mappings, apr, hf) = setup();
        let strict = compare_models(&apr, &hf, &mappings, 0.0);
        // With threshold 0.0, identical data should still pass (error == 0.0)
        assert!(strict.overall_passed);
        assert!((strict.threshold - 0.0).abs() < f64::EPSILON);
    }
}
