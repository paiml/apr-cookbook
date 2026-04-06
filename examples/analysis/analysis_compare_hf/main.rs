#![allow(unused_imports)]
//! # APR vs HuggingFace SafeTensors Comparison
//! **CLI Equivalent**: `apr compare-hf`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! CLI equivalent: `apr compare-hf model.apr --repo my-org/my-model --threshold 1e-5`
//!
//! Performs bit-for-bit tensor comparison between a local APR model and
//! HuggingFace SafeTensors weights. Maps HF naming conventions to APR naming,
//! then computes per-tensor metrics: max absolute error, mean absolute error,
//! cosine similarity, and L2 distance. Reports PASS/FAIL per tensor and overall.
//!
//!
//! ## Format Variants
//! ```bash
//! apr inspect model.apr          # APR native format
//! apr inspect model.gguf         # GGUF (llama.cpp compatible)
//! apr inspect model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use rand::Rng;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
