#![allow(unused_imports)]
//! # Layer Activation Snapshots for Visual Regression Testing
//!
//! CLI equivalent: `apr probar model.apr --layers all --compare baseline.json`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Exports per-layer activation statistics (histogram, mean, std, min, max,
//! kurtosis) and compares two snapshots to detect regressions. Regression
//! criteria: mean shift > 0.1, std change > 20%, or histogram KL divergence > 0.5.
//!
//! ## What this demonstrates
//! - Deterministic synthetic activation generation per layer
//! - Histogram binning and KL divergence computation
//! - Statistical snapshot comparison for regression testing
//! - ASCII bar chart rendering of activation distributions
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
    let mut ctx = RecipeContext::new("analysis_probar")?;

    println!("=== APR Layer Activation Snapshots (probar) ===\n");

    let specs = layer_specs();
    let n_bins = 10;

    // --- Section 1: Generate baseline activations and snapshots ---
    println!("--- Section 1: Baseline Activation Snapshots ---");
    let mut baseline_snapshots = Vec::new();
    for spec in &specs {
        let activations = generate_activations(ctx.rng(), spec);
        let snapshot = compute_snapshot(spec.name, &activations, n_bins);
        println!(
            "  {:<10} mean={:>7.4}  std={:>7.4}  min={:>7.4}  max={:>7.4}  kurt={:>7.4}",
            snapshot.name,
            snapshot.mean,
            snapshot.std,
            snapshot.min,
            snapshot.max,
            snapshot.kurtosis
        );
        baseline_snapshots.push(snapshot);
    }

    let baseline_manifest = ProbarManifest {
        source_model: "model-v1-baseline".to_string(),
        layers: baseline_snapshots.clone(),
    };
    println!(
        "  Manifest: {} layers from '{}'",
        baseline_manifest.layers.len(),
        baseline_manifest.source_model
    );

    // --- Section 2: ASCII histograms for baseline ---
    println!("\n--- Section 2: Baseline Histograms ---");
    for snapshot in &baseline_snapshots {
        let chart = render_histogram_ascii(snapshot, 30);
        println!("{chart}");
    }

    // --- Section 3: Generate post-optimization activations ---
    println!("--- Section 3: Post-Optimization Snapshots ---");
    let mut optimized_snapshots = Vec::new();
    for spec in &specs {
        // Simulate optimization: shift center slightly, tighten spread
        let opt_spec = LayerSpec {
            name: spec.name,
            size: spec.size,
            center: spec.center + 0.05,
            spread: spec.spread * 0.9,
        };
        let activations = generate_activations(ctx.rng(), &opt_spec);
        let snapshot = compute_snapshot(spec.name, &activations, n_bins);
        println!(
            "  {:<10} mean={:>7.4}  std={:>7.4}  min={:>7.4}  max={:>7.4}  kurt={:>7.4}",
            snapshot.name,
            snapshot.mean,
            snapshot.std,
            snapshot.min,
            snapshot.max,
            snapshot.kurtosis
        );
        optimized_snapshots.push(snapshot);
    }

    // --- Section 4: Compare snapshots for regression ---
    println!("\n--- Section 4: Regression Analysis ---");
    let mut diffs = Vec::new();
    for (before, after) in baseline_snapshots.iter().zip(optimized_snapshots.iter()) {
        let diff = compare_snapshots(before, after);
        diffs.push(diff);
    }

    let table = render_diff_table(&diffs, &baseline_snapshots, &optimized_snapshots);
    println!("{table}");

    // --- Section 5: Summary ---
    println!("--- Section 5: Summary ---");
    let total_layers = diffs.len();
    let regressed_count = diffs.iter().filter(|d| d.regressed).count();
    let passed_count = total_layers - regressed_count;
    println!("  Total layers:   {total_layers}");
    println!("  Passed:         {passed_count}");
    println!("  Regressed:      {regressed_count}");

    if regressed_count > 0 {
        println!("\n  Regressed layers:");
        for diff in diffs.iter().filter(|d| d.regressed) {
            println!(
                "    - {} (mean_delta={:.4}, std_ratio={:.2}x, KL={:.4})",
                diff.layer_name, diff.mean_delta, diff.std_ratio, diff.kl_divergence
            );
        }
    }

    let verdict = if regressed_count == 0 { "PASS" } else { "FAIL" };
    println!("\n  Overall verdict: {verdict}");

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<f64> {
        let mut ctx = RecipeContext::new("probar_test").expect("context");
        let spec = LayerSpec {
            name: "test",
            size: 256,
            center: 0.0,
            spread: 1.0,
        };
        generate_activations(ctx.rng(), &spec)
    }

    #[test]
    fn test_compute_stats_known_values() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, std, min, max) = compute_stats(&data);
        assert!((mean - 3.0).abs() < 1e-10);
        assert!((std - (2.0_f64).sqrt()).abs() < 1e-10);
        assert!((min - 1.0).abs() < 1e-10);
        assert!((max - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_stats_empty() {
        let (mean, std, min, max) = compute_stats(&[]);
        assert!((mean).abs() < 1e-10);
        assert!((std).abs() < 1e-10);
        assert!((min).abs() < 1e-10);
        assert!((max).abs() < 1e-10);
    }

    #[test]
    fn test_kurtosis_normal_distribution() {
        // For a large sample from near-normal distribution, excess kurtosis ~ 0
        let data = sample_data();
        let (mean, std, _, _) = compute_stats(&data);
        let kurt = compute_kurtosis(&data, mean, std);
        // Approximate: for sum-of-12-uniforms, kurtosis should be near zero
        assert!(
            kurt.abs() < 1.0,
            "Kurtosis {kurt} should be near zero for near-normal data"
        );
    }

    #[test]
    fn test_histogram_bin_counts_sum_to_total() {
        let data = sample_data();
        let (_, _, min, max) = compute_stats(&data);
        let (counts, _) = build_histogram(&data, min, max, 10);
        let total: usize = counts.iter().sum();
        assert_eq!(total, data.len(), "All values must be counted");
    }

    #[test]
    fn test_histogram_edges_monotonic() {
        let data = sample_data();
        let (_, _, min, max) = compute_stats(&data);
        let (_, edges) = build_histogram(&data, min, max, 10);
        for i in 1..edges.len() {
            assert!(
                edges[i] >= edges[i - 1],
                "Bin edges must be monotonically non-decreasing"
            );
        }
    }

    #[test]
    fn test_kl_divergence_identical_distributions() {
        let p = vec![10, 20, 30, 20, 10];
        let q = vec![10, 20, 30, 20, 10];
        let kl = kl_divergence(&p, &q);
        assert!(
            kl < 1e-10,
            "KL divergence of identical distributions should be ~0, got {kl}"
        );
    }

    #[test]
    fn test_kl_divergence_different_distributions() {
        let p = vec![100, 0, 0, 0, 0];
        let q = vec![0, 0, 0, 0, 100];
        let kl = kl_divergence(&p, &q);
        assert!(
            kl > 0.0,
            "KL divergence of very different distributions must be positive"
        );
    }

    #[test]
    fn test_compare_identical_snapshots_no_regression() {
        let data = sample_data();
        let snap = compute_snapshot("layer", &data, 10);
        let diff = compare_snapshots(&snap, &snap);
        assert!(!diff.regressed, "Identical snapshots must not regress");
        assert!(diff.mean_delta < 1e-10);
        assert!((diff.std_ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compare_shifted_mean_triggers_regression() {
        let data_before: Vec<f64> = (0..200).map(|i| i as f64 * 0.01).collect();
        let data_after: Vec<f64> = data_before.iter().map(|x| x + 0.5).collect();
        let before = compute_snapshot("layer", &data_before, 10);
        let after = compute_snapshot("layer", &data_after, 10);
        let diff = compare_snapshots(&before, &after);
        assert!(
            diff.regressed,
            "Mean shift of 0.5 should trigger regression"
        );
        assert!(diff.mean_delta > 0.1);
    }

    #[test]
    fn test_render_histogram_ascii_contains_layer_name() {
        let data = sample_data();
        let snap = compute_snapshot("embed", &data, 10);
        let rendered = render_histogram_ascii(&snap, 20);
        assert!(
            rendered.contains("embed"),
            "Histogram should include layer name"
        );
        assert!(
            rendered.contains('#'),
            "Histogram should include bar characters"
        );
    }
}
