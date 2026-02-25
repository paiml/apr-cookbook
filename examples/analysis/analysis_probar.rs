//! # Layer Activation Snapshots for Visual Regression Testing
//!
//! CLI equivalent: `apr probar model.apr --layers all --compare baseline.json`
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

use apr_cookbook::prelude::*;
use rand::Rng;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Per-layer activation statistics snapshot.
#[derive(Debug, Clone)]
struct LayerSnapshot {
    name: String,
    histogram: Vec<usize>,
    bin_edges: Vec<f64>,
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    kurtosis: f64,
}

/// Comparison result between two snapshots of the same layer.
#[derive(Debug, Clone)]
struct SnapshotDiff {
    layer_name: String,
    mean_delta: f64,
    std_ratio: f64,
    kl_divergence: f64,
    regressed: bool,
}

/// Manifest holding all layer snapshots for a model checkpoint.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ProbarManifest {
    source_model: String,
    layers: Vec<LayerSnapshot>,
}

// ---------------------------------------------------------------------------
// Layer definitions
// ---------------------------------------------------------------------------

/// Layer specification: name, activation count, distribution center, spread.
struct LayerSpec {
    name: &'static str,
    size: usize,
    center: f64,
    spread: f64,
}

fn layer_specs() -> Vec<LayerSpec> {
    vec![
        LayerSpec {
            name: "embed",
            size: 1024,
            center: 0.0,
            spread: 0.3,
        },
        LayerSpec {
            name: "attn_0",
            size: 512,
            center: 0.05,
            spread: 0.5,
        },
        LayerSpec {
            name: "ffn_0",
            size: 512,
            center: 0.1,
            spread: 0.8,
        },
        LayerSpec {
            name: "attn_1",
            size: 512,
            center: 0.02,
            spread: 0.45,
        },
        LayerSpec {
            name: "ffn_1",
            size: 512,
            center: 0.08,
            spread: 0.75,
        },
        LayerSpec {
            name: "output",
            size: 256,
            center: 0.0,
            spread: 1.0,
        },
    ]
}

// ---------------------------------------------------------------------------
// Activation generation
// ---------------------------------------------------------------------------

/// Generate synthetic activations for a layer using the recipe RNG.
fn generate_activations(rng: &mut impl Rng, spec: &LayerSpec) -> Vec<f64> {
    (0..spec.size)
        .map(|_| {
            // Box-Muller approximation via sum of uniforms (central limit)
            let u: f64 = (0..12).map(|_| rng.gen::<f64>()).sum::<f64>() - 6.0;
            spec.center + u * spec.spread
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Compute basic statistics: mean, std, min, max.
fn compute_stats(data: &[f64]) -> (f64, f64, f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (mean, std, min, max)
}

/// Compute excess kurtosis (Fisher definition).
fn compute_kurtosis(data: &[f64], mean: f64, std: f64) -> f64 {
    if data.len() < 4 || std < 1e-12 {
        return 0.0;
    }
    let n = data.len() as f64;
    let m4 = data.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n;
    m4 - 3.0
}

/// Build a histogram with `n_bins` uniform bins spanning [min, max].
fn build_histogram(
    data: &[f64],
    min_val: f64,
    max_val: f64,
    n_bins: usize,
) -> (Vec<usize>, Vec<f64>) {
    let range = max_val - min_val;
    let bin_width = if range < 1e-12 {
        1.0
    } else {
        range / n_bins as f64
    };
    let mut counts = vec![0usize; n_bins];
    let edges: Vec<f64> = (0..=n_bins)
        .map(|i| min_val + i as f64 * bin_width)
        .collect();

    for &val in data {
        let idx = ((val - min_val) / bin_width) as usize;
        let clamped = idx.min(n_bins - 1);
        counts[clamped] += 1;
    }

    (counts, edges)
}

// ---------------------------------------------------------------------------
// Snapshot construction
// ---------------------------------------------------------------------------

/// Compute a full snapshot for a single layer's activations.
fn compute_snapshot(name: &str, activations: &[f64], n_bins: usize) -> LayerSnapshot {
    let (mean, std, min, max) = compute_stats(activations);
    let kurtosis = compute_kurtosis(activations, mean, std);
    let (histogram, bin_edges) = build_histogram(activations, min, max, n_bins);

    LayerSnapshot {
        name: name.to_string(),
        histogram,
        bin_edges,
        mean,
        std,
        min,
        max,
        kurtosis,
    }
}

// ---------------------------------------------------------------------------
// Comparison / regression detection
// ---------------------------------------------------------------------------

/// KL divergence D_KL(P || Q) with Laplace smoothing.
fn kl_divergence(p: &[usize], q: &[usize]) -> f64 {
    let n = p.len().min(q.len());
    if n == 0 {
        return 0.0;
    }
    let total_p: f64 = p.iter().sum::<usize>() as f64 + n as f64;
    let total_q: f64 = q.iter().sum::<usize>() as f64 + n as f64;

    let mut kl = 0.0;
    for i in 0..n {
        let pi = (p[i] as f64 + 1.0) / total_p;
        let qi = (q[i] as f64 + 1.0) / total_q;
        kl += pi * (pi / qi).ln();
    }
    kl
}

/// Compare two snapshots and determine if a regression occurred.
///
/// Regression criteria:
///   - mean shift > 0.1
///   - std ratio outside [0.8, 1.2] (> 20% change)
///   - histogram KL divergence > 0.5
fn compare_snapshots(before: &LayerSnapshot, after: &LayerSnapshot) -> SnapshotDiff {
    let mean_delta = (after.mean - before.mean).abs();
    let std_ratio = if before.std.abs() < 1e-12 {
        1.0
    } else {
        after.std / before.std
    };
    let kl = kl_divergence(&before.histogram, &after.histogram);

    let mean_regressed = mean_delta > 0.1;
    let std_regressed = !(0.8..=1.2).contains(&std_ratio);
    let kl_regressed = kl > 0.5;
    let regressed = mean_regressed || std_regressed || kl_regressed;

    SnapshotDiff {
        layer_name: before.name.clone(),
        mean_delta,
        std_ratio,
        kl_divergence: kl,
        regressed,
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Render a single layer's histogram as an ASCII bar chart.
fn render_histogram_ascii(snapshot: &LayerSnapshot, bar_width: usize) -> String {
    let max_count = snapshot.histogram.iter().copied().max().unwrap_or(1).max(1);
    let mut output = String::new();
    output.push_str(&format!(
        "  Histogram: {} ({} bins)\n",
        snapshot.name,
        snapshot.histogram.len()
    ));

    for (i, &count) in snapshot.histogram.iter().enumerate() {
        let lo = snapshot.bin_edges.get(i).copied().unwrap_or(0.0);
        let hi = snapshot.bin_edges.get(i + 1).copied().unwrap_or(0.0);
        let bar_len = (count as f64 / max_count as f64 * bar_width as f64) as usize;
        let bar: String = "#".repeat(bar_len);
        output.push_str(&format!(
            "  [{:>6.2}, {:>6.2}) |{:<width$}| {}\n",
            lo,
            hi,
            bar,
            count,
            width = bar_width
        ));
    }
    output
}

/// Render the regression diff table.
fn render_diff_table(
    diffs: &[SnapshotDiff],
    before: &[LayerSnapshot],
    after: &[LayerSnapshot],
) -> String {
    let mut output = String::new();
    output.push_str(&format!(
        "  {:<10} {:<12} {:>10} {:>10} {:>10} {:>10}\n",
        "Layer", "Metric", "Before", "After", "Delta", "Verdict"
    ));
    output.push_str(&format!("  {}\n", "-".repeat(66)));

    for diff in diffs {
        let b = before.iter().find(|s| s.name == diff.layer_name);
        let a = after.iter().find(|s| s.name == diff.layer_name);
        let (Some(b_snap), Some(a_snap)) = (b, a) else {
            continue;
        };

        let mean_verdict = if diff.mean_delta > 0.1 {
            "REGRESS"
        } else {
            "OK"
        };
        output.push_str(&format!(
            "  {:<10} {:<12} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
            diff.layer_name, "mean", b_snap.mean, a_snap.mean, diff.mean_delta, mean_verdict
        ));

        let std_verdict = if diff.std_ratio < 0.8 || diff.std_ratio > 1.2 {
            "REGRESS"
        } else {
            "OK"
        };
        output.push_str(&format!(
            "  {:<10} {:<12} {:>10.4} {:>10.4} {:>10.2}x {:>9}\n",
            "", "std", b_snap.std, a_snap.std, diff.std_ratio, std_verdict
        ));

        let kl_verdict = if diff.kl_divergence > 0.5 {
            "REGRESS"
        } else {
            "OK"
        };
        output.push_str(&format!(
            "  {:<10} {:<12} {:>10} {:>10} {:>10.4} {:>10}\n",
            "", "KL div", "-", "-", diff.kl_divergence, kl_verdict
        ));

        let overall = if diff.regressed { "FAIL" } else { "PASS" };
        output.push_str(&format!(
            "  {:<10} {:<12} {:>10} {:>10} {:>10} {:>10}\n",
            "", "overall", "", "", "", overall
        ));
    }
    output
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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
