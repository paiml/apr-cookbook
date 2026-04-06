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

/// Per-layer activation statistics snapshot.
#[derive(Debug, Clone)]
pub struct LayerSnapshot {
    pub name: String,
    pub histogram: Vec<usize>,
    pub bin_edges: Vec<f64>,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub kurtosis: f64,
}

/// Comparison result between two snapshots of the same layer.
#[derive(Debug, Clone)]
pub struct SnapshotDiff {
    pub layer_name: String,
    pub mean_delta: f64,
    pub std_ratio: f64,
    pub kl_divergence: f64,
    pub regressed: bool,
}

/// Manifest holding all layer snapshots for a model checkpoint.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ProbarManifest {
    pub source_model: String,
    pub layers: Vec<LayerSnapshot>,
}

// ---------------------------------------------------------------------------
// Layer definitions
// ---------------------------------------------------------------------------

/// Layer specification: name, activation count, distribution center, spread.
pub struct LayerSpec {
    pub name: &'static str,
    pub size: usize,
    pub center: f64,
    pub spread: f64,
}

pub fn layer_specs() -> Vec<LayerSpec> {
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
pub fn generate_activations(rng: &mut impl Rng, spec: &LayerSpec) -> Vec<f64> {
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
pub fn compute_stats(data: &[f64]) -> (f64, f64, f64, f64) {
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
pub fn compute_kurtosis(data: &[f64], mean: f64, std: f64) -> f64 {
    if data.len() < 4 || std < 1e-12 {
        return 0.0;
    }
    let n = data.len() as f64;
    let m4 = data.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n;
    m4 - 3.0
}

/// Build a histogram with `n_bins` uniform bins spanning [min, max].
pub fn build_histogram(
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
pub fn compute_snapshot(name: &str, activations: &[f64], n_bins: usize) -> LayerSnapshot {
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
pub fn kl_divergence(p: &[usize], q: &[usize]) -> f64 {
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

// Compare two snapshots and determine if a regression occurred.
//
// Regression criteria:
//   - mean shift > 0.1
//   - std ratio outside [0.8, 1.2] (> 20% change)
///   - histogram KL divergence > 0.5
pub fn compare_snapshots(before: &LayerSnapshot, after: &LayerSnapshot) -> SnapshotDiff {
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
pub fn render_histogram_ascii(snapshot: &LayerSnapshot, bar_width: usize) -> String {
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
pub fn render_diff_table(
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
