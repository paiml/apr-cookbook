//! # APR Model Activation Trace
//!
//! CLI equivalent: `apr trace model.apr --stats --anomalies`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Performs layer-by-layer statistical analysis of model tensor activations,
//! computing per-layer statistics (mean, std, L2 norm, min, max, NaN/Inf counts)
//! and detecting anomalies such as high-variance spikes, dead layers, NaN/Inf
//! presence, and gradient explosion.
//!
//!
//! ## Format Variants
//! ```bash
//! apr trace model.apr          # APR native format
//! apr trace model.gguf         # GGUF (llama.cpp compatible)
//! apr trace model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use rand::Rng;
use std::fmt;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TensorStats {
    mean: f64,
    std: f64,
    l2_norm: f64,
    min: f64,
    max: f64,
    nan_count: usize,
    inf_count: usize,
    zero_count: usize,
}

impl fmt::Display for TensorStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mean={:>8.4} std={:>8.4} L2={:>8.2} min={:>8.4} max={:>8.4}",
            self.mean, self.std, self.l2_norm, self.min, self.max
        )
    }
}

#[derive(Debug, Clone)]
struct LayerTrace {
    name: String,
    shape: Vec<usize>,
    stats: TensorStats,
    anomalies: Vec<String>,
}

#[derive(Debug, Clone)]
struct TraceReport {
    total_layers: usize,
    anomaly_count: usize,
    healthy: bool,
    anomalous_layers: Vec<String>,
}

impl fmt::Display for TraceReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Total layers traced: {}", self.total_layers)?;
        writeln!(f, "Anomalies found:     {}", self.anomaly_count)?;
        writeln!(
            f,
            "Health verdict:      {}",
            if self.healthy { "HEALTHY" } else { "UNHEALTHY" }
        )?;
        if !self.anomalous_layers.is_empty() {
            writeln!(f, "Anomalous layers:")?;
            for name in &self.anomalous_layers {
                writeln!(f, "  - {name}")?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Statistics computation
// ---------------------------------------------------------------------------

fn compute_stats(activations: &[f64]) -> TensorStats {
    if activations.is_empty() {
        return TensorStats {
            mean: 0.0,
            std: 0.0,
            l2_norm: 0.0,
            min: 0.0,
            max: 0.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };
    }

    let nan_count = activations.iter().filter(|v| v.is_nan()).count();
    let inf_count = activations.iter().filter(|v| v.is_infinite()).count();
    let zero_count = activations.iter().filter(|v| **v == 0.0).count();

    // Filter to finite values for statistical calculations
    let finite: Vec<f64> = activations
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();

    if finite.is_empty() {
        return TensorStats {
            mean: f64::NAN,
            std: f64::NAN,
            l2_norm: 0.0,
            min: f64::NAN,
            max: f64::NAN,
            nan_count,
            inf_count,
            zero_count,
        };
    }

    let n = finite.len() as f64;
    let sum: f64 = finite.iter().sum();
    let mean = sum / n;

    let variance = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    let l2_norm = finite.iter().map(|v| v * v).sum::<f64>().sqrt();
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    TensorStats {
        mean,
        std,
        l2_norm,
        min,
        max,
        nan_count,
        inf_count,
        zero_count,
    }
}

// ---------------------------------------------------------------------------
// Anomaly detection
// ---------------------------------------------------------------------------

fn detect_anomalies(stats: &TensorStats, median_std: f64, l2_threshold: f64) -> Vec<String> {
    let mut anomalies = Vec::new();

    if stats.nan_count > 0 {
        anomalies.push(format!("NaN detected ({} values)", stats.nan_count));
    }

    if stats.inf_count > 0 {
        anomalies.push(format!("Inf detected ({} values)", stats.inf_count));
    }

    // High variance spike: std > 3x the median std across all layers
    if median_std > 0.0 && stats.std > 3.0 * median_std {
        anomalies.push(format!(
            "High variance spike (std={:.4}, threshold={:.4})",
            stats.std,
            3.0 * median_std
        ));
    }

    // Dead layer: all values are zero
    if stats.l2_norm == 0.0 && stats.nan_count == 0 && stats.inf_count == 0 {
        anomalies.push("Dead layer (all zeros)".to_string());
    }

    // Gradient explosion: L2 norm exceeds threshold
    if stats.l2_norm > l2_threshold {
        anomalies.push(format!(
            "Gradient explosion (L2={:.2}, threshold={:.2})",
            stats.l2_norm, l2_threshold
        ));
    }

    anomalies
}

fn compute_median_std(traces: &[LayerTrace]) -> f64 {
    let mut stds: Vec<f64> = traces
        .iter()
        .map(|t| t.stats.std)
        .filter(|s| s.is_finite())
        .collect();

    if stds.is_empty() {
        return 0.0;
    }

    stds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = stds.len() / 2;
    if stds.len() % 2 == 0 {
        (stds[mid - 1] + stds[mid]) / 2.0
    } else {
        stds[mid]
    }
}

// ---------------------------------------------------------------------------
// Activation generation
// ---------------------------------------------------------------------------

fn generate_normal_activations(rng: &mut impl Rng, size: usize, mean: f64, std: f64) -> Vec<f64> {
    (0..size)
        .map(|_| {
            // Box-Muller transform for normal distribution
            let u1: f64 = rng.gen_range(1e-10..1.0);
            let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
            mean + std * (-2.0 * u1.ln()).sqrt() * u2.cos()
        })
        .collect()
}

fn generate_layer_activations(rng: &mut impl Rng, layer_name: &str, size: usize) -> Vec<f64> {
    match layer_name {
        "embedding" => generate_normal_activations(rng, size, 0.0, 0.02),
        name if name.starts_with("transformer_") => {
            generate_normal_activations(rng, size, 0.0, 0.1)
        }
        "layernorm" => generate_normal_activations(rng, size, 0.0, 1.0),
        "output" => generate_normal_activations(rng, size, 0.0, 0.5),
        "lm_head" => generate_normal_activations(rng, size, 0.0, 0.3),
        _ => generate_normal_activations(rng, size, 0.0, 0.1),
    }
}

// ---------------------------------------------------------------------------
// Trace pipeline
// ---------------------------------------------------------------------------

fn trace_model(layers: &[(&str, Vec<usize>, Vec<f64>)], l2_threshold: f64) -> Vec<LayerTrace> {
    // First pass: compute stats without anomaly detection
    let mut traces: Vec<LayerTrace> = layers
        .iter()
        .map(|(name, shape, activations)| {
            let stats = compute_stats(activations);
            LayerTrace {
                name: (*name).to_string(),
                shape: shape.clone(),
                stats,
                anomalies: Vec::new(),
            }
        })
        .collect();

    // Compute median std across all layers for spike detection
    let median_std = compute_median_std(&traces);

    // Second pass: detect anomalies using global context
    for trace in &mut traces {
        trace.anomalies = detect_anomalies(&trace.stats, median_std, l2_threshold);
    }

    traces
}

fn build_trace_report(traces: &[LayerTrace]) -> TraceReport {
    let anomalous_layers: Vec<String> = traces
        .iter()
        .filter(|t| !t.anomalies.is_empty())
        .map(|t| t.name.clone())
        .collect();

    let anomaly_count: usize = traces.iter().map(|t| t.anomalies.len()).sum();

    TraceReport {
        total_layers: traces.len(),
        anomaly_count,
        healthy: anomalous_layers.is_empty(),
        anomalous_layers,
    }
}

fn format_shape(shape: &[usize]) -> String {
    shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("analysis_trace")?;

    println!("=== APR Model Activation Trace ===\n");

    // --- Section 1: Generate synthetic model activations ---
    println!("--- Section 1: Generate Synthetic Activations ---");

    let hidden_dim = 256;
    let seq_len = 32;
    let activation_size = hidden_dim * seq_len;

    let layer_defs: Vec<(&str, Vec<usize>)> = vec![
        ("embedding", vec![seq_len, hidden_dim]),
        ("transformer_0", vec![seq_len, hidden_dim]),
        ("transformer_1", vec![seq_len, hidden_dim]),
        ("transformer_2", vec![seq_len, hidden_dim]),
        ("transformer_3", vec![seq_len, hidden_dim]),
        ("transformer_4", vec![seq_len, hidden_dim]),
        ("transformer_5", vec![seq_len, hidden_dim]),
        ("transformer_6", vec![seq_len, hidden_dim]),
        ("transformer_7", vec![seq_len, hidden_dim]),
        ("layernorm", vec![seq_len, hidden_dim]),
        ("output", vec![seq_len, hidden_dim]),
        ("lm_head", vec![seq_len, hidden_dim]),
    ];

    let rng = ctx.rng();
    let mut layers: Vec<(&str, Vec<usize>, Vec<f64>)> = Vec::new();

    for (name, shape) in &layer_defs {
        let activations = generate_layer_activations(rng, name, activation_size);
        layers.push((name, shape.clone(), activations));
    }

    // Inject anomalies for demonstration:
    // 1. Inject NaN values into transformer_3
    if let Some(layer) = layers
        .iter_mut()
        .find(|(name, _, _)| *name == "transformer_3")
    {
        for val in layer.2.iter_mut().take(5) {
            *val = f64::NAN;
        }
        println!("  Injected 5 NaN values into transformer_3");
    }

    // 2. Inject high-variance spike into transformer_6
    if let Some(layer) = layers
        .iter_mut()
        .find(|(name, _, _)| *name == "transformer_6")
    {
        let rng_spike = ctx.rng();
        layer.2 = generate_normal_activations(rng_spike, activation_size, 0.0, 50.0);
        println!("  Injected high-variance spike into transformer_6 (std=50.0)");
    }

    println!("  Generated activations for {} layers\n", layers.len());

    // --- Section 2: Compute per-layer statistics ---
    println!("--- Section 2: Per-Layer Statistics ---");

    let l2_threshold = 1000.0;
    let traces = trace_model(&layers, l2_threshold);

    println!(
        "\n{:<16} {:<10} {:>8} {:>8} {:>10} {:>8} {:>8} {:>4} {:>4}",
        "Layer", "Shape", "Mean", "Std", "L2", "Min", "Max", "NaN", "Inf"
    );
    println!("{}", "-".repeat(90));
    for trace in &traces {
        println!(
            "{:<16} {:<10} {:>8.4} {:>8.4} {:>10.2} {:>8.4} {:>8.4} {:>4} {:>4}",
            trace.name,
            format_shape(&trace.shape),
            trace.stats.mean,
            trace.stats.std,
            trace.stats.l2_norm,
            trace.stats.min,
            trace.stats.max,
            trace.stats.nan_count,
            trace.stats.inf_count,
        );
    }

    // --- Section 3: Anomaly detection ---
    println!("\n--- Section 3: Anomaly Detection ---");

    let median_std = compute_median_std(&traces);
    println!("Median std across layers: {median_std:.4}");
    println!("L2 explosion threshold:   {l2_threshold:.2}\n");

    for trace in &traces {
        if trace.anomalies.is_empty() {
            println!("  {:<16} OK", trace.name);
        } else {
            for anomaly in &trace.anomalies {
                println!("  {:<16} ANOMALY: {}", trace.name, anomaly);
            }
        }
    }

    // --- Section 4: Trace summary ---
    println!("\n--- Section 4: Trace Summary ---");

    let report = build_trace_report(&traces);
    println!("{report}");

    // Verify expected anomalies were detected
    assert!(
        report
            .anomalous_layers
            .contains(&"transformer_3".to_string()),
        "NaN injection in transformer_3 should be detected"
    );
    assert!(
        report
            .anomalous_layers
            .contains(&"transformer_6".to_string()),
        "High-variance spike in transformer_6 should be detected"
    );
    assert!(
        !report.healthy,
        "Model with injected anomalies should be unhealthy"
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

    fn normal_activations(n: usize) -> Vec<f64> {
        let mut ctx = RecipeContext::new("trace_test").expect("context");
        generate_normal_activations(ctx.rng(), n, 0.0, 1.0)
    }

    #[test]
    fn test_compute_stats_normal() {
        let data = normal_activations(10_000);
        let stats = compute_stats(&data);
        assert!(
            stats.mean.abs() < 0.1,
            "Mean of standard normal should be near 0, got {}",
            stats.mean
        );
        assert!(
            (stats.std - 1.0).abs() < 0.15,
            "Std of standard normal should be near 1.0, got {}",
            stats.std
        );
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);
    }

    #[test]
    fn test_compute_stats_empty() {
        let stats = compute_stats(&[]);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std, 0.0);
        assert_eq!(stats.l2_norm, 0.0);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);
        assert_eq!(stats.zero_count, 0);
    }

    #[test]
    fn test_compute_stats_with_nan() {
        let data = vec![1.0, 2.0, f64::NAN, 4.0, f64::NAN];
        let stats = compute_stats(&data);
        assert_eq!(stats.nan_count, 2);
        // Mean should be computed from finite values only: (1+2+4)/3
        let expected_mean = 7.0 / 3.0;
        assert!(
            (stats.mean - expected_mean).abs() < 1e-10,
            "Mean should exclude NaN values"
        );
    }

    #[test]
    fn test_compute_stats_with_inf() {
        let data = vec![1.0, f64::INFINITY, 3.0, f64::NEG_INFINITY];
        let stats = compute_stats(&data);
        assert_eq!(stats.inf_count, 2);
        // Finite values: 1.0, 3.0 -> mean = 2.0
        assert!((stats.mean - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_stats_all_zeros() {
        let data = vec![0.0; 100];
        let stats = compute_stats(&data);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std, 0.0);
        assert_eq!(stats.l2_norm, 0.0);
        assert_eq!(stats.zero_count, 100);
    }

    #[test]
    fn test_detect_nan_anomaly() {
        let stats = TensorStats {
            mean: 0.0,
            std: 0.1,
            l2_norm: 5.0,
            min: -1.0,
            max: 1.0,
            nan_count: 3,
            inf_count: 0,
            zero_count: 0,
        };
        let anomalies = detect_anomalies(&stats, 0.1, 1000.0);
        assert!(anomalies.iter().any(|a| a.contains("NaN")));
    }

    #[test]
    fn test_detect_inf_anomaly() {
        let stats = TensorStats {
            mean: 0.0,
            std: 0.1,
            l2_norm: 5.0,
            min: -1.0,
            max: 1.0,
            nan_count: 0,
            inf_count: 2,
            zero_count: 0,
        };
        let anomalies = detect_anomalies(&stats, 0.1, 1000.0);
        assert!(anomalies.iter().any(|a| a.contains("Inf")));
    }

    #[test]
    fn test_detect_high_variance_spike() {
        let stats = TensorStats {
            mean: 0.0,
            std: 50.0,
            l2_norm: 100.0,
            min: -200.0,
            max: 200.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };
        // Median std = 0.1, threshold = 0.3; 50.0 >> 0.3
        let anomalies = detect_anomalies(&stats, 0.1, 1000.0);
        assert!(anomalies.iter().any(|a| a.contains("High variance")));
    }

    #[test]
    fn test_detect_dead_layer() {
        let stats = TensorStats {
            mean: 0.0,
            std: 0.0,
            l2_norm: 0.0,
            min: 0.0,
            max: 0.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 100,
        };
        let anomalies = detect_anomalies(&stats, 0.1, 1000.0);
        assert!(anomalies.iter().any(|a| a.contains("Dead layer")));
    }

    #[test]
    fn test_detect_gradient_explosion() {
        let stats = TensorStats {
            mean: 5.0,
            std: 10.0,
            l2_norm: 5000.0,
            min: -100.0,
            max: 100.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };
        let anomalies = detect_anomalies(&stats, 0.1, 1000.0);
        assert!(anomalies.iter().any(|a| a.contains("Gradient explosion")));
    }

    #[test]
    fn test_no_anomalies_for_healthy_layer() {
        let stats = TensorStats {
            mean: 0.01,
            std: 0.1,
            l2_norm: 5.0,
            min: -0.5,
            max: 0.5,
            nan_count: 0,
            inf_count: 0,
            zero_count: 10,
        };
        let anomalies = detect_anomalies(&stats, 0.1, 1000.0);
        assert!(
            anomalies.is_empty(),
            "Healthy layer should have no anomalies"
        );
    }

    #[test]
    fn test_trace_report_healthy() {
        let traces = vec![LayerTrace {
            name: "layer_0".to_string(),
            shape: vec![32, 64],
            stats: TensorStats {
                mean: 0.0,
                std: 0.1,
                l2_norm: 10.0,
                min: -0.5,
                max: 0.5,
                nan_count: 0,
                inf_count: 0,
                zero_count: 0,
            },
            anomalies: vec![],
        }];
        let report = build_trace_report(&traces);
        assert!(report.healthy);
        assert_eq!(report.anomaly_count, 0);
        assert_eq!(report.total_layers, 1);
    }

    #[test]
    fn test_trace_report_unhealthy() {
        let traces = vec![
            LayerTrace {
                name: "good_layer".to_string(),
                shape: vec![32, 64],
                stats: TensorStats {
                    mean: 0.0,
                    std: 0.1,
                    l2_norm: 10.0,
                    min: -0.5,
                    max: 0.5,
                    nan_count: 0,
                    inf_count: 0,
                    zero_count: 0,
                },
                anomalies: vec![],
            },
            LayerTrace {
                name: "bad_layer".to_string(),
                shape: vec![32, 64],
                stats: TensorStats {
                    mean: f64::NAN,
                    std: f64::NAN,
                    l2_norm: 0.0,
                    min: f64::NAN,
                    max: f64::NAN,
                    nan_count: 100,
                    inf_count: 0,
                    zero_count: 0,
                },
                anomalies: vec!["NaN detected (100 values)".to_string()],
            },
        ];
        let report = build_trace_report(&traces);
        assert!(!report.healthy);
        assert_eq!(report.anomaly_count, 1);
        assert_eq!(report.anomalous_layers, vec!["bad_layer"]);
    }

    #[test]
    fn test_compute_median_std_odd() {
        let traces = vec![
            make_trace("a", 0.1),
            make_trace("b", 0.5),
            make_trace("c", 0.3),
        ];
        let median = compute_median_std(&traces);
        assert!(
            (median - 0.3).abs() < 1e-10,
            "Median of [0.1, 0.3, 0.5] = 0.3, got {median}"
        );
    }

    fn make_trace(name: &str, std: f64) -> LayerTrace {
        LayerTrace {
            name: name.to_string(),
            shape: vec![1],
            stats: TensorStats {
                mean: 0.0,
                std,
                l2_norm: 1.0,
                min: -1.0,
                max: 1.0,
                nan_count: 0,
                inf_count: 0,
                zero_count: 0,
            },
            anomalies: vec![],
        }
    }
}
