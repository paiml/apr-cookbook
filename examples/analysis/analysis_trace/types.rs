//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
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
use std::fmt;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TensorStats {
    pub mean: f64,
    pub std: f64,
    pub l2_norm: f64,
    pub min: f64,
    pub max: f64,
    pub nan_count: usize,
    pub inf_count: usize,
    pub zero_count: usize,
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
pub struct LayerTrace {
    pub name: String,
    pub shape: Vec<usize>,
    pub stats: TensorStats,
    pub anomalies: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TraceReport {
    pub total_layers: usize,
    pub anomaly_count: usize,
    pub healthy: bool,
    pub anomalous_layers: Vec<String>,
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

pub fn compute_stats(activations: &[f64]) -> TensorStats {
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

pub fn detect_anomalies(stats: &TensorStats, median_std: f64, l2_threshold: f64) -> Vec<String> {
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

pub fn compute_median_std(traces: &[LayerTrace]) -> f64 {
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

pub fn generate_normal_activations(
    rng: &mut impl Rng,
    size: usize,
    mean: f64,
    std: f64,
) -> Vec<f64> {
    (0..size)
        .map(|_| {
            // Box-Muller transform for normal distribution
            let u1: f64 = rng.gen_range(1e-10..1.0);
            let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
            mean + std * (-2.0 * u1.ln()).sqrt() * u2.cos()
        })
        .collect()
}

pub fn generate_layer_activations(rng: &mut impl Rng, layer_name: &str, size: usize) -> Vec<f64> {
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

pub fn trace_model(layers: &[(&str, Vec<usize>, Vec<f64>)], l2_threshold: f64) -> Vec<LayerTrace> {
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

pub fn build_trace_report(traces: &[LayerTrace]) -> TraceReport {
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

pub fn format_shape(shape: &[usize]) -> String {
    shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
