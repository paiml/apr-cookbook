//! Model Drift Detection for Production ML Monitoring
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Demonstrates PSI, KS statistic, confidence monitoring, and accuracy
//! degradation detection for production ML systems using only `std`.
//!
//! ```bash
//! cargo run --example model_drift_detection
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

const PSI_NUM_BINS: usize = 10;
const NUM_FEATURES: usize = 4;
const WINDOW_SIZE: usize = 50;
const PSI_EPSILON: f64 = 1e-6;
const FEATURE_NAMES: [&str; NUM_FEATURES] = ["age", "income", "credit_score", "debt_ratio"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

fn psi_severity(psi: f64) -> Severity {
    if psi < 0.1 {
        Severity::Low
    } else if psi < 0.2 {
        Severity::Medium
    } else if psi < 0.3 {
        Severity::High
    } else {
        Severity::Critical
    }
}

fn ks_severity(ks: f64) -> Severity {
    if ks < 0.15 {
        Severity::Low
    } else if ks < 0.25 {
        Severity::Medium
    } else if ks < 0.4 {
        Severity::High
    } else {
        Severity::Critical
    }
}

#[derive(Debug, Clone)]
struct DriftAlert {
    source: String,
    severity: Severity,
    message: String,
    metric_value: f64,
}

impl fmt::Display for DriftAlert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}: {} (value={:.4})",
            self.severity, self.source, self.message, self.metric_value
        )
    }
}

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.state.hash(&mut hasher);
        self.state = hasher.finish();
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    fn next_normal(&mut self, mean: f64, std: f64) -> f64 {
        let sum: f64 = (0..12).map(|_| self.next_f64()).sum();
        mean + (sum - 6.0) * std
    }
}

// --- PSI ---

fn histogram_proportions(values: &[f64], min_val: f64, max_val: f64, num_bins: usize) -> Vec<f64> {
    let mut counts = vec![0u64; num_bins];
    let range = max_val - min_val;
    let bin_width = if range.abs() < f64::EPSILON {
        1.0
    } else {
        range / num_bins as f64
    };
    for &v in values {
        let idx = ((v - min_val) / bin_width).floor() as usize;
        counts[idx.min(num_bins - 1)] += 1;
    }
    let total = values.len() as f64;
    counts.iter().map(|&c| c as f64 / total).collect()
}

fn compute_psi(baseline: &[f64], production: &[f64], num_bins: usize) -> f64 {
    if baseline.is_empty() || production.is_empty() {
        return 0.0;
    }
    let min_val = baseline.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = baseline.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let p = histogram_proportions(baseline, min_val, max_val, num_bins);
    let q = histogram_proportions(production, min_val, max_val, num_bins);
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            let pi_s = pi.max(PSI_EPSILON);
            let qi_s = qi.max(PSI_EPSILON);
            (pi_s - qi_s) * (pi_s / qi_s).ln()
        })
        .sum()
}

// --- KS Statistic ---

fn compute_ks_statistic(baseline: &[f64], production: &[f64]) -> f64 {
    if baseline.is_empty() || production.is_empty() {
        return 0.0;
    }
    let mut sorted_b = baseline.to_vec();
    let mut sorted_p = production.to_vec();
    sorted_b.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted_p.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (n_b, n_p) = (sorted_b.len() as f64, sorted_p.len() as f64);
    let (mut max_diff, mut i_b, mut i_p): (f64, usize, usize) = (0.0, 0, 0);
    while i_b < sorted_b.len() && i_p < sorted_p.len() {
        if sorted_b[i_b] <= sorted_p[i_p] {
            max_diff = max_diff.max(((i_b + 1) as f64 / n_b - i_p as f64 / n_p).abs());
            i_b += 1;
        } else {
            max_diff = max_diff.max(((i_p + 1) as f64 / n_p - i_b as f64 / n_b).abs());
            i_p += 1;
        }
    }
    while i_b < sorted_b.len() {
        max_diff = max_diff.max(((i_b + 1) as f64 / n_b - 1.0).abs());
        i_b += 1;
    }
    while i_p < sorted_p.len() {
        max_diff = max_diff.max(((i_p + 1) as f64 / n_p - 1.0).abs());
        i_p += 1;
    }
    max_diff
}

// --- Confidence Monitor ---

struct ConfidenceMonitor {
    confidences: Vec<f64>,
    entropies: Vec<f64>,
    capacity: usize,
}

impl ConfidenceMonitor {
    fn new(capacity: usize) -> Self {
        Self {
            confidences: Vec::with_capacity(capacity),
            entropies: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn record(&mut self, probabilities: &[f64]) {
        if probabilities.is_empty() {
            return;
        }
        let max_conf = probabilities
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let entropy: f64 = probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        if self.confidences.len() >= self.capacity {
            self.confidences.remove(0);
            self.entropies.remove(0);
        }
        self.confidences.push(max_conf);
        self.entropies.push(entropy);
    }

    fn mean_confidence(&self) -> f64 {
        if self.confidences.is_empty() {
            return 0.0;
        }
        self.confidences.iter().sum::<f64>() / self.confidences.len() as f64
    }

    fn mean_entropy(&self) -> f64 {
        if self.entropies.is_empty() {
            return 0.0;
        }
        self.entropies.iter().sum::<f64>() / self.entropies.len() as f64
    }

    fn low_confidence_rate(&self, threshold: f64) -> f64 {
        if self.confidences.is_empty() {
            return 0.0;
        }
        self.confidences.iter().filter(|&&c| c < threshold).count() as f64
            / self.confidences.len() as f64
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.confidences.len()
    }
}

// --- Accuracy Tracker ---

struct AccuracyTracker {
    outcomes: Vec<bool>,
    capacity: usize,
    baseline_accuracy: f64,
}

impl AccuracyTracker {
    fn new(capacity: usize, baseline_accuracy: f64) -> Self {
        Self {
            outcomes: Vec::with_capacity(capacity),
            capacity,
            baseline_accuracy,
        }
    }

    fn record(&mut self, correct: bool) {
        if self.outcomes.len() >= self.capacity {
            self.outcomes.remove(0);
        }
        self.outcomes.push(correct);
    }

    fn accuracy(&self) -> f64 {
        if self.outcomes.is_empty() {
            return 0.0;
        }
        self.outcomes.iter().filter(|&&o| o).count() as f64 / self.outcomes.len() as f64
    }

    fn accuracy_drop(&self) -> f64 {
        (self.baseline_accuracy - self.accuracy()).max(0.0)
    }

    fn is_degraded(&self, tolerance: f64) -> bool {
        if self.outcomes.is_empty() {
            return false;
        }
        self.accuracy_drop() > tolerance
    }
}

// --- Data Generation ---

struct FeatureDistribution {
    means: [f64; NUM_FEATURES],
    stds: [f64; NUM_FEATURES],
}

fn baseline_distribution() -> FeatureDistribution {
    FeatureDistribution {
        means: [35.0, 55000.0, 700.0, 0.3],
        stds: [10.0, 15000.0, 50.0, 0.1],
    }
}

fn drifted_distribution() -> FeatureDistribution {
    FeatureDistribution {
        means: [42.0, 48000.0, 680.0, 0.4],
        stds: [12.0, 18000.0, 60.0, 0.15],
    }
}

fn extract_feature(samples: &[[f64; NUM_FEATURES]], feature_idx: usize) -> Vec<f64> {
    samples.iter().map(|s| s[feature_idx]).collect()
}

fn sample_features(
    dist: &FeatureDistribution,
    n: usize,
    rng: &mut DeterministicRng,
) -> Vec<[f64; NUM_FEATURES]> {
    (0..n)
        .map(|_| {
            let mut features = [0.0; NUM_FEATURES];
            for (i, feat) in features.iter_mut().enumerate() {
                *feat = rng.next_normal(dist.means[i], dist.stds[i]);
            }
            features
        })
        .collect()
}

fn simulate_prediction(features: &[f64; NUM_FEATURES], seed: u64, drifted: bool) -> Vec<f64> {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    for &f in features {
        f.to_bits().hash(&mut hasher);
    }
    let h = hasher.finish();
    let base = features[2] / 800.0;
    let noise = (h as f64 / u64::MAX as f64 - 0.5) * 0.2;
    let degradation = if drifted { 0.3 } else { 0.0 };
    let logits = [
        base + noise - degradation,
        (1.0 - base) * 0.6 + noise * 0.5,
        0.2 + noise * 0.3 + degradation * 0.5,
    ];
    let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum_exp: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum_exp).collect()
}

fn simulate_ground_truth(features: &[f64; NUM_FEATURES], seed: u64) -> usize {
    let mut hasher = DefaultHasher::new();
    (seed, "truth").hash(&mut hasher);
    for &f in features {
        f.to_bits().hash(&mut hasher);
    }
    let h = hasher.finish();
    if features[2] > 720.0 {
        0
    } else if features[2] > 650.0 {
        1
    } else if h % 3 == 0 {
        0
    } else {
        2
    }
}

#[derive(Debug, Clone)]
struct WindowStats {
    window_id: usize,
    mean_confidence: f64,
    mean_entropy: f64,
    accuracy: f64,
    alerts: Vec<DriftAlert>,
}

fn main() {
    println!("=== Model Drift Detection Example ===\n");
    let mut rng = DeterministicRng::new(42);
    let baseline_dist = baseline_distribution();
    let baseline_samples = sample_features(&baseline_dist, 200, &mut rng);

    println!("1. Baseline: {} samples", baseline_samples.len());
    for (i, name) in FEATURE_NAMES.iter().enumerate() {
        let v = extract_feature(&baseline_samples, i);
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        println!("   {name:>14}: mean={mean:>10.2}, std={std:>10.2}");
    }

    let drifted_dist = drifted_distribution();
    let production_samples = sample_features(&drifted_dist, 200, &mut rng);

    println!("\n2. PSI Computation (baseline=200, production=200, bins={PSI_NUM_BINS})");
    let mut psi_alerts: Vec<DriftAlert> = Vec::new();
    for (i, name) in FEATURE_NAMES.iter().enumerate() {
        let psi = compute_psi(
            &extract_feature(&baseline_samples, i),
            &extract_feature(&production_samples, i),
            PSI_NUM_BINS,
        );
        let severity = psi_severity(psi);
        println!("   {name:>14}: PSI={psi:.4}  [{severity}]");
        if severity >= Severity::Medium {
            psi_alerts.push(DriftAlert {
                source: format!("PSI/{name}"),
                severity,
                message: format!("Shift on '{name}'"),
                metric_value: psi,
            });
        }
    }

    println!("\n3. KS Statistic");
    let mut ks_alerts: Vec<DriftAlert> = Vec::new();
    for (i, name) in FEATURE_NAMES.iter().enumerate() {
        let ks = compute_ks_statistic(
            &extract_feature(&baseline_samples, i),
            &extract_feature(&production_samples, i),
        );
        let severity = ks_severity(ks);
        println!("   {name:>14}: KS={ks:.4}  [{severity}]");
        if severity >= Severity::Medium {
            ks_alerts.push(DriftAlert {
                source: format!("KS/{name}"),
                severity,
                message: format!("CDF deviation on '{name}'"),
                metric_value: ks,
            });
        }
    }

    println!("\n4. Confidence Monitoring");
    let mut confidence_monitor = ConfidenceMonitor::new(WINDOW_SIZE);
    for (idx, sample) in baseline_samples.iter().enumerate().take(WINDOW_SIZE) {
        confidence_monitor.record(&simulate_prediction(sample, idx as u64, false));
    }
    let mut drifted_monitor = ConfidenceMonitor::new(WINDOW_SIZE);
    for (idx, sample) in production_samples.iter().enumerate().take(WINDOW_SIZE) {
        drifted_monitor.record(&simulate_prediction(sample, (idx + 1000) as u64, true));
    }
    println!(
        "   Baseline: conf={:.4}, entropy={:.4}",
        confidence_monitor.mean_confidence(),
        confidence_monitor.mean_entropy()
    );
    println!(
        "   Drifted:  conf={:.4}, entropy={:.4}",
        drifted_monitor.mean_confidence(),
        drifted_monitor.mean_entropy()
    );
    println!(
        "   Low-conf rate (<0.5): {:.2}%",
        drifted_monitor.low_confidence_rate(0.5) * 100.0
    );

    println!("\n5. Time-Window Analysis");
    let mut all_window_stats: Vec<WindowStats> = Vec::with_capacity(5);
    for window_id in 0..5 {
        let drift_factor = if window_id < 2 {
            0.0
        } else {
            (window_id - 1) as f64 * 0.3
        };
        let window_dist = FeatureDistribution {
            means: [
                baseline_dist.means[0] + drift_factor * 3.0,
                baseline_dist.means[1] - drift_factor * 4000.0,
                baseline_dist.means[2] - drift_factor * 15.0,
                baseline_dist.means[3] + drift_factor * 0.05,
            ],
            stds: baseline_dist.stds,
        };
        let window_samples = sample_features(&window_dist, WINDOW_SIZE, &mut rng);
        let mut alerts: Vec<DriftAlert> = Vec::new();
        for (i, name) in FEATURE_NAMES.iter().enumerate() {
            let base_feat = extract_feature(&baseline_samples, i);
            let win_feat = extract_feature(&window_samples, i);
            let psi = compute_psi(&base_feat, &win_feat, PSI_NUM_BINS);
            let ks = compute_ks_statistic(&base_feat, &win_feat);
            if psi_severity(psi) >= Severity::Medium {
                alerts.push(DriftAlert {
                    source: format!("PSI/{name}"),
                    severity: psi_severity(psi),
                    message: format!("W{window_id}: PSI on '{name}'"),
                    metric_value: psi,
                });
            }
            if ks_severity(ks) >= Severity::Medium {
                alerts.push(DriftAlert {
                    source: format!("KS/{name}"),
                    severity: ks_severity(ks),
                    message: format!("W{window_id}: KS on '{name}'"),
                    metric_value: ks,
                });
            }
        }
        let mut win_monitor = ConfidenceMonitor::new(WINDOW_SIZE);
        let mut win_accuracy = AccuracyTracker::new(WINDOW_SIZE, 0.75);
        let is_drifted = window_id >= 2;
        for (idx, sample) in window_samples.iter().enumerate() {
            let probs = simulate_prediction(sample, (window_id * 1000 + idx) as u64, is_drifted);
            win_monitor.record(&probs);
            let predicted = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i);
            win_accuracy.record(
                predicted == simulate_ground_truth(sample, (window_id * 1000 + idx) as u64),
            );
        }
        if win_accuracy.is_degraded(0.10) {
            alerts.push(DriftAlert {
                source: "Accuracy".to_string(),
                severity: Severity::High,
                message: format!(
                    "W{window_id}: accuracy dropped {:.1}%",
                    win_accuracy.accuracy_drop() * 100.0
                ),
                metric_value: win_accuracy.accuracy(),
            });
        }
        let stats = WindowStats {
            window_id,
            mean_confidence: win_monitor.mean_confidence(),
            mean_entropy: win_monitor.mean_entropy(),
            accuracy: win_accuracy.accuracy(),
            alerts,
        };
        println!(
            "   W{}: conf={:.4}, entropy={:.4}, acc={:.2}%, alerts={}",
            stats.window_id,
            stats.mean_confidence,
            stats.mean_entropy,
            stats.accuracy * 100.0,
            stats.alerts.len()
        );
        all_window_stats.push(stats);
    }

    println!("\n6. Alert Summary");
    let total_alerts: usize = all_window_stats.iter().map(|w| w.alerts.len()).sum();
    let mut all_alerts: Vec<&DriftAlert> =
        all_window_stats.iter().flat_map(|w| &w.alerts).collect();
    all_alerts.sort_by(|a, b| b.severity.cmp(&a.severity));
    println!("   Total: {total_alerts}");
    for (i, alert) in all_alerts.iter().enumerate().take(6) {
        println!("   {}. {alert}", i + 1);
    }
    println!("\n=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psi_identical_distributions() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let psi = compute_psi(&data, &data, PSI_NUM_BINS);
        assert!(
            psi.abs() < 1e-10,
            "PSI of identical should be ~0, got {psi}"
        );
    }

    #[test]
    fn test_psi_shifted_distribution() {
        let baseline: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let shifted: Vec<f64> = (0..200).map(|i| (i as f64) + 100.0).collect();
        assert!(compute_psi(&baseline, &shifted, PSI_NUM_BINS) > 0.1);
    }

    #[test]
    fn test_psi_empty_inputs() {
        let empty: Vec<f64> = Vec::new();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert!(compute_psi(&empty, &data, 5).abs() < f64::EPSILON);
        assert!(compute_psi(&data, &empty, 5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ks_identical_distributions() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert!(compute_ks_statistic(&data, &data) < 0.02);
    }

    #[test]
    fn test_ks_completely_separated() {
        let a: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let b: Vec<f64> = (100..150).map(|i| i as f64).collect();
        assert!((compute_ks_statistic(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ks_empty_inputs() {
        let empty: Vec<f64> = Vec::new();
        let data: Vec<f64> = vec![1.0, 2.0];
        assert!(compute_ks_statistic(&empty, &data).abs() < f64::EPSILON);
    }

    #[test]
    fn test_confidence_monitor_mean_and_capacity() {
        let mut monitor = ConfidenceMonitor::new(3);
        monitor.record(&[0.8, 0.1, 0.1]);
        monitor.record(&[0.6, 0.2, 0.2]);
        assert!((monitor.mean_confidence() - 0.7).abs() < 1e-10);
        for i in 0..10 {
            monitor.record(&[(i as f64) * 0.1, 0.5, 0.4]);
        }
        assert_eq!(monitor.len(), 3, "Monitor should not exceed capacity");
    }

    #[test]
    fn test_accuracy_tracker_degradation() {
        let mut tracker = AccuracyTracker::new(10, 0.9);
        for _ in 0..5 {
            tracker.record(true);
            tracker.record(false);
        }
        assert!((tracker.accuracy() - 0.5).abs() < 1e-10);
        assert!(tracker.is_degraded(0.1));
        assert!(!tracker.is_degraded(0.5));
    }

    #[test]
    fn test_severity_thresholds_and_ordering() {
        assert_eq!(psi_severity(0.05), Severity::Low);
        assert_eq!(psi_severity(0.15), Severity::Medium);
        assert_eq!(psi_severity(0.25), Severity::High);
        assert_eq!(psi_severity(0.35), Severity::Critical);
        assert_eq!(ks_severity(0.10), Severity::Low);
        assert_eq!(ks_severity(0.20), Severity::Medium);
        assert_eq!(ks_severity(0.30), Severity::High);
        assert_eq!(ks_severity(0.50), Severity::Critical);
        assert!(Severity::Low < Severity::Medium);
        assert!(Severity::High < Severity::Critical);
    }

    #[test]
    fn test_simulate_prediction_produces_valid_distribution() {
        let features = [35.0, 55000.0, 700.0, 0.3];
        let probs = simulate_prediction(&features, 42, false);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }
}
