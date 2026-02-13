//! Model Drift Detection for Production ML Monitoring
//!
//! Demonstrates data drift and concept drift detection techniques for
//! production ML systems using statistical methods that require zero
//! external dependencies beyond `std`.
//!
//! # Drift Detection Techniques
//!
//! - **Population Stability Index (PSI)**: Measures distribution shift between
//!   baseline and production feature distributions using binned histograms
//! - **Kolmogorov-Smirnov Statistic**: Per-feature maximum deviation between
//!   empirical CDFs to detect individual feature drift
//! - **Prediction Confidence Monitoring**: Tracks softmax entropy and mean
//!   confidence over sliding time windows
//! - **Accuracy Degradation Detection**: Compares rolling accuracy against a
//!   baseline threshold to trigger retraining alerts
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                 Model Drift Detection Pipeline                  │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  Baseline Distribution ──► PSI Computation ──► Alert Engine     │
//! │         │                                          ▲            │
//! │         ▼                                          │            │
//! │  Production Stream ──► KS Statistic ──────────────►│            │
//! │         │                                          │            │
//! │         ▼                                          │            │
//! │  Predictions ──► Confidence Monitor ──────────────►│            │
//! │         │                                          │            │
//! │         ▼                                          │            │
//! │  Ground Truth ──► Accuracy Tracker ───────────────►│            │
//! │                                                    │            │
//! │                                              ┌─────┴─────┐     │
//! │                                              │  Severity  │     │
//! │                                              │  LOW/MED/  │     │
//! │                                              │  HIGH/CRIT │     │
//! │                                              └───────────┘     │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example model_drift_detection
//! ```
//!
//! # Recipe Metadata
//!
//! - **Category**: Monitoring
//! - **Complexity**: Intermediate
//! - **Dependencies**: None (std only)
//! - **IIUR**: Isolated, Idempotent, Useful, Reproducible

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

// ============================================================================
// Constants
// ============================================================================

/// Number of bins for PSI histogram computation
const PSI_NUM_BINS: usize = 10;

/// Number of features in the simulated dataset
const NUM_FEATURES: usize = 4;

/// Size of the rolling window for time-window analysis
const WINDOW_SIZE: usize = 50;

/// Smoothing constant to prevent log(0) in PSI
const PSI_EPSILON: f64 = 1e-6;

/// Feature names for display
const FEATURE_NAMES: [&str; NUM_FEATURES] = ["age", "income", "credit_score", "debt_ratio"];

// ============================================================================
// Alert Severity
// ============================================================================

/// Severity level for drift alerts
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Severity {
    /// Informational, no action required
    Low,
    /// Worth investigating, monitor closely
    Medium,
    /// Likely requires intervention
    High,
    /// Immediate action required, model may be unreliable
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

/// Classifies a PSI value into a severity level.
///
/// Standard thresholds from industry practice:
/// - PSI < 0.1  => Low (no significant shift)
/// - PSI < 0.2  => Medium (moderate shift)
/// - PSI < 0.3  => High (significant shift)
/// - PSI >= 0.3 => Critical (major distribution change)
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

/// Classifies a KS statistic into a severity level.
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

// ============================================================================
// Drift Alert
// ============================================================================

/// A single drift alert produced by the monitoring pipeline
#[derive(Debug, Clone)]
struct DriftAlert {
    /// Source detector that raised the alert
    source: String,
    /// Severity level
    severity: Severity,
    /// Human-readable message
    message: String,
    /// Numeric metric value that triggered the alert
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

// ============================================================================
// Deterministic RNG
// ============================================================================

/// Deterministic pseudo-random number generator using `DefaultHasher`.
///
/// Produces repeatable sequences given the same seed, suitable for
/// simulation without pulling in external crate dependencies.
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance state and return a pseudo-random `u64`.
    fn next_u64(&mut self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.state.hash(&mut hasher);
        self.state = hasher.finish();
        self.state
    }

    /// Return a float in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Approximate normal(mean, std) via the Irwin-Hall approximation (sum of 12 uniforms).
    fn next_normal(&mut self, mean: f64, std: f64) -> f64 {
        let sum: f64 = (0..12).map(|_| self.next_f64()).sum();
        // sum of 12 U(0,1) has mean=6, var=1
        mean + (sum - 6.0) * std
    }
}

// ============================================================================
// PSI (Population Stability Index)
// ============================================================================

/// Bins a slice of values into a histogram with `num_bins` equal-width bins
/// spanning `[min_val, max_val]`.  Returns proportions (each bin count / total).
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
        let clamped = idx.min(num_bins - 1);
        counts[clamped] += 1;
    }

    let total = values.len() as f64;
    counts.iter().map(|&c| c as f64 / total).collect()
}

/// Compute the Population Stability Index between a baseline distribution
/// and a production distribution.
///
/// PSI = sum_i (P_i - Q_i) * ln(P_i / Q_i)
///
/// where P is baseline and Q is production proportions across bins.
fn compute_psi(baseline: &[f64], production: &[f64], num_bins: usize) -> f64 {
    if baseline.is_empty() || production.is_empty() {
        return 0.0;
    }

    // Determine shared range from baseline
    let min_val = baseline.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = baseline.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let p = histogram_proportions(baseline, min_val, max_val, num_bins);
    let q = histogram_proportions(production, min_val, max_val, num_bins);

    let mut psi = 0.0;
    for (i, (&pi, &qi)) in p.iter().zip(q.iter()).enumerate() {
        let _ = i; // used only for iteration
        let pi_s = pi.max(PSI_EPSILON);
        let qi_s = qi.max(PSI_EPSILON);
        psi += (pi_s - qi_s) * (pi_s / qi_s).ln();
    }
    psi
}

// ============================================================================
// KS Statistic
// ============================================================================

/// Compute a Kolmogorov-Smirnov-like statistic between two samples.
///
/// Returns the maximum absolute difference between the two empirical CDFs.
fn compute_ks_statistic(baseline: &[f64], production: &[f64]) -> f64 {
    if baseline.is_empty() || production.is_empty() {
        return 0.0;
    }

    let mut sorted_b = baseline.to_vec();
    let mut sorted_p = production.to_vec();
    sorted_b.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted_p.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n_b = sorted_b.len() as f64;
    let n_p = sorted_p.len() as f64;

    let mut max_diff: f64 = 0.0;
    let mut i_b: usize = 0;
    let mut i_p: usize = 0;

    while i_b < sorted_b.len() && i_p < sorted_p.len() {
        let cdf_b = (i_b + 1) as f64 / n_b;
        let cdf_p = (i_p + 1) as f64 / n_p;

        if sorted_b[i_b] <= sorted_p[i_p] {
            let diff = (cdf_b - (i_p as f64 / n_p)).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            i_b += 1;
        } else {
            let diff = (cdf_p - (i_b as f64 / n_b)).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            i_p += 1;
        }
    }

    // Handle remaining tail
    while i_b < sorted_b.len() {
        let cdf_b = (i_b + 1) as f64 / n_b;
        let diff = (cdf_b - 1.0).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        i_b += 1;
    }
    while i_p < sorted_p.len() {
        let cdf_p = (i_p + 1) as f64 / n_p;
        let diff = (cdf_p - 1.0).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        i_p += 1;
    }

    max_diff
}

// ============================================================================
// Prediction Confidence Monitor
// ============================================================================

/// Tracks prediction confidence over a rolling window.
struct ConfidenceMonitor {
    /// Rolling buffer of max-confidence values
    confidences: Vec<f64>,
    /// Rolling buffer of prediction entropies
    entropies: Vec<f64>,
    /// Maximum capacity
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

    /// Record a prediction's softmax probability vector.
    fn record(&mut self, probabilities: &[f64]) {
        if probabilities.is_empty() {
            return;
        }

        // Max confidence
        let max_conf = probabilities
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // Shannon entropy: -sum(p * ln(p))
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

    /// Mean confidence in the current window.
    fn mean_confidence(&self) -> f64 {
        if self.confidences.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.confidences.iter().sum();
        sum / self.confidences.len() as f64
    }

    /// Mean entropy in the current window.
    fn mean_entropy(&self) -> f64 {
        if self.entropies.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.entropies.iter().sum();
        sum / self.entropies.len() as f64
    }

    /// Fraction of predictions with confidence below `threshold`.
    fn low_confidence_rate(&self, threshold: f64) -> f64 {
        if self.confidences.is_empty() {
            return 0.0;
        }
        let count = self.confidences.iter().filter(|&&c| c < threshold).count();
        count as f64 / self.confidences.len() as f64
    }

    /// Number of recorded predictions (used in tests).
    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.confidences.len()
    }
}

// ============================================================================
// Accuracy Tracker
// ============================================================================

/// Tracks prediction accuracy over a rolling window and detects degradation.
struct AccuracyTracker {
    /// Rolling buffer: true if prediction was correct
    outcomes: Vec<bool>,
    /// Maximum window size
    capacity: usize,
    /// Baseline accuracy to compare against
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

    /// Record whether a prediction was correct.
    fn record(&mut self, correct: bool) {
        if self.outcomes.len() >= self.capacity {
            self.outcomes.remove(0);
        }
        self.outcomes.push(correct);
    }

    /// Current rolling accuracy.
    fn accuracy(&self) -> f64 {
        if self.outcomes.is_empty() {
            return 0.0;
        }
        let correct = self.outcomes.iter().filter(|&&o| o).count();
        correct as f64 / self.outcomes.len() as f64
    }

    /// Absolute drop from baseline accuracy.
    fn accuracy_drop(&self) -> f64 {
        (self.baseline_accuracy - self.accuracy()).max(0.0)
    }

    /// Whether accuracy has degraded beyond a given tolerance.
    /// Returns false if no outcomes have been recorded yet.
    fn is_degraded(&self, tolerance: f64) -> bool {
        if self.outcomes.is_empty() {
            return false;
        }
        self.accuracy_drop() > tolerance
    }

    /// Number of recorded outcomes (used in tests).
    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.outcomes.len()
    }
}

// ============================================================================
// Data Generation
// ============================================================================

/// Feature distribution parameters
struct FeatureDistribution {
    means: [f64; NUM_FEATURES],
    stds: [f64; NUM_FEATURES],
}

/// Generate a baseline feature distribution.
fn baseline_distribution() -> FeatureDistribution {
    FeatureDistribution {
        means: [35.0, 55000.0, 700.0, 0.3],
        stds: [10.0, 15000.0, 50.0, 0.1],
    }
}

/// Generate a drifted feature distribution (simulates production shift).
fn drifted_distribution() -> FeatureDistribution {
    FeatureDistribution {
        means: [42.0, 48000.0, 680.0, 0.4],
        stds: [12.0, 18000.0, 60.0, 0.15],
    }
}

/// Sample `n` observations from a `FeatureDistribution`.
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

/// Simulate a softmax prediction given feature values and a deterministic seed.
fn simulate_prediction(features: &[f64; NUM_FEATURES], seed: u64, drifted: bool) -> Vec<f64> {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    for &f in features {
        f.to_bits().hash(&mut hasher);
    }
    let h = hasher.finish();

    // Produce 3-class logits biased by feature[2] (credit_score proxy)
    let base = features[2] / 800.0;
    let noise = (h as f64 / u64::MAX as f64 - 0.5) * 0.2;

    let degradation = if drifted { 0.3 } else { 0.0 };

    let logits = [
        base + noise - degradation,
        (1.0 - base) * 0.6 + noise * 0.5,
        0.2 + noise * 0.3 + degradation * 0.5,
    ];

    // Softmax
    let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum_exp: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum_exp).collect()
}

/// Simulate ground truth label.
fn simulate_ground_truth(features: &[f64; NUM_FEATURES], seed: u64) -> usize {
    let mut hasher = DefaultHasher::new();
    (seed, "truth").hash(&mut hasher);
    for &f in features {
        f.to_bits().hash(&mut hasher);
    }
    let h = hasher.finish();

    // Class is primarily driven by credit_score
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

// ============================================================================
// Window Statistics
// ============================================================================

/// Summary statistics for a single monitoring window.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct WindowStats {
    window_id: usize,
    psi_values: [f64; NUM_FEATURES],
    ks_values: [f64; NUM_FEATURES],
    mean_confidence: f64,
    mean_entropy: f64,
    accuracy: f64,
    alerts: Vec<DriftAlert>,
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== Model Drift Detection Example ===\n");

    let mut rng = DeterministicRng::new(42);

    // =========================================================================
    // Section 1: Baseline Distribution Setup
    // =========================================================================
    println!("1. Baseline Distribution Setup");
    println!("   ─────────────────────────────────────────");

    let baseline_dist = baseline_distribution();
    let baseline_samples = sample_features(&baseline_dist, 200, &mut rng);

    println!("   Samples:  {}", baseline_samples.len());
    for (i, name) in FEATURE_NAMES.iter().enumerate() {
        let values: Vec<f64> = baseline_samples.iter().map(|s| s[i]).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        println!(
            "   {:>14}: mean={:>10.2}, std={:>10.2}",
            name,
            mean,
            variance.sqrt()
        );
    }
    println!();

    // =========================================================================
    // Section 2: PSI Computation
    // =========================================================================
    println!("2. PSI Computation");
    println!("   ─────────────────────────────────────────");

    let drifted_dist = drifted_distribution();
    let production_samples = sample_features(&drifted_dist, 200, &mut rng);

    println!("   Comparing baseline (n=200) vs production (n=200)");
    println!("   Bins: {PSI_NUM_BINS}");
    println!();

    let mut psi_alerts: Vec<DriftAlert> = Vec::new();
    for (i, name) in FEATURE_NAMES.iter().enumerate() {
        let baseline_feat: Vec<f64> = baseline_samples.iter().map(|s| s[i]).collect();
        let production_feat: Vec<f64> = production_samples.iter().map(|s| s[i]).collect();

        let psi = compute_psi(&baseline_feat, &production_feat, PSI_NUM_BINS);
        let severity = psi_severity(psi);

        println!("   {:>14}: PSI={:.4}  [{}]", name, psi, severity);

        if severity >= Severity::Medium {
            psi_alerts.push(DriftAlert {
                source: format!("PSI/{name}"),
                severity,
                message: format!("Distribution shift detected for feature '{name}'"),
                metric_value: psi,
            });
        }
    }
    println!("   Alerts raised: {}", psi_alerts.len());
    println!();

    // =========================================================================
    // Section 3: Feature Drift Detection (per-feature KS-like statistic)
    // =========================================================================
    println!("3. Feature Drift Detection (KS Statistic)");
    println!("   ─────────────────────────────────────────");

    let mut ks_alerts: Vec<DriftAlert> = Vec::new();
    for (i, name) in FEATURE_NAMES.iter().enumerate() {
        let baseline_feat: Vec<f64> = baseline_samples.iter().map(|s| s[i]).collect();
        let production_feat: Vec<f64> = production_samples.iter().map(|s| s[i]).collect();

        let ks = compute_ks_statistic(&baseline_feat, &production_feat);
        let severity = ks_severity(ks);

        println!("   {:>14}: KS={:.4}  [{}]", name, ks, severity);

        if severity >= Severity::Medium {
            ks_alerts.push(DriftAlert {
                source: format!("KS/{name}"),
                severity,
                message: format!("Feature CDF deviation detected for '{name}'"),
                metric_value: ks,
            });
        }
    }
    println!("   Alerts raised: {}", ks_alerts.len());
    println!();

    // =========================================================================
    // Section 4: Prediction Confidence Monitoring
    // =========================================================================
    println!("4. Prediction Confidence Monitoring");
    println!("   ─────────────────────────────────────────");

    let mut confidence_monitor = ConfidenceMonitor::new(WINDOW_SIZE);

    // Record baseline predictions
    for (idx, sample) in baseline_samples.iter().enumerate().take(WINDOW_SIZE) {
        let probs = simulate_prediction(sample, idx as u64, false);
        confidence_monitor.record(&probs);
    }

    let baseline_confidence = confidence_monitor.mean_confidence();
    let baseline_entropy = confidence_monitor.mean_entropy();
    println!("   Baseline mean confidence: {:.4}", baseline_confidence);
    println!("   Baseline mean entropy:    {:.4}", baseline_entropy);

    // Record drifted predictions
    let mut drifted_monitor = ConfidenceMonitor::new(WINDOW_SIZE);
    for (idx, sample) in production_samples.iter().enumerate().take(WINDOW_SIZE) {
        let probs = simulate_prediction(sample, (idx + 1000) as u64, true);
        drifted_monitor.record(&probs);
    }

    let drifted_confidence = drifted_monitor.mean_confidence();
    let drifted_entropy = drifted_monitor.mean_entropy();
    println!("   Drifted  mean confidence: {:.4}", drifted_confidence);
    println!("   Drifted  mean entropy:    {:.4}", drifted_entropy);
    println!(
        "   Confidence delta:         {:.4}",
        baseline_confidence - drifted_confidence
    );

    let low_conf_rate = drifted_monitor.low_confidence_rate(0.5);
    println!(
        "   Low-confidence rate (<0.5): {:.2}%",
        low_conf_rate * 100.0
    );
    println!();

    // =========================================================================
    // Section 5: Time-Window Analysis (simulating production windows)
    // =========================================================================
    println!("5. Time-Window Analysis");
    println!("   ─────────────────────────────────────────");

    let num_windows = 5;
    let samples_per_window = WINDOW_SIZE;
    let mut all_window_stats: Vec<WindowStats> = Vec::with_capacity(num_windows);

    // Window 0-1: no drift; windows 2-4: increasing drift
    for window_id in 0..num_windows {
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

        let window_samples = sample_features(&window_dist, samples_per_window, &mut rng);

        // PSI and KS per feature
        let mut psi_values = [0.0; NUM_FEATURES];
        let mut ks_values = [0.0; NUM_FEATURES];
        for i in 0..NUM_FEATURES {
            let base_feat: Vec<f64> = baseline_samples.iter().map(|s| s[i]).collect();
            let win_feat: Vec<f64> = window_samples.iter().map(|s| s[i]).collect();
            psi_values[i] = compute_psi(&base_feat, &win_feat, PSI_NUM_BINS);
            ks_values[i] = compute_ks_statistic(&base_feat, &win_feat);
        }

        // Confidence
        let mut win_monitor = ConfidenceMonitor::new(samples_per_window);
        let mut win_accuracy = AccuracyTracker::new(samples_per_window, 0.75);
        let is_drifted = window_id >= 2;

        for (idx, sample) in window_samples.iter().enumerate() {
            let probs = simulate_prediction(sample, (window_id * 1000 + idx) as u64, is_drifted);
            win_monitor.record(&probs);

            let predicted = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx);
            let truth = simulate_ground_truth(sample, (window_id * 1000 + idx) as u64);
            win_accuracy.record(predicted == truth);
        }

        // Collect alerts for this window
        let mut alerts: Vec<DriftAlert> = Vec::new();
        for (i, name) in FEATURE_NAMES.iter().enumerate() {
            if psi_severity(psi_values[i]) >= Severity::Medium {
                alerts.push(DriftAlert {
                    source: format!("PSI/{name}"),
                    severity: psi_severity(psi_values[i]),
                    message: format!("Window {window_id}: PSI drift on '{name}'"),
                    metric_value: psi_values[i],
                });
            }
            if ks_severity(ks_values[i]) >= Severity::Medium {
                alerts.push(DriftAlert {
                    source: format!("KS/{name}"),
                    severity: ks_severity(ks_values[i]),
                    message: format!("Window {window_id}: KS drift on '{name}'"),
                    metric_value: ks_values[i],
                });
            }
        }
        if win_accuracy.is_degraded(0.10) {
            alerts.push(DriftAlert {
                source: "Accuracy".to_string(),
                severity: Severity::High,
                message: format!(
                    "Window {window_id}: accuracy dropped {:.1}% from baseline",
                    win_accuracy.accuracy_drop() * 100.0
                ),
                metric_value: win_accuracy.accuracy(),
            });
        }

        let stats = WindowStats {
            window_id,
            psi_values,
            ks_values,
            mean_confidence: win_monitor.mean_confidence(),
            mean_entropy: win_monitor.mean_entropy(),
            accuracy: win_accuracy.accuracy(),
            alerts,
        };

        println!(
            "   Window {}: confidence={:.4}, entropy={:.4}, accuracy={:.2}%, alerts={}",
            stats.window_id,
            stats.mean_confidence,
            stats.mean_entropy,
            stats.accuracy * 100.0,
            stats.alerts.len(),
        );

        all_window_stats.push(stats);
    }
    println!();

    // =========================================================================
    // Section 6: Alert Summary & Recommendations
    // =========================================================================
    println!("6. Alert Summary & Recommendations");
    println!("   ─────────────────────────────────────────");

    let total_alerts: usize = all_window_stats.iter().map(|w| w.alerts.len()).sum();
    let critical_count = all_window_stats
        .iter()
        .flat_map(|w| &w.alerts)
        .filter(|a| a.severity == Severity::Critical)
        .count();
    let high_count = all_window_stats
        .iter()
        .flat_map(|w| &w.alerts)
        .filter(|a| a.severity == Severity::High)
        .count();

    println!("   Total alerts:    {total_alerts}");
    println!("   Critical:        {critical_count}");
    println!("   High:            {high_count}");
    println!();

    // Print the top alerts by severity
    let mut all_alerts: Vec<&DriftAlert> =
        all_window_stats.iter().flat_map(|w| &w.alerts).collect();
    all_alerts.sort_by(|a, b| b.severity.cmp(&a.severity));

    let display_count = all_alerts.len().min(8);
    println!("   Top {} alerts:", display_count);
    for (i, alert) in all_alerts.iter().enumerate().take(display_count) {
        println!("     {}. {}", i + 1, alert);
    }
    println!();

    // Recommendations
    println!("   Recommendations:");
    if critical_count > 0 {
        println!("     - URGENT: Retrain model immediately; critical distribution shift detected");
    }
    if high_count > 0 {
        println!("     - Schedule model retraining within current sprint");
    }
    if total_alerts > 0 {
        println!("     - Review feature pipelines for upstream data changes");
        println!("     - Consider expanding monitoring to additional features");
    } else {
        println!("     - No drift detected; model is performing within expected bounds");
    }

    println!("\n=== Example Complete ===");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psi_identical_distributions() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let psi = compute_psi(&data, &data, PSI_NUM_BINS);
        assert!(
            psi.abs() < 1e-10,
            "PSI of identical distributions should be ~0, got {psi}"
        );
    }

    #[test]
    fn test_psi_shifted_distribution() {
        let baseline: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let shifted: Vec<f64> = (0..200).map(|i| (i as f64) + 100.0).collect();
        let psi = compute_psi(&baseline, &shifted, PSI_NUM_BINS);
        assert!(
            psi > 0.1,
            "PSI of shifted distribution should be > 0.1, got {psi}"
        );
    }

    #[test]
    fn test_psi_empty_inputs() {
        let empty: Vec<f64> = Vec::new();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert!((compute_psi(&empty, &data, 5)).abs() < f64::EPSILON);
        assert!((compute_psi(&data, &empty, 5)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ks_identical_distributions() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let ks = compute_ks_statistic(&data, &data);
        assert!(
            ks < 0.02,
            "KS of identical distributions should be ~0, got {ks}"
        );
    }

    #[test]
    fn test_ks_completely_separated() {
        let a: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let b: Vec<f64> = (100..150).map(|i| i as f64).collect();
        let ks = compute_ks_statistic(&a, &b);
        assert!(
            (ks - 1.0).abs() < 1e-10,
            "KS of fully separated distributions should be ~1.0, got {ks}"
        );
    }

    #[test]
    fn test_ks_empty_inputs() {
        let empty: Vec<f64> = Vec::new();
        let data: Vec<f64> = vec![1.0, 2.0];
        assert!((compute_ks_statistic(&empty, &data)).abs() < f64::EPSILON);
        assert!((compute_ks_statistic(&data, &empty)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_confidence_monitor_mean() {
        let mut monitor = ConfidenceMonitor::new(10);
        monitor.record(&[0.8, 0.1, 0.1]);
        monitor.record(&[0.6, 0.2, 0.2]);

        let mean = monitor.mean_confidence();
        assert!(
            (mean - 0.7).abs() < 1e-10,
            "Mean confidence should be 0.7, got {mean}"
        );
        assert_eq!(monitor.len(), 2);
    }

    #[test]
    fn test_confidence_monitor_entropy() {
        let mut monitor = ConfidenceMonitor::new(10);
        // Uniform distribution has max entropy for 3 classes: ln(3) ~ 1.0986
        let uniform = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        monitor.record(&uniform);

        let entropy = monitor.mean_entropy();
        assert!(
            (entropy - 3.0_f64.ln()).abs() < 1e-6,
            "Entropy of uniform should be ln(3), got {entropy}"
        );
    }

    #[test]
    fn test_confidence_monitor_low_rate() {
        let mut monitor = ConfidenceMonitor::new(10);
        monitor.record(&[0.3, 0.4, 0.3]); // max=0.4, below 0.5
        monitor.record(&[0.7, 0.2, 0.1]); // max=0.7, above 0.5

        let rate = monitor.low_confidence_rate(0.5);
        assert!(
            (rate - 0.5).abs() < 1e-10,
            "Low confidence rate should be 0.5, got {rate}"
        );
    }

    #[test]
    fn test_accuracy_tracker_basic() {
        let mut tracker = AccuracyTracker::new(10, 0.8);
        tracker.record(true);
        tracker.record(true);
        tracker.record(false);

        assert!(
            (tracker.accuracy() - 2.0 / 3.0).abs() < 1e-10,
            "Accuracy should be 2/3"
        );
        assert_eq!(tracker.len(), 3);
    }

    #[test]
    fn test_accuracy_tracker_degradation() {
        let mut tracker = AccuracyTracker::new(10, 0.9);
        // 50% accuracy is a 40% drop from 90% baseline
        for _ in 0..5 {
            tracker.record(true);
            tracker.record(false);
        }

        assert!(
            tracker.is_degraded(0.1),
            "Should detect degradation at 0.1 tolerance"
        );
        assert!(
            !tracker.is_degraded(0.5),
            "Should not trigger at 0.5 tolerance"
        );
    }

    #[test]
    fn test_accuracy_tracker_rolling_window() {
        let mut tracker = AccuracyTracker::new(4, 0.8);
        // Fill with correct
        for _ in 0..4 {
            tracker.record(true);
        }
        assert!((tracker.accuracy() - 1.0).abs() < 1e-10);

        // Push out old values with incorrect
        for _ in 0..4 {
            tracker.record(false);
        }
        assert!(
            tracker.accuracy().abs() < 1e-10,
            "All wrong after window rolls"
        );
    }

    #[test]
    fn test_psi_severity_thresholds() {
        assert_eq!(psi_severity(0.05), Severity::Low);
        assert_eq!(psi_severity(0.15), Severity::Medium);
        assert_eq!(psi_severity(0.25), Severity::High);
        assert_eq!(psi_severity(0.35), Severity::Critical);
    }

    #[test]
    fn test_ks_severity_thresholds() {
        assert_eq!(ks_severity(0.10), Severity::Low);
        assert_eq!(ks_severity(0.20), Severity::Medium);
        assert_eq!(ks_severity(0.30), Severity::High);
        assert_eq!(ks_severity(0.50), Severity::Critical);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Low < Severity::Medium);
        assert!(Severity::Medium < Severity::High);
        assert!(Severity::High < Severity::Critical);
    }

    #[test]
    fn test_deterministic_rng_reproducibility() {
        let mut rng1 = DeterministicRng::new(123);
        let mut rng2 = DeterministicRng::new(123);

        let seq1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
        let seq2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();
        assert_eq!(seq1, seq2, "Same seed must produce identical sequences");
    }

    #[test]
    fn test_histogram_proportions_sum_to_one() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let props = histogram_proportions(&values, 0.0, 99.0, PSI_NUM_BINS);
        let sum: f64 = props.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Proportions should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_simulate_prediction_produces_valid_distribution() {
        let features = [35.0, 55000.0, 700.0, 0.3];
        let probs = simulate_prediction(&features, 42, false);
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Probabilities should sum to 1.0, got {sum}"
        );
        assert!(
            probs.iter().all(|&p| p >= 0.0 && p <= 1.0),
            "All probabilities should be in [0,1]"
        );
    }

    #[test]
    fn test_confidence_monitor_respects_capacity() {
        let mut monitor = ConfidenceMonitor::new(3);
        for i in 0..10 {
            monitor.record(&[(i as f64) * 0.1, 0.5, 0.4]);
        }
        assert_eq!(monitor.len(), 3, "Monitor should not exceed capacity");
    }

    #[test]
    fn test_drift_alert_display() {
        let alert = DriftAlert {
            source: "PSI/age".to_string(),
            severity: Severity::High,
            message: "Distribution shift detected".to_string(),
            metric_value: 0.25,
        };
        let display = format!("{alert}");
        assert!(display.contains("HIGH"));
        assert!(display.contains("PSI/age"));
        assert!(display.contains("0.25"));
    }

    #[test]
    fn test_sample_features_count() {
        let dist = baseline_distribution();
        let mut rng = DeterministicRng::new(99);
        let samples = sample_features(&dist, 50, &mut rng);
        assert_eq!(samples.len(), 50);
    }

    #[test]
    fn test_window_stats_structure() {
        let stats = WindowStats {
            window_id: 0,
            psi_values: [0.01, 0.02, 0.03, 0.04],
            ks_values: [0.05, 0.06, 0.07, 0.08],
            mean_confidence: 0.75,
            mean_entropy: 0.5,
            accuracy: 0.80,
            alerts: Vec::new(),
        };
        assert_eq!(stats.window_id, 0);
        assert_eq!(stats.alerts.len(), 0);
        assert!((stats.accuracy - 0.80).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_monitor_empty() {
        let monitor = ConfidenceMonitor::new(10);
        assert!((monitor.mean_confidence()).abs() < f64::EPSILON);
        assert!((monitor.mean_entropy()).abs() < f64::EPSILON);
        assert!((monitor.low_confidence_rate(0.5)).abs() < f64::EPSILON);
        assert_eq!(monitor.len(), 0);
    }

    #[test]
    fn test_accuracy_tracker_empty() {
        let tracker = AccuracyTracker::new(10, 0.9);
        assert!((tracker.accuracy()).abs() < f64::EPSILON);
        assert_eq!(tracker.len(), 0);
        assert!(!tracker.is_degraded(0.1));
    }
}
