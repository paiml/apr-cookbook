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
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

pub const PSI_NUM_BINS: usize = 10;
pub const NUM_FEATURES: usize = 4;
pub const WINDOW_SIZE: usize = 50;
pub const PSI_EPSILON: f64 = 1e-6;
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = ["age", "income", "credit_score", "debt_ratio"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
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

pub fn psi_severity(psi: f64) -> Severity {
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

pub fn ks_severity(ks: f64) -> Severity {
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
pub struct DriftAlert {
    pub source: String,
    pub severity: Severity,
    pub message: String,
    pub metric_value: f64,
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

pub struct DeterministicRng {
    pub state: u64,
}

impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.state.hash(&mut hasher);
        self.state = hasher.finish();
        self.state
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    pub fn next_normal(&mut self, mean: f64, std: f64) -> f64 {
        let sum: f64 = (0..12).map(|_| self.next_f64()).sum();
        mean + (sum - 6.0) * std
    }
}

// --- PSI ---

pub fn histogram_proportions(
    values: &[f64],
    min_val: f64,
    max_val: f64,
    num_bins: usize,
) -> Vec<f64> {
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

pub fn compute_psi(baseline: &[f64], production: &[f64], num_bins: usize) -> f64 {
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

pub fn compute_ks_statistic(baseline: &[f64], production: &[f64]) -> f64 {
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

pub struct ConfidenceMonitor {
    pub confidences: Vec<f64>,
    pub entropies: Vec<f64>,
    pub capacity: usize,
}

impl ConfidenceMonitor {
    pub fn new(capacity: usize) -> Self {
        Self {
            confidences: Vec::with_capacity(capacity),
            entropies: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn record(&mut self, probabilities: &[f64]) {
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

    pub fn mean_confidence(&self) -> f64 {
        if self.confidences.is_empty() {
            return 0.0;
        }
        self.confidences.iter().sum::<f64>() / self.confidences.len() as f64
    }

    pub fn mean_entropy(&self) -> f64 {
        if self.entropies.is_empty() {
            return 0.0;
        }
        self.entropies.iter().sum::<f64>() / self.entropies.len() as f64
    }

    pub fn low_confidence_rate(&self, threshold: f64) -> f64 {
        if self.confidences.is_empty() {
            return 0.0;
        }
        self.confidences.iter().filter(|&&c| c < threshold).count() as f64
            / self.confidences.len() as f64
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.confidences.len()
    }
}

// --- Accuracy Tracker ---

pub struct AccuracyTracker {
    pub outcomes: Vec<bool>,
    pub capacity: usize,
    pub baseline_accuracy: f64,
}

impl AccuracyTracker {
    pub fn new(capacity: usize, baseline_accuracy: f64) -> Self {
        Self {
            outcomes: Vec::with_capacity(capacity),
            capacity,
            baseline_accuracy,
        }
    }

    pub fn record(&mut self, correct: bool) {
        if self.outcomes.len() >= self.capacity {
            self.outcomes.remove(0);
        }
        self.outcomes.push(correct);
    }

    pub fn accuracy(&self) -> f64 {
        if self.outcomes.is_empty() {
            return 0.0;
        }
        self.outcomes.iter().filter(|&&o| o).count() as f64 / self.outcomes.len() as f64
    }

    pub fn accuracy_drop(&self) -> f64 {
        (self.baseline_accuracy - self.accuracy()).max(0.0)
    }

    pub fn is_degraded(&self, tolerance: f64) -> bool {
        if self.outcomes.is_empty() {
            return false;
        }
        self.accuracy_drop() > tolerance
    }
}

// --- Data Generation ---

pub struct FeatureDistribution {
    pub means: [f64; NUM_FEATURES],
    pub stds: [f64; NUM_FEATURES],
}

pub fn baseline_distribution() -> FeatureDistribution {
    FeatureDistribution {
        means: [35.0, 55000.0, 700.0, 0.3],
        stds: [10.0, 15000.0, 50.0, 0.1],
    }
}

pub fn drifted_distribution() -> FeatureDistribution {
    FeatureDistribution {
        means: [42.0, 48000.0, 680.0, 0.4],
        stds: [12.0, 18000.0, 60.0, 0.15],
    }
}

pub fn extract_feature(samples: &[[f64; NUM_FEATURES]], feature_idx: usize) -> Vec<f64> {
    samples.iter().map(|s| s[feature_idx]).collect()
}

pub fn sample_features(
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

pub fn simulate_prediction(features: &[f64; NUM_FEATURES], seed: u64, drifted: bool) -> Vec<f64> {
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

pub fn simulate_ground_truth(features: &[f64; NUM_FEATURES], seed: u64) -> usize {
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
pub struct WindowStats {
    pub window_id: usize,
    pub mean_confidence: f64,
    pub mean_entropy: f64,
    pub accuracy: f64,
    pub alerts: Vec<DriftAlert>,
}
