//! # Demo E: Continuous Online Training (Defect Prediction)
//!
//! Implements continuous online learning from execution traces to predict
//! software defects using incremental SGD and Passive-Aggressive algorithms.
//!
//! ## Toyota Way Principles
//!
//! - **Jidoka**: Automatic defect detection stops bad code
//! - **Kaizen**: Model continuously improves from new data
//! - **Genchi Genbutsu**: Learn from real execution traces
//!
//! ## Architecture
//!
//! ```text
//! Traces → Feature Extraction → Online Learner → Predictions
//!                                     ↑
//!                              Feedback Loop
//! ```

use std::collections::VecDeque;

/// Feature vector dimension for trace features
pub const FEATURE_DIM: usize = 64;

/// Maximum samples to retain for drift detection
pub const DRIFT_WINDOW: usize = 100;

// ============================================================================
// Execution Trace Features
// ============================================================================

/// Execution trace from profiling
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    /// Function call count
    pub call_count: u64,
    /// Maximum call depth
    pub max_depth: u32,
    /// Memory allocated (bytes)
    pub memory_allocated: u64,
    /// Memory freed (bytes)
    pub memory_freed: u64,
    /// Execution time (microseconds)
    pub execution_time_us: u64,
    /// I/O operations count
    pub io_ops: u32,
    /// Branch misprediction estimate
    pub branch_misses: u32,
    /// Whether this trace had a defect (label)
    pub has_defect: Option<bool>,
}

impl ExecutionTrace {
    /// Create a new trace
    #[must_use]
    pub fn new() -> Self {
        Self {
            call_count: 0,
            max_depth: 0,
            memory_allocated: 0,
            memory_freed: 0,
            execution_time_us: 0,
            io_ops: 0,
            branch_misses: 0,
            has_defect: None,
        }
    }

    /// Create trace with label
    #[must_use]
    pub fn with_defect(mut self, has_defect: bool) -> Self {
        self.has_defect = Some(has_defect);
        self
    }

    /// Extract feature vector from trace
    #[must_use]
    pub fn to_features(&self) -> FeatureVector {
        let mut features = [0.0_f32; FEATURE_DIM];

        // Normalize features to 0-1 range
        features[0] = (self.call_count as f32).ln_1p() / 20.0;
        features[1] = self.max_depth as f32 / 100.0;
        features[2] = (self.memory_allocated as f32).ln_1p() / 30.0;
        features[3] = (self.memory_freed as f32).ln_1p() / 30.0;
        features[4] = (self.execution_time_us as f32).ln_1p() / 20.0;
        features[5] = self.io_ops as f32 / 1000.0;
        features[6] = self.branch_misses as f32 / 10000.0;

        // Memory leak indicator
        let leak_ratio = if self.memory_allocated > 0 {
            1.0 - (self.memory_freed as f32 / self.memory_allocated as f32)
        } else {
            0.0
        };
        features[7] = leak_ratio.clamp(0.0, 1.0);

        // Complexity indicator
        features[8] = (self.call_count as f32 * self.max_depth as f32).ln_1p() / 25.0;

        // I/O intensity
        features[9] = if self.execution_time_us > 0 {
            (self.io_ops as f32 / self.execution_time_us as f32 * 1000.0).min(1.0)
        } else {
            0.0
        };

        FeatureVector(features)
    }

    /// Check if trace indicates memory leak pattern
    #[must_use]
    pub fn has_memory_leak_pattern(&self) -> bool {
        self.memory_allocated > self.memory_freed * 2 && self.memory_allocated > 1024
    }

    /// Check if trace indicates infinite loop pattern
    #[must_use]
    pub fn has_infinite_loop_pattern(&self) -> bool {
        self.call_count > 100000 && self.io_ops == 0
    }
}

impl Default for ExecutionTrace {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature vector for ML model
#[derive(Debug, Clone)]
pub struct FeatureVector(pub [f32; FEATURE_DIM]);

impl FeatureVector {
    /// Create zero vector
    #[must_use]
    pub fn zeros() -> Self {
        Self([0.0; FEATURE_DIM])
    }

    /// Dot product with weights
    #[must_use]
    pub fn dot(&self, weights: &[f32; FEATURE_DIM]) -> f32 {
        self.0.iter().zip(weights.iter()).map(|(x, w)| x * w).sum()
    }

    /// L2 norm squared
    #[must_use]
    pub fn norm_squared(&self) -> f32 {
        self.0.iter().map(|x| x * x).sum()
    }
}

// ============================================================================
// Online Learning Algorithms
// ============================================================================

/// Online SGD learner for binary classification
#[derive(Debug, Clone)]
pub struct OnlineSGD {
    /// Model weights
    pub weights: [f32; FEATURE_DIM],
    /// Bias term
    pub bias: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// L2 regularization
    pub l2_reg: f32,
    /// Samples seen
    pub samples_seen: u64,
}

impl OnlineSGD {
    /// Create new SGD learner
    #[must_use]
    pub fn new(learning_rate: f32) -> Self {
        Self {
            weights: [0.0; FEATURE_DIM],
            bias: 0.0,
            learning_rate,
            l2_reg: 0.001,
            samples_seen: 0,
        }
    }

    /// Set L2 regularization
    #[must_use]
    pub fn with_l2_reg(mut self, l2: f32) -> Self {
        self.l2_reg = l2;
        self
    }

    /// Predict probability of defect (sigmoid)
    #[must_use]
    pub fn predict_proba(&self, features: &FeatureVector) -> f32 {
        let logit = features.dot(&self.weights) + self.bias;
        sigmoid(logit)
    }

    /// Predict binary label
    #[must_use]
    pub fn predict(&self, features: &FeatureVector) -> bool {
        self.predict_proba(features) > 0.5
    }

    /// Update model with single sample
    pub fn update(&mut self, features: &FeatureVector, label: bool) {
        let y = if label { 1.0 } else { 0.0 };
        let pred = self.predict_proba(features);
        let error = pred - y;

        // Gradient descent step
        for (w, &x) in self.weights.iter_mut().zip(features.0.iter()) {
            *w -= self.learning_rate * (error * x + self.l2_reg * *w);
        }
        self.bias -= self.learning_rate * error;

        self.samples_seen += 1;
    }

    /// Get model statistics
    #[must_use]
    pub fn stats(&self) -> ModelStats {
        let weight_norm: f32 = self.weights.iter().map(|w| w * w).sum::<f32>().sqrt();
        ModelStats {
            samples_seen: self.samples_seen,
            weight_norm,
            bias: self.bias,
        }
    }
}

/// Passive-Aggressive learner (margin-based online learning)
#[derive(Debug, Clone)]
pub struct PassiveAggressive {
    /// Model weights
    pub weights: [f32; FEATURE_DIM],
    /// Aggressiveness parameter
    pub c: f32,
    /// Samples seen
    pub samples_seen: u64,
}

impl PassiveAggressive {
    /// Create new PA learner
    #[must_use]
    pub fn new(c: f32) -> Self {
        Self {
            weights: [0.0; FEATURE_DIM],
            c,
            samples_seen: 0,
        }
    }

    /// Predict score (not probability)
    #[must_use]
    pub fn predict_score(&self, features: &FeatureVector) -> f32 {
        features.dot(&self.weights)
    }

    /// Predict binary label
    #[must_use]
    pub fn predict(&self, features: &FeatureVector) -> bool {
        self.predict_score(features) > 0.0
    }

    /// Update with hinge loss
    pub fn update(&mut self, features: &FeatureVector, label: bool) {
        let y = if label { 1.0 } else { -1.0 };
        let score = self.predict_score(features);
        let loss = (1.0 - y * score).max(0.0);

        if loss > 0.0 {
            let norm_sq = features.norm_squared().max(1e-10);
            let tau = (loss / norm_sq).min(self.c);

            for (w, &x) in self.weights.iter_mut().zip(features.0.iter()) {
                *w += tau * y * x;
            }
        }

        self.samples_seen += 1;
    }
}

/// Model statistics
#[derive(Debug, Clone)]
pub struct ModelStats {
    pub samples_seen: u64,
    pub weight_norm: f32,
    pub bias: f32,
}

// ============================================================================
// Concept Drift Detection
// ============================================================================

/// Drift detector using ADWIN-like approach
#[derive(Debug)]
pub struct DriftDetector {
    /// Recent prediction errors
    errors: VecDeque<f32>,
    /// Warning threshold
    warning_threshold: f32,
    /// Drift threshold
    drift_threshold: f32,
}

impl DriftDetector {
    /// Create new drift detector
    #[must_use]
    pub fn new() -> Self {
        Self {
            errors: VecDeque::with_capacity(DRIFT_WINDOW),
            warning_threshold: 0.1,
            drift_threshold: 0.2,
        }
    }

    /// Add prediction error
    pub fn add_error(&mut self, predicted: bool, actual: bool) {
        let error = if predicted == actual { 0.0 } else { 1.0 };

        if self.errors.len() >= DRIFT_WINDOW {
            self.errors.pop_front();
        }
        self.errors.push_back(error);
    }

    /// Calculate current error rate
    #[must_use]
    pub fn error_rate(&self) -> f32 {
        if self.errors.is_empty() {
            return 0.0;
        }
        self.errors.iter().sum::<f32>() / self.errors.len() as f32
    }

    /// Check for drift
    #[must_use]
    pub fn detect_drift(&self) -> DriftStatus {
        if self.errors.len() < 10 {
            return DriftStatus::Stable;
        }

        let rate = self.error_rate();

        // Compare first and second half
        let mid = self.errors.len() / 2;
        let first_half: f32 = self.errors.iter().take(mid).sum::<f32>() / mid as f32;
        let second_half: f32 =
            self.errors.iter().skip(mid).sum::<f32>() / (self.errors.len() - mid) as f32;

        let diff = (second_half - first_half).abs();

        if diff > self.drift_threshold || rate > 0.4 {
            DriftStatus::Drift
        } else if diff > self.warning_threshold || rate > 0.3 {
            DriftStatus::Warning
        } else {
            DriftStatus::Stable
        }
    }

    /// Reset detector
    pub fn reset(&mut self) {
        self.errors.clear();
    }
}

impl Default for DriftDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Drift detection status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftStatus {
    /// No drift detected
    Stable,
    /// Potential drift warning
    Warning,
    /// Drift confirmed
    Drift,
}

// ============================================================================
// Online Training Pipeline
// ============================================================================

/// Complete online training pipeline
pub struct OnlineTrainingPipeline {
    /// Primary learner
    sgd: OnlineSGD,
    /// Backup learner (PA)
    pa: PassiveAggressive,
    /// Drift detector
    drift: DriftDetector,
    /// Use SGD or PA
    use_sgd: bool,
    /// Performance metrics
    metrics: PipelineMetrics,
}

impl OnlineTrainingPipeline {
    /// Create new pipeline
    #[must_use]
    pub fn new() -> Self {
        Self {
            sgd: OnlineSGD::new(0.01),
            pa: PassiveAggressive::new(1.0),
            drift: DriftDetector::new(),
            use_sgd: true,
            metrics: PipelineMetrics::new(),
        }
    }

    /// Process a labeled trace
    pub fn train(&mut self, trace: &ExecutionTrace) {
        let Some(label) = trace.has_defect else {
            return;
        };

        let features = trace.to_features();

        // Get prediction before update
        let predicted = if self.use_sgd {
            self.sgd.predict(&features)
        } else {
            self.pa.predict(&features)
        };

        // Update metrics
        self.metrics.update(predicted, label);

        // Update drift detector
        self.drift.add_error(predicted, label);

        // Check for drift and possibly switch models
        match self.drift.detect_drift() {
            DriftStatus::Drift => {
                self.use_sgd = !self.use_sgd;
                self.drift.reset();
            }
            DriftStatus::Warning => {
                // Train both models
                self.sgd.update(&features, label);
                self.pa.update(&features, label);
                return;
            }
            DriftStatus::Stable => {}
        }

        // Update active model
        if self.use_sgd {
            self.sgd.update(&features, label);
        } else {
            self.pa.update(&features, label);
        }
    }

    /// Predict defect probability
    #[must_use]
    pub fn predict(&self, trace: &ExecutionTrace) -> DefectPrediction {
        let features = trace.to_features();

        let (probability, confidence) = if self.use_sgd {
            let p = self.sgd.predict_proba(&features);
            (p, (p - 0.5).abs() * 2.0)
        } else {
            let score = self.pa.predict_score(&features);
            let p = sigmoid(score);
            (p, score.abs().min(1.0))
        };

        DefectPrediction {
            is_defect: probability > 0.5,
            probability,
            confidence,
            model_type: if self.use_sgd { "SGD" } else { "PA" },
        }
    }

    /// Get pipeline metrics
    #[must_use]
    pub fn metrics(&self) -> &PipelineMetrics {
        &self.metrics
    }

    /// Get drift status
    #[must_use]
    pub fn drift_status(&self) -> DriftStatus {
        self.drift.detect_drift()
    }
}

impl Default for OnlineTrainingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Defect prediction result
#[derive(Debug, Clone)]
pub struct DefectPrediction {
    pub is_defect: bool,
    pub probability: f32,
    pub confidence: f32,
    pub model_type: &'static str,
}

/// Pipeline performance metrics
#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    pub true_positives: u64,
    pub true_negatives: u64,
    pub false_positives: u64,
    pub false_negatives: u64,
}

impl PipelineMetrics {
    fn new() -> Self {
        Self {
            true_positives: 0,
            true_negatives: 0,
            false_positives: 0,
            false_negatives: 0,
        }
    }

    fn update(&mut self, predicted: bool, actual: bool) {
        match (predicted, actual) {
            (true, true) => self.true_positives += 1,
            (false, false) => self.true_negatives += 1,
            (true, false) => self.false_positives += 1,
            (false, true) => self.false_negatives += 1,
        }
    }

    /// Calculate accuracy
    #[must_use]
    pub fn accuracy(&self) -> f32 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        (self.true_positives + self.true_negatives) as f32 / total as f32
    }

    /// Calculate precision
    #[must_use]
    pub fn precision(&self) -> f32 {
        let denom = self.true_positives + self.false_positives;
        if denom == 0 {
            return 0.0;
        }
        self.true_positives as f32 / denom as f32
    }

    /// Calculate recall
    #[must_use]
    pub fn recall(&self) -> f32 {
        let denom = self.true_positives + self.false_negatives;
        if denom == 0 {
            return 0.0;
        }
        self.true_positives as f32 / denom as f32
    }

    /// Calculate F1 score
    #[must_use]
    pub fn f1_score(&self) -> f32 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            return 0.0;
        }
        2.0 * p * r / (p + r)
    }

    /// Total samples
    #[must_use]
    pub fn total(&self) -> u64 {
        self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
    }
}

// ============================================================================
// Trace Generator (for testing)
// ============================================================================

/// Generate synthetic traces for testing
pub struct TraceGenerator {
    rng: SimpleRng,
    defect_rate: f32,
}

impl TraceGenerator {
    /// Create generator with defect rate
    #[must_use]
    pub fn new(seed: u64, defect_rate: f32) -> Self {
        Self {
            rng: SimpleRng::new(seed),
            defect_rate: defect_rate.clamp(0.0, 1.0),
        }
    }

    /// Generate a trace (with realistic defect patterns)
    pub fn generate(&mut self) -> ExecutionTrace {
        let is_defect = self.rng.next_f32() < self.defect_rate;

        let mut trace = ExecutionTrace::new();

        if is_defect {
            // Generate defective trace patterns
            match self.rng.next_u64() % 4 {
                0 => {
                    // Memory leak
                    trace.memory_allocated = 1_000_000 + self.rng.next_u64() % 10_000_000;
                    trace.memory_freed = trace.memory_allocated / 10;
                }
                1 => {
                    // Infinite loop pattern
                    trace.call_count = 500_000 + self.rng.next_u64() % 1_000_000;
                    trace.io_ops = 0;
                    trace.execution_time_us = 5_000_000;
                }
                2 => {
                    // Deep recursion
                    trace.max_depth = 500 + (self.rng.next_u64() % 500) as u32;
                    trace.call_count = 10000;
                }
                _ => {
                    // Resource exhaustion
                    trace.io_ops = 50000 + (self.rng.next_u64() % 50000) as u32;
                    trace.execution_time_us = 10_000_000;
                }
            }
        } else {
            // Normal trace
            trace.call_count = 100 + self.rng.next_u64() % 10000;
            trace.max_depth = 5 + (self.rng.next_u64() % 20) as u32;
            trace.memory_allocated = 10000 + self.rng.next_u64() % 100000;
            trace.memory_freed = trace.memory_allocated - self.rng.next_u64() % 1000;
            trace.execution_time_us = 1000 + self.rng.next_u64() % 50000;
            trace.io_ops = (self.rng.next_u64() % 100) as u32;
            trace.branch_misses = (self.rng.next_u64() % 1000) as u32;
        }

        trace.has_defect = Some(is_defect);
        trace
    }
}

// ============================================================================
// Utilities
// ============================================================================

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("=== Demo E: Continuous Online Training ===\n");

    let mut pipeline = OnlineTrainingPipeline::new();
    let mut generator = TraceGenerator::new(42, 0.2); // 20% defect rate

    println!("--- Training Phase (500 traces) ---");
    for i in 0..500 {
        let trace = generator.generate();
        pipeline.train(&trace);

        if (i + 1) % 100 == 0 {
            let m = pipeline.metrics();
            println!(
                "  After {} traces: Acc={:.2}%, F1={:.3}, Drift={:?}",
                i + 1,
                m.accuracy() * 100.0,
                m.f1_score(),
                pipeline.drift_status()
            );
        }
    }

    println!("\n--- Final Metrics ---");
    let m = pipeline.metrics();
    println!("Total samples: {}", m.total());
    println!("Accuracy: {:.2}%", m.accuracy() * 100.0);
    println!("Precision: {:.2}%", m.precision() * 100.0);
    println!("Recall: {:.2}%", m.recall() * 100.0);
    println!("F1 Score: {:.3}", m.f1_score());

    println!("\n--- Prediction Examples ---");
    for _ in 0..5 {
        let trace = generator.generate();
        let pred = pipeline.predict(&trace);
        println!(
            "  Trace: defect={:?} → Predicted: {} (prob={:.2}, conf={:.2}, model={})",
            trace.has_defect,
            if pred.is_defect { "DEFECT" } else { "OK" },
            pred.probability,
            pred.confidence,
            pred.model_type
        );
    }

    println!("\n=== Demo E Complete ===");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_trace_new() {
        let trace = ExecutionTrace::new();
        assert_eq!(trace.call_count, 0);
        assert!(trace.has_defect.is_none());
    }

    #[test]
    fn test_execution_trace_with_defect() {
        let trace = ExecutionTrace::new().with_defect(true);
        assert_eq!(trace.has_defect, Some(true));
    }

    #[test]
    fn test_trace_to_features() {
        let mut trace = ExecutionTrace::new();
        trace.call_count = 1000;
        trace.max_depth = 10;
        let features = trace.to_features();
        assert!(features.0[0] > 0.0); // call_count feature
    }

    #[test]
    fn test_memory_leak_pattern() {
        let mut trace = ExecutionTrace::new();
        trace.memory_allocated = 10000;
        trace.memory_freed = 1000;
        assert!(trace.has_memory_leak_pattern());
    }

    #[test]
    fn test_no_memory_leak() {
        let mut trace = ExecutionTrace::new();
        trace.memory_allocated = 10000;
        trace.memory_freed = 9000;
        assert!(!trace.has_memory_leak_pattern());
    }

    #[test]
    fn test_infinite_loop_pattern() {
        let mut trace = ExecutionTrace::new();
        trace.call_count = 500000;
        trace.io_ops = 0;
        assert!(trace.has_infinite_loop_pattern());
    }

    #[test]
    fn test_feature_vector_zeros() {
        let fv = FeatureVector::zeros();
        assert!(fv.0.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_feature_vector_dot() {
        let fv = FeatureVector([1.0; FEATURE_DIM]);
        let weights = [2.0; FEATURE_DIM];
        let result = fv.dot(&weights);
        assert!((result - (FEATURE_DIM as f32 * 2.0)).abs() < 0.01);
    }

    #[test]
    fn test_feature_vector_norm() {
        let mut arr = [0.0; FEATURE_DIM];
        arr[0] = 3.0;
        arr[1] = 4.0;
        let fv = FeatureVector(arr);
        assert!((fv.norm_squared() - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_online_sgd_new() {
        let sgd = OnlineSGD::new(0.01);
        assert_eq!(sgd.samples_seen, 0);
        assert!((sgd.learning_rate - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_online_sgd_predict() {
        let sgd = OnlineSGD::new(0.01);
        let fv = FeatureVector::zeros();
        let prob = sgd.predict_proba(&fv);
        assert!((prob - 0.5).abs() < 0.01); // Sigmoid(0) = 0.5
    }

    #[test]
    fn test_online_sgd_update() {
        let mut sgd = OnlineSGD::new(0.1);
        let fv = FeatureVector([0.5; FEATURE_DIM]);
        sgd.update(&fv, true);
        assert_eq!(sgd.samples_seen, 1);
    }

    #[test]
    fn test_passive_aggressive_new() {
        let pa = PassiveAggressive::new(1.0);
        assert_eq!(pa.samples_seen, 0);
    }

    #[test]
    fn test_passive_aggressive_predict() {
        let pa = PassiveAggressive::new(1.0);
        let fv = FeatureVector::zeros();
        assert!(!pa.predict(&fv)); // Score 0 → false
    }

    #[test]
    fn test_passive_aggressive_update() {
        let mut pa = PassiveAggressive::new(1.0);
        let fv = FeatureVector([0.5; FEATURE_DIM]);
        pa.update(&fv, true);
        assert_eq!(pa.samples_seen, 1);
    }

    #[test]
    fn test_drift_detector_new() {
        let dd = DriftDetector::new();
        assert!((dd.error_rate() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_drift_detector_add_error() {
        let mut dd = DriftDetector::new();
        dd.add_error(true, false);
        assert!((dd.error_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_drift_detector_stable() {
        let mut dd = DriftDetector::new();
        for _ in 0..20 {
            dd.add_error(true, true);
        }
        assert_eq!(dd.detect_drift(), DriftStatus::Stable);
    }

    #[test]
    fn test_drift_detector_drift() {
        let mut dd = DriftDetector::new();
        // First half correct
        for _ in 0..50 {
            dd.add_error(true, true);
        }
        // Second half wrong
        for _ in 0..50 {
            dd.add_error(true, false);
        }
        assert_eq!(dd.detect_drift(), DriftStatus::Drift);
    }

    #[test]
    fn test_pipeline_new() {
        let pipeline = OnlineTrainingPipeline::new();
        assert_eq!(pipeline.metrics().total(), 0);
    }

    #[test]
    fn test_pipeline_train() {
        let mut pipeline = OnlineTrainingPipeline::new();
        let trace = ExecutionTrace::new().with_defect(true);
        pipeline.train(&trace);
        assert_eq!(pipeline.metrics().total(), 1);
    }

    #[test]
    fn test_pipeline_predict() {
        let pipeline = OnlineTrainingPipeline::new();
        let trace = ExecutionTrace::new();
        let pred = pipeline.predict(&trace);
        assert!(pred.probability >= 0.0 && pred.probability <= 1.0);
    }

    #[test]
    fn test_pipeline_metrics_accuracy() {
        let mut m = PipelineMetrics::new();
        m.true_positives = 80;
        m.true_negatives = 10;
        m.false_positives = 5;
        m.false_negatives = 5;
        assert!((m.accuracy() - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_pipeline_metrics_precision() {
        let mut m = PipelineMetrics::new();
        m.true_positives = 80;
        m.false_positives = 20;
        assert!((m.precision() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_pipeline_metrics_recall() {
        let mut m = PipelineMetrics::new();
        m.true_positives = 80;
        m.false_negatives = 20;
        assert!((m.recall() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_pipeline_metrics_f1() {
        let mut m = PipelineMetrics::new();
        m.true_positives = 80;
        m.false_positives = 10;
        m.false_negatives = 10;
        // P = 80/90 ≈ 0.889, R = 80/90 ≈ 0.889
        // F1 = 2*0.889*0.889 / (0.889+0.889) ≈ 0.889
        assert!(m.f1_score() > 0.85);
    }

    #[test]
    fn test_trace_generator() {
        let mut gen = TraceGenerator::new(42, 0.5);
        let trace = gen.generate();
        assert!(trace.has_defect.is_some());
    }

    #[test]
    fn test_trace_generator_deterministic() {
        let mut gen1 = TraceGenerator::new(42, 0.5);
        let mut gen2 = TraceGenerator::new(42, 0.5);
        let t1 = gen1.generate();
        let t2 = gen2.generate();
        assert_eq!(t1.has_defect, t2.has_defect);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.01);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_feature_vector_bounded(
            call_count in 0u64..1_000_000,
            max_depth in 0u32..1000
        ) {
            let mut trace = ExecutionTrace::new();
            trace.call_count = call_count;
            trace.max_depth = max_depth;
            let features = trace.to_features();
            for &f in &features.0 {
                prop_assert!(f.is_finite());
            }
        }

        #[test]
        fn prop_sgd_probability_bounded(seed in 0u64..1000) {
            let sgd = OnlineSGD::new(0.01);
            let mut rng = SimpleRng::new(seed);
            let mut arr = [0.0; FEATURE_DIM];
            for v in &mut arr {
                *v = rng.next_f32();
            }
            let fv = FeatureVector(arr);
            let prob = sgd.predict_proba(&fv);
            prop_assert!(prob >= 0.0);
            prop_assert!(prob <= 1.0);
        }

        #[test]
        fn prop_pa_update_increases_samples(n in 1usize..50) {
            let mut pa = PassiveAggressive::new(1.0);
            let fv = FeatureVector([0.5; FEATURE_DIM]);
            for _ in 0..n {
                pa.update(&fv, true);
            }
            prop_assert_eq!(pa.samples_seen, n as u64);
        }

        #[test]
        fn prop_metrics_total_correct(tp in 0u64..100, tn in 0u64..100, fp in 0u64..100, fn_ in 0u64..100) {
            let m = PipelineMetrics {
                true_positives: tp,
                true_negatives: tn,
                false_positives: fp,
                false_negatives: fn_,
            };
            prop_assert_eq!(m.total(), tp + tn + fp + fn_);
        }

        #[test]
        fn prop_accuracy_bounded(tp in 0u64..100, tn in 0u64..100, fp in 0u64..100, fn_ in 0u64..100) {
            let m = PipelineMetrics {
                true_positives: tp,
                true_negatives: tn,
                false_positives: fp,
                false_negatives: fn_,
            };
            let acc = m.accuracy();
            prop_assert!(acc >= 0.0);
            prop_assert!(acc <= 1.0);
        }

        #[test]
        fn prop_pipeline_training_increments(n in 1usize..20) {
            let mut pipeline = OnlineTrainingPipeline::new();
            for i in 0..n {
                let trace = ExecutionTrace::new().with_defect(i % 2 == 0);
                pipeline.train(&trace);
            }
            prop_assert_eq!(pipeline.metrics().total(), n as u64);
        }

        #[test]
        fn prop_drift_detector_bounded_size(n in 1usize..200) {
            let mut dd = DriftDetector::new();
            for i in 0..n {
                dd.add_error(i % 2 == 0, true);
            }
            prop_assert!(dd.errors.len() <= DRIFT_WINDOW);
        }

        #[test]
        fn prop_generator_produces_valid_traces(seed in 0u64..1000, n in 1usize..20) {
            let mut gen = TraceGenerator::new(seed, 0.3);
            for _ in 0..n {
                let trace = gen.generate();
                prop_assert!(trace.has_defect.is_some());
            }
        }

        #[test]
        fn prop_norm_squared_non_negative(seed in 0u64..1000) {
            let mut rng = SimpleRng::new(seed);
            let mut arr = [0.0; FEATURE_DIM];
            for v in &mut arr {
                *v = rng.next_f32() * 2.0 - 1.0;
            }
            let fv = FeatureVector(arr);
            prop_assert!(fv.norm_squared() >= 0.0);
        }

        #[test]
        fn prop_sigmoid_bounded(x in -100.0f32..100.0) {
            let s = sigmoid(x);
            prop_assert!(s >= 0.0);
            prop_assert!(s <= 1.0);
        }
    }
}
