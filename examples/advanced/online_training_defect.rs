//! Continuous Online Training (Defect Prediction)
//!
//! Incremental SGD and Passive-Aggressive algorithms for predicting software
//! defects from execution traces, with concept drift detection.
//!
//! ```bash
//! cargo run --example online_training_defect
//! ```

use std::collections::VecDeque;

pub const FEATURE_DIM: usize = 64;
pub const DRIFT_WINDOW: usize = 100;

// --- Execution Trace ---

#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    pub call_count: u64,
    pub max_depth: u32,
    pub memory_allocated: u64,
    pub memory_freed: u64,
    pub execution_time_us: u64,
    pub io_ops: u32,
    pub branch_misses: u32,
    pub has_defect: Option<bool>,
}

impl ExecutionTrace {
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
    #[must_use]
    pub fn with_defect(mut self, has_defect: bool) -> Self {
        self.has_defect = Some(has_defect);
        self
    }

    #[must_use]
    pub fn to_features(&self) -> FeatureVector {
        let mut f = [0.0_f32; FEATURE_DIM];
        f[0] = (self.call_count as f32).ln_1p() / 20.0;
        f[1] = self.max_depth as f32 / 100.0;
        f[2] = (self.memory_allocated as f32).ln_1p() / 30.0;
        f[3] = (self.memory_freed as f32).ln_1p() / 30.0;
        f[4] = (self.execution_time_us as f32).ln_1p() / 20.0;
        f[5] = self.io_ops as f32 / 1000.0;
        f[6] = self.branch_misses as f32 / 10000.0;
        f[7] = if self.memory_allocated > 0 {
            (1.0 - self.memory_freed as f32 / self.memory_allocated as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };
        f[8] = (self.call_count as f32 * self.max_depth as f32).ln_1p() / 25.0;
        f[9] = if self.execution_time_us > 0 {
            (self.io_ops as f32 / self.execution_time_us as f32 * 1000.0).min(1.0)
        } else {
            0.0
        };
        FeatureVector(f)
    }

    #[must_use]
    pub fn has_memory_leak_pattern(&self) -> bool {
        self.memory_allocated > self.memory_freed * 2 && self.memory_allocated > 1024
    }
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

// --- Feature Vector ---

#[derive(Debug, Clone)]
pub struct FeatureVector(pub [f32; FEATURE_DIM]);

impl FeatureVector {
    #[must_use]
    pub fn zeros() -> Self {
        Self([0.0; FEATURE_DIM])
    }
    #[must_use]
    pub fn dot(&self, weights: &[f32; FEATURE_DIM]) -> f32 {
        self.0.iter().zip(weights.iter()).map(|(x, w)| x * w).sum()
    }
    #[must_use]
    pub fn norm_squared(&self) -> f32 {
        self.0.iter().map(|x| x * x).sum()
    }
}

// --- Online SGD ---

#[derive(Debug, Clone)]
pub struct OnlineSGD {
    pub weights: [f32; FEATURE_DIM],
    pub bias: f32,
    pub learning_rate: f32,
    pub l2_reg: f32,
    pub samples_seen: u64,
}

impl OnlineSGD {
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
    #[must_use]
    pub fn with_l2_reg(mut self, l2: f32) -> Self {
        self.l2_reg = l2;
        self
    }
    #[must_use]
    pub fn predict_proba(&self, features: &FeatureVector) -> f32 {
        sigmoid(features.dot(&self.weights) + self.bias)
    }
    #[must_use]
    pub fn predict(&self, features: &FeatureVector) -> bool {
        self.predict_proba(features) > 0.5
    }

    pub fn update(&mut self, features: &FeatureVector, label: bool) {
        let error = self.predict_proba(features) - if label { 1.0 } else { 0.0 };
        for (w, &x) in self.weights.iter_mut().zip(features.0.iter()) {
            *w -= self.learning_rate * (error * x + self.l2_reg * *w);
        }
        self.bias -= self.learning_rate * error;
        self.samples_seen += 1;
    }

    #[must_use]
    pub fn stats(&self) -> ModelStats {
        ModelStats {
            samples_seen: self.samples_seen,
            weight_norm: self.weights.iter().map(|w| w * w).sum::<f32>().sqrt(),
            bias: self.bias,
        }
    }
}

// --- Passive-Aggressive ---

#[derive(Debug, Clone)]
pub struct PassiveAggressive {
    pub weights: [f32; FEATURE_DIM],
    pub c: f32,
    pub samples_seen: u64,
}

impl PassiveAggressive {
    #[must_use]
    pub fn new(c: f32) -> Self {
        Self {
            weights: [0.0; FEATURE_DIM],
            c,
            samples_seen: 0,
        }
    }
    #[must_use]
    pub fn predict_score(&self, features: &FeatureVector) -> f32 {
        features.dot(&self.weights)
    }
    #[must_use]
    pub fn predict(&self, features: &FeatureVector) -> bool {
        self.predict_score(features) > 0.0
    }

    pub fn update(&mut self, features: &FeatureVector, label: bool) {
        let y = if label { 1.0 } else { -1.0 };
        let loss = (1.0 - y * self.predict_score(features)).max(0.0);
        if loss > 0.0 {
            let tau = (loss / features.norm_squared().max(1e-10)).min(self.c);
            for (w, &x) in self.weights.iter_mut().zip(features.0.iter()) {
                *w += tau * y * x;
            }
        }
        self.samples_seen += 1;
    }
}

#[derive(Debug, Clone)]
pub struct ModelStats {
    pub samples_seen: u64,
    pub weight_norm: f32,
    pub bias: f32,
}

// --- Drift Detection ---

#[derive(Debug)]
pub struct DriftDetector {
    errors: VecDeque<f32>,
    warning_threshold: f32,
    drift_threshold: f32,
}

impl DriftDetector {
    #[must_use]
    pub fn new() -> Self {
        Self {
            errors: VecDeque::with_capacity(DRIFT_WINDOW),
            warning_threshold: 0.1,
            drift_threshold: 0.2,
        }
    }

    pub fn add_error(&mut self, predicted: bool, actual: bool) {
        if self.errors.len() >= DRIFT_WINDOW {
            self.errors.pop_front();
        }
        self.errors
            .push_back(if predicted == actual { 0.0 } else { 1.0 });
    }

    #[must_use]
    pub fn error_rate(&self) -> f32 {
        if self.errors.is_empty() {
            0.0
        } else {
            self.errors.iter().sum::<f32>() / self.errors.len() as f32
        }
    }

    #[must_use]
    pub fn detect_drift(&self) -> DriftStatus {
        if self.errors.len() < 10 {
            return DriftStatus::Stable;
        }
        let rate = self.error_rate();
        let mid = self.errors.len() / 2;
        let first: f32 = self.errors.iter().take(mid).sum::<f32>() / mid as f32;
        let second: f32 =
            self.errors.iter().skip(mid).sum::<f32>() / (self.errors.len() - mid) as f32;
        let diff = (second - first).abs();
        if diff > self.drift_threshold || rate > 0.4 {
            DriftStatus::Drift
        } else if diff > self.warning_threshold || rate > 0.3 {
            DriftStatus::Warning
        } else {
            DriftStatus::Stable
        }
    }

    pub fn reset(&mut self) {
        self.errors.clear();
    }
}

impl Default for DriftDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftStatus {
    Stable,
    Warning,
    Drift,
}

// --- Pipeline ---

pub struct OnlineTrainingPipeline {
    sgd: OnlineSGD,
    pa: PassiveAggressive,
    drift: DriftDetector,
    use_sgd: bool,
    metrics: PipelineMetrics,
}

impl OnlineTrainingPipeline {
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

    pub fn train(&mut self, trace: &ExecutionTrace) {
        let Some(label) = trace.has_defect else {
            return;
        };
        let features = trace.to_features();
        let predicted = if self.use_sgd {
            self.sgd.predict(&features)
        } else {
            self.pa.predict(&features)
        };
        self.metrics.update(predicted, label);
        self.drift.add_error(predicted, label);
        match self.drift.detect_drift() {
            DriftStatus::Drift => {
                self.use_sgd = !self.use_sgd;
                self.drift.reset();
            }
            DriftStatus::Warning => {
                self.sgd.update(&features, label);
                self.pa.update(&features, label);
                return;
            }
            DriftStatus::Stable => {}
        }
        if self.use_sgd {
            self.sgd.update(&features, label);
        } else {
            self.pa.update(&features, label);
        }
    }

    #[must_use]
    pub fn predict(&self, trace: &ExecutionTrace) -> DefectPrediction {
        let features = trace.to_features();
        let (probability, confidence) = if self.use_sgd {
            let p = self.sgd.predict_proba(&features);
            (p, (p - 0.5).abs() * 2.0)
        } else {
            let s = self.pa.predict_score(&features);
            (sigmoid(s), s.abs().min(1.0))
        };
        DefectPrediction {
            is_defect: probability > 0.5,
            probability,
            confidence,
            model_type: if self.use_sgd { "SGD" } else { "PA" },
        }
    }

    #[must_use]
    pub fn metrics(&self) -> &PipelineMetrics {
        &self.metrics
    }
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

#[derive(Debug, Clone)]
pub struct DefectPrediction {
    pub is_defect: bool,
    pub probability: f32,
    pub confidence: f32,
    pub model_type: &'static str,
}

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
    #[must_use]
    pub fn accuracy(&self) -> f32 {
        let t = self.total();
        if t == 0 {
            0.0
        } else {
            (self.true_positives + self.true_negatives) as f32 / t as f32
        }
    }
    #[must_use]
    pub fn precision(&self) -> f32 {
        let d = self.true_positives + self.false_positives;
        if d == 0 {
            0.0
        } else {
            self.true_positives as f32 / d as f32
        }
    }
    #[must_use]
    pub fn recall(&self) -> f32 {
        let d = self.true_positives + self.false_negatives;
        if d == 0 {
            0.0
        } else {
            self.true_positives as f32 / d as f32
        }
    }
    #[must_use]
    pub fn f1_score(&self) -> f32 {
        let (p, r) = (self.precision(), self.recall());
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
    #[must_use]
    pub fn total(&self) -> u64 {
        self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
    }
}

// --- Trace Generator ---

pub struct TraceGenerator {
    rng: SimpleRng,
    defect_rate: f32,
}

impl TraceGenerator {
    #[must_use]
    pub fn new(seed: u64, defect_rate: f32) -> Self {
        Self {
            rng: SimpleRng::new(seed),
            defect_rate: defect_rate.clamp(0.0, 1.0),
        }
    }

    pub fn generate(&mut self) -> ExecutionTrace {
        let is_defect = self.rng.next_f32() < self.defect_rate;
        let mut t = ExecutionTrace::new();
        if is_defect {
            match self.rng.next_u64() % 4 {
                0 => {
                    t.memory_allocated = 1_000_000 + self.rng.next_u64() % 10_000_000;
                    t.memory_freed = t.memory_allocated / 10;
                }
                1 => {
                    t.call_count = 500_000 + self.rng.next_u64() % 1_000_000;
                    t.io_ops = 0;
                    t.execution_time_us = 5_000_000;
                }
                2 => {
                    t.max_depth = 500 + (self.rng.next_u64() % 500) as u32;
                    t.call_count = 10000;
                }
                _ => {
                    t.io_ops = 50000 + (self.rng.next_u64() % 50000) as u32;
                    t.execution_time_us = 10_000_000;
                }
            }
        } else {
            t.call_count = 100 + self.rng.next_u64() % 10000;
            t.max_depth = 5 + (self.rng.next_u64() % 20) as u32;
            t.memory_allocated = 10000 + self.rng.next_u64() % 100000;
            t.memory_freed = t.memory_allocated - self.rng.next_u64() % 1000;
            t.execution_time_us = 1000 + self.rng.next_u64() % 50000;
            t.io_ops = (self.rng.next_u64() % 100) as u32;
            t.branch_misses = (self.rng.next_u64() % 1000) as u32;
        }
        t.has_defect = Some(is_defect);
        t
    }
}

// --- Utilities ---

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

fn main() {
    println!("=== Demo E: Continuous Online Training ===\n");
    let mut pipeline = OnlineTrainingPipeline::new();
    let mut generator = TraceGenerator::new(42, 0.2);

    println!("--- Training Phase (500 traces) ---");
    for i in 0..500 {
        pipeline.train(&generator.generate());
        if (i + 1) % 100 == 0 {
            let m = pipeline.metrics();
            println!(
                "  After {}: Acc={:.2}%, F1={:.3}, Drift={:?}",
                i + 1,
                m.accuracy() * 100.0,
                m.f1_score(),
                pipeline.drift_status()
            );
        }
    }

    let m = pipeline.metrics();
    println!("\n--- Final Metrics ---");
    println!(
        "Total: {}, Acc: {:.2}%, Prec: {:.2}%, Rec: {:.2}%, F1: {:.3}",
        m.total(),
        m.accuracy() * 100.0,
        m.precision() * 100.0,
        m.recall() * 100.0,
        m.f1_score()
    );

    println!("\n--- Predictions ---");
    for _ in 0..5 {
        let trace = generator.generate();
        let pred = pipeline.predict(&trace);
        println!(
            "  defect={:?} -> {} (p={:.2}, conf={:.2}, {})",
            trace.has_defect,
            if pred.is_defect { "DEFECT" } else { "OK" },
            pred.probability,
            pred.confidence,
            pred.model_type
        );
    }
    println!("\n=== Demo E Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_to_features_and_patterns() {
        let mut t = ExecutionTrace::new();
        t.call_count = 1000;
        t.max_depth = 10;
        assert!(t.to_features().0[0] > 0.0);
        t.memory_allocated = 10000;
        t.memory_freed = 1000;
        assert!(t.has_memory_leak_pattern());
        t.call_count = 500000;
        t.io_ops = 0;
        assert!(t.has_infinite_loop_pattern());
    }

    #[test]
    fn test_feature_vector_ops() {
        let fv = FeatureVector([1.0; FEATURE_DIM]);
        assert!((fv.dot(&[2.0; FEATURE_DIM]) - (FEATURE_DIM as f32 * 2.0)).abs() < 0.01);
        let mut arr = [0.0; FEATURE_DIM];
        arr[0] = 3.0;
        arr[1] = 4.0;
        assert!((FeatureVector(arr).norm_squared() - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_sgd_predict_and_update() {
        let mut sgd = OnlineSGD::new(0.1);
        assert!((sgd.predict_proba(&FeatureVector::zeros()) - 0.5).abs() < 0.01);
        sgd.update(&FeatureVector([0.5; FEATURE_DIM]), true);
        assert_eq!(sgd.samples_seen, 1);
    }

    #[test]
    fn test_pa_predict_and_update() {
        let mut pa = PassiveAggressive::new(1.0);
        assert!(!pa.predict(&FeatureVector::zeros()));
        pa.update(&FeatureVector([0.5; FEATURE_DIM]), true);
        assert_eq!(pa.samples_seen, 1);
    }

    #[test]
    fn test_drift_detector() {
        let mut dd = DriftDetector::new();
        for _ in 0..20 {
            dd.add_error(true, true);
        }
        assert_eq!(dd.detect_drift(), DriftStatus::Stable);
        dd = DriftDetector::new();
        for _ in 0..50 {
            dd.add_error(true, true);
        }
        for _ in 0..50 {
            dd.add_error(true, false);
        }
        assert_eq!(dd.detect_drift(), DriftStatus::Drift);
    }

    #[test]
    fn test_pipeline_train_and_predict() {
        let mut p = OnlineTrainingPipeline::new();
        p.train(&ExecutionTrace::new().with_defect(true));
        assert_eq!(p.metrics().total(), 1);
        let pred = p.predict(&ExecutionTrace::new());
        assert!((0.0..=1.0).contains(&pred.probability));
    }

    #[test]
    fn test_metrics() {
        let mut m = PipelineMetrics {
            true_positives: 80,
            true_negatives: 10,
            false_positives: 5,
            false_negatives: 5,
        };
        assert!((m.accuracy() - 0.9).abs() < 0.01);
        m = PipelineMetrics {
            true_positives: 80,
            true_negatives: 0,
            false_positives: 10,
            false_negatives: 10,
        };
        assert!(m.f1_score() > 0.85);
    }

    #[test]
    fn test_trace_generator() {
        let mut gen = TraceGenerator::new(42, 0.5);
        assert!(gen.generate().has_defect.is_some());
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
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_feature_vector_bounded(call_count in 0u64..1_000_000, max_depth in 0u32..1000) {
            let mut trace = ExecutionTrace::new();
            trace.call_count = call_count; trace.max_depth = max_depth;
            for &f in &trace.to_features().0 { prop_assert!(f.is_finite()); }
        }

        #[test]
        fn prop_sgd_probability_bounded(seed in 0u64..1000) {
            let sgd = OnlineSGD::new(0.01);
            let mut rng = SimpleRng::new(seed);
            let mut arr = [0.0; FEATURE_DIM];
            for v in &mut arr { *v = rng.next_f32(); }
            let prob = sgd.predict_proba(&FeatureVector(arr));
            prop_assert!(prob >= 0.0 && prob <= 1.0);
        }

        #[test]
        fn prop_metrics_total(tp in 0u64..100, tn in 0u64..100, fp in 0u64..100, fn_ in 0u64..100) {
            let m = PipelineMetrics { true_positives: tp, true_negatives: tn, false_positives: fp, false_negatives: fn_ };
            prop_assert_eq!(m.total(), tp + tn + fp + fn_);
            let acc = m.accuracy();
            prop_assert!(acc >= 0.0 && acc <= 1.0);
        }
    }
}
