//! Model A/B Testing Example
//!
//! Demonstrates serving two model versions simultaneously with traffic
//! splitting, collecting metrics for each variant, and computing
//! statistical significance of performance differences.
//!
//! # Architecture
//!
//! ```text
//! Request → [Router] → 70% → Model A (baseline)
//!                    → 30% → Model B (candidate)
//!           [Metrics Collector] → Statistical Test → Winner
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example model_ab_testing
//! ```
//!
//! ## References
//! - Crankshaw, D. et al. (2017). *Clipper: A Low-Latency Online Prediction Serving System*. NSDI. arXiv:1612.03079

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const INPUT_DIM: usize = 16;
const OUTPUT_DIM: usize = 4;

/// Deterministic weight initialization
fn init_weights(size: usize, seed: u64) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let mut h = DefaultHasher::new();
            (seed, i).hash(&mut h);
            (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.2
        })
        .collect()
}

/// Simple inference model
struct Model {
    weights: Vec<f32>,
    bias: Vec<f32>,
    version: String,
}

impl Model {
    fn new(version: &str, seed: u64) -> Self {
        Self {
            weights: init_weights(OUTPUT_DIM * INPUT_DIM, seed),
            bias: init_weights(OUTPUT_DIM, seed + 1),
            version: version.to_string(),
        }
    }

    fn predict(&self, input: &[f32]) -> Vec<f32> {
        let mut output = self.bias.clone();
        for (o, out) in output.iter_mut().enumerate() {
            for (i, &x) in input.iter().enumerate() {
                *out += self.weights[o * INPUT_DIM + i] * x;
            }
        }
        // Softmax
        let max = output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = output.iter().map(|&o| (o - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }
}

/// A/B test variant assignment
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    A,
    B,
}

// Variant names are used via Display-style formatting in main output

/// Traffic router with deterministic splitting
struct Router {
    b_traffic_pct: f32,
}

impl Router {
    fn new(b_traffic_pct: f32) -> Self {
        Self { b_traffic_pct }
    }

    fn assign(&self, request_id: u64) -> Variant {
        let mut h = DefaultHasher::new();
        request_id.hash(&mut h);
        let hash_val = h.finish() as f32 / u64::MAX as f32;
        if hash_val < self.b_traffic_pct {
            Variant::B
        } else {
            Variant::A
        }
    }
}

/// Per-variant metrics collector
struct VariantMetrics {
    latencies_us: Vec<u64>,
    predictions_correct: usize,
    predictions_total: usize,
    confidence_sum: f32,
}

impl VariantMetrics {
    fn new() -> Self {
        Self {
            latencies_us: Vec::new(),
            predictions_correct: 0,
            predictions_total: 0,
            confidence_sum: 0.0,
        }
    }

    fn record(&mut self, latency_us: u64, correct: bool, confidence: f32) {
        self.latencies_us.push(latency_us);
        self.predictions_total += 1;
        if correct {
            self.predictions_correct += 1;
        }
        self.confidence_sum += confidence;
    }

    fn accuracy(&self) -> f64 {
        if self.predictions_total == 0 {
            return 0.0;
        }
        self.predictions_correct as f64 / self.predictions_total as f64
    }

    fn avg_confidence(&self) -> f32 {
        if self.predictions_total == 0 {
            return 0.0;
        }
        self.confidence_sum / self.predictions_total as f32
    }

    fn avg_latency_us(&self) -> f64 {
        if self.latencies_us.is_empty() {
            return 0.0;
        }
        self.latencies_us.iter().sum::<u64>() as f64 / self.latencies_us.len() as f64
    }

    fn p95_latency_us(&self) -> u64 {
        if self.latencies_us.is_empty() {
            return 0;
        }
        let mut sorted = self.latencies_us.clone();
        sorted.sort_unstable();
        let idx = (sorted.len() as f64 * 0.95) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

/// A/B test experiment
struct Experiment {
    model_a: Model,
    model_b: Model,
    router: Router,
    metrics_a: VariantMetrics,
    metrics_b: VariantMetrics,
}

impl Experiment {
    fn new(model_a: Model, model_b: Model, b_traffic_pct: f32) -> Self {
        Self {
            model_a,
            model_b,
            router: Router::new(b_traffic_pct),
            metrics_a: VariantMetrics::new(),
            metrics_b: VariantMetrics::new(),
        }
    }

    fn process_request(&mut self, request_id: u64, input: &[f32], true_label: usize) {
        let variant = self.router.assign(request_id);

        let start = std::time::Instant::now();
        let probs = match variant {
            Variant::A => self.model_a.predict(input),
            Variant::B => self.model_b.predict(input),
        };
        let latency_us = start.elapsed().as_micros() as u64;

        let predicted = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map_or(0, |(i, _)| i);
        let correct = predicted == true_label;
        let confidence = probs[predicted];

        match variant {
            Variant::A => self.metrics_a.record(latency_us, correct, confidence),
            Variant::B => self.metrics_b.record(latency_us, correct, confidence),
        }
    }

    /// Compute two-proportion z-test for accuracy difference
    fn significance_test(&self) -> (f64, bool) {
        let n_a = self.metrics_a.predictions_total as f64;
        let n_b = self.metrics_b.predictions_total as f64;
        if n_a < 10.0 || n_b < 10.0 {
            return (0.0, false);
        }

        let p_a = self.metrics_a.accuracy();
        let p_b = self.metrics_b.accuracy();
        let p_pooled = (self.metrics_a.predictions_correct + self.metrics_b.predictions_correct)
            as f64
            / (n_a + n_b);

        let se = (p_pooled * (1.0 - p_pooled) * (1.0 / n_a + 1.0 / n_b)).sqrt();
        if se < 1e-10 {
            return (0.0, false);
        }

        let z = (p_b - p_a) / se;
        // p < 0.05 corresponds to |z| > 1.96
        let significant = z.abs() > 1.96;
        (z, significant)
    }
}

/// Generate labeled test data
fn generate_data(n: usize, seed: u64) -> Vec<(Vec<f32>, usize)> {
    (0..n)
        .map(|i| {
            let input: Vec<f32> = (0..INPUT_DIM)
                .map(|j| {
                    let mut h = DefaultHasher::new();
                    (seed, i, j).hash(&mut h);
                    h.finish() as f32 / u64::MAX as f32 - 0.5
                })
                .collect();
            let mut h = DefaultHasher::new();
            (seed, "label", i).hash(&mut h);
            let label = h.finish() as usize % OUTPUT_DIM;
            (input, label)
        })
        .collect()
}

fn main() {
    println!("=== Model A/B Testing Example ===\n");

    let seed = 42;

    // =========================================================================
    // Section 1: Experiment Setup
    // =========================================================================
    println!("1. Experiment Setup");
    println!("   ─────────────────────────────────────────");

    let model_a = Model::new("v1.0", seed);
    let model_b = Model::new("v1.1", seed + 100);

    println!("   Model A: {} (baseline)", model_a.version);
    println!("   Model B: {} (candidate)", model_b.version);
    println!("   Traffic split: 70% A / 30% B");
    println!("   Input dim: {INPUT_DIM}, Output dim: {OUTPUT_DIM}");
    println!();

    // =========================================================================
    // Section 2: Traffic Routing
    // =========================================================================
    println!("2. Traffic Routing Validation");
    println!("   ─────────────────────────────────────────");

    let router = Router::new(0.30);
    let mut a_count = 0;
    let mut b_count = 0;
    for i in 0..1000 {
        match router.assign(i) {
            Variant::A => a_count += 1,
            Variant::B => b_count += 1,
        }
    }
    println!("   1000 requests: A={}, B={}", a_count, b_count);
    println!(
        "   Actual split: {:.1}% A / {:.1}% B",
        f64::from(a_count) / 10.0,
        f64::from(b_count) / 10.0
    );
    println!();

    // =========================================================================
    // Section 3: Run Experiment
    // =========================================================================
    println!("3. Running Experiment");
    println!("   ─────────────────────────────────────────");

    let mut experiment = Experiment::new(
        Model::new("v1.0", seed),
        Model::new("v1.1", seed + 100),
        0.30,
    );

    let test_data = generate_data(500, seed);
    for (i, (input, label)) in test_data.iter().enumerate() {
        experiment.process_request(i as u64, input, *label);
    }

    println!("   Processed {} requests", test_data.len());
    println!(
        "   Variant A: {} requests",
        experiment.metrics_a.predictions_total
    );
    println!(
        "   Variant B: {} requests",
        experiment.metrics_b.predictions_total
    );
    println!();

    // =========================================================================
    // Section 4: Metrics Comparison
    // =========================================================================
    println!("4. Metrics Comparison");
    println!("   ─────────────────────────────────────────");

    println!(
        "   {:>15} {:>12} {:>12}",
        "Metric", "A (baseline)", "B (candidate)"
    );
    println!("   {}", "─".repeat(42));
    println!(
        "   {:>15} {:>11.1}% {:>11.1}%",
        "Accuracy",
        experiment.metrics_a.accuracy() * 100.0,
        experiment.metrics_b.accuracy() * 100.0
    );
    println!(
        "   {:>15} {:>11.2}% {:>11.2}%",
        "Avg Confidence",
        experiment.metrics_a.avg_confidence() * 100.0,
        experiment.metrics_b.avg_confidence() * 100.0
    );
    println!(
        "   {:>15} {:>10.1}us {:>10.1}us",
        "Avg Latency",
        experiment.metrics_a.avg_latency_us(),
        experiment.metrics_b.avg_latency_us()
    );
    println!(
        "   {:>15} {:>10}us {:>10}us",
        "P95 Latency",
        experiment.metrics_a.p95_latency_us(),
        experiment.metrics_b.p95_latency_us()
    );
    println!(
        "   {:>15} {:>12} {:>12}",
        "Sample Size",
        experiment.metrics_a.predictions_total,
        experiment.metrics_b.predictions_total
    );
    println!();

    // =========================================================================
    // Section 5: Statistical Significance
    // =========================================================================
    println!("5. Statistical Significance");
    println!("   ─────────────────────────────────────────");

    let (z_score, significant) = experiment.significance_test();
    let acc_diff = (experiment.metrics_b.accuracy() - experiment.metrics_a.accuracy()) * 100.0;

    println!("   Z-score: {:.4}", z_score);
    println!("   Significant (p<0.05): {}", significant);
    println!("   Accuracy difference: {:.2}pp", acc_diff);

    if significant {
        if acc_diff > 0.0 {
            println!("   Recommendation: PROMOTE Model B (statistically better)");
        } else {
            println!("   Recommendation: KEEP Model A (Model B is worse)");
        }
    } else {
        println!("   Recommendation: CONTINUE experiment (not enough evidence)");
    }
    println!();

    // =========================================================================
    // Section 6: Traffic Split Sweep
    // =========================================================================
    println!("6. Traffic Split Sensitivity");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>8} {:>8} {:>8} {:>10} {:>8}",
        "B%", "A_acc", "B_acc", "Z-score", "Sig?"
    );
    println!("   {}", "─".repeat(46));

    for b_pct in [10, 20, 30, 50] {
        let mut exp = Experiment::new(
            Model::new("v1.0", seed),
            Model::new("v1.1", seed + 100),
            b_pct as f32 / 100.0,
        );
        for (i, (input, label)) in test_data.iter().enumerate() {
            exp.process_request(i as u64, input, *label);
        }
        let (z, sig) = exp.significance_test();
        println!(
            "   {:>7}% {:>7.1}% {:>7.1}% {:>10.4} {:>8}",
            b_pct,
            exp.metrics_a.accuracy() * 100.0,
            exp.metrics_b.accuracy() * 100.0,
            z,
            if sig { "yes" } else { "no" }
        );
    }
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_predict_probabilities_sum() {
        let model = Model::new("test", 42);
        let input = vec![0.5; INPUT_DIM];
        let probs = model.predict(&input);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_model_deterministic() {
        let model = Model::new("test", 42);
        let input = vec![0.3; INPUT_DIM];
        assert_eq!(model.predict(&input), model.predict(&input));
    }

    #[test]
    fn test_router_deterministic() {
        let router = Router::new(0.5);
        let v1 = router.assign(123);
        let v2 = router.assign(123);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_router_split_approximate() {
        let router = Router::new(0.30);
        let b_count = (0..10000)
            .filter(|&i| router.assign(i) == Variant::B)
            .count();
        let b_pct = b_count as f64 / 100.0;
        assert!(
            (b_pct - 30.0).abs() < 5.0,
            "Expected ~30% B traffic, got {:.1}%",
            b_pct
        );
    }

    #[test]
    fn test_variant_metrics_accuracy() {
        let mut m = VariantMetrics::new();
        m.record(100, true, 0.9);
        m.record(100, true, 0.8);
        m.record(100, false, 0.5);
        assert!((m.accuracy() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_variant_metrics_p95() {
        let mut m = VariantMetrics::new();
        for i in 0..100 {
            m.record(i, true, 0.9);
        }
        let p95 = m.p95_latency_us();
        assert!(p95 >= 90, "P95 should be >= 90, got {}", p95);
    }

    #[test]
    fn test_experiment_processes_requests() {
        let mut exp = Experiment::new(Model::new("a", 42), Model::new("b", 43), 0.5);
        let data = generate_data(100, 42);
        for (i, (input, label)) in data.iter().enumerate() {
            exp.process_request(i as u64, input, *label);
        }
        assert_eq!(
            exp.metrics_a.predictions_total + exp.metrics_b.predictions_total,
            100
        );
    }

    #[test]
    fn test_significance_test_needs_samples() {
        let exp = Experiment::new(Model::new("a", 42), Model::new("b", 43), 0.5);
        let (_, sig) = exp.significance_test();
        assert!(!sig, "Should not be significant with no samples");
    }
}
