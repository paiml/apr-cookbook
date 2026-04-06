#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub const INPUT_DIM: usize = 16;
pub const OUTPUT_DIM: usize = 4;

/// Deterministic weight initialization
pub fn init_weights(size: usize, seed: u64) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let mut h = DefaultHasher::new();
            (seed, i).hash(&mut h);
            (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.2
        })
        .collect()
}

/// Simple inference model
pub struct Model {
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub version: String,
}

impl Model {
    pub fn new(version: &str, seed: u64) -> Self {
        Self {
            weights: init_weights(OUTPUT_DIM * INPUT_DIM, seed),
            bias: init_weights(OUTPUT_DIM, seed + 1),
            version: version.to_string(),
        }
    }

    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
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
pub enum Variant {
    A,
    B,
}

// Variant names are used via Display-style formatting in main output

/// Traffic router with deterministic splitting
pub struct Router {
    pub b_traffic_pct: f32,
}

impl Router {
    pub fn new(b_traffic_pct: f32) -> Self {
        Self { b_traffic_pct }
    }

    pub fn assign(&self, request_id: u64) -> Variant {
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
pub struct VariantMetrics {
    pub latencies_us: Vec<u64>,
    pub predictions_correct: usize,
    pub predictions_total: usize,
    pub confidence_sum: f32,
}

impl VariantMetrics {
    pub fn new() -> Self {
        Self {
            latencies_us: Vec::new(),
            predictions_correct: 0,
            predictions_total: 0,
            confidence_sum: 0.0,
        }
    }

    pub fn record(&mut self, latency_us: u64, correct: bool, confidence: f32) {
        self.latencies_us.push(latency_us);
        self.predictions_total += 1;
        if correct {
            self.predictions_correct += 1;
        }
        self.confidence_sum += confidence;
    }

    pub fn accuracy(&self) -> f64 {
        if self.predictions_total == 0 {
            return 0.0;
        }
        self.predictions_correct as f64 / self.predictions_total as f64
    }

    pub fn avg_confidence(&self) -> f32 {
        if self.predictions_total == 0 {
            return 0.0;
        }
        self.confidence_sum / self.predictions_total as f32
    }

    pub fn avg_latency_us(&self) -> f64 {
        if self.latencies_us.is_empty() {
            return 0.0;
        }
        self.latencies_us.iter().sum::<u64>() as f64 / self.latencies_us.len() as f64
    }

    pub fn p95_latency_us(&self) -> u64 {
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
pub struct Experiment {
    pub model_a: Model,
    pub model_b: Model,
    pub router: Router,
    pub metrics_a: VariantMetrics,
    pub metrics_b: VariantMetrics,
}

impl Experiment {
    pub fn new(model_a: Model, model_b: Model, b_traffic_pct: f32) -> Self {
        Self {
            model_a,
            model_b,
            router: Router::new(b_traffic_pct),
            metrics_a: VariantMetrics::new(),
            metrics_b: VariantMetrics::new(),
        }
    }

    pub fn process_request(&mut self, request_id: u64, input: &[f32], true_label: usize) {
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
    pub fn significance_test(&self) -> (f64, bool) {
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
pub fn generate_data(n: usize, seed: u64) -> Vec<(Vec<f32>, usize)> {
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
