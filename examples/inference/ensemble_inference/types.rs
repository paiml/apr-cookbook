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
use std::time::Instant;

pub const INPUT_DIM: usize = 16;
pub const HIDDEN_DIM: usize = 8;
pub const OUTPUT_DIM: usize = 4;
pub const CLASS_NAMES: [&str; OUTPUT_DIM] = ["cat", "dog", "bird", "fish"];

/// Deterministic weight init
pub fn init_weights(size: usize, seed: u64) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let mut h = DefaultHasher::new();
            (seed, i).hash(&mut h);
            (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.2
        })
        .collect()
}

/// Simple two-layer classifier
pub struct Classifier {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub name: String,
}

impl Classifier {
    pub fn new(name: &str, seed: u64) -> Self {
        Self {
            w1: init_weights(HIDDEN_DIM * INPUT_DIM, seed),
            b1: vec![0.0; HIDDEN_DIM],
            w2: init_weights(OUTPUT_DIM * HIDDEN_DIM, seed + 1),
            b2: vec![0.0; OUTPUT_DIM],
            name: name.to_string(),
        }
    }

    /// Forward pass returning logits
    pub fn logits(&self, input: &[f32]) -> Vec<f32> {
        let mut hidden = self.b1.clone();
        for (o, h) in hidden.iter_mut().enumerate() {
            for (i, &x) in input.iter().enumerate() {
                *h += self.w1[o * INPUT_DIM + i] * x;
            }
            *h = h.max(0.0);
        }

        let mut output = self.b2.clone();
        for (o, out) in output.iter_mut().enumerate() {
            for (i, &h) in hidden.iter().enumerate() {
                *out += self.w2[o * HIDDEN_DIM + i] * h;
            }
        }
        output
    }

    /// Forward pass returning probabilities
    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        softmax(&self.logits(input))
    }
}

/// Softmax normalization
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Argmax helper
pub fn argmax(probs: &[f32]) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map_or(0, |(i, _)| i)
}

/// Ensemble strategy
#[derive(Clone, Copy)]
pub enum Strategy {
    MajorityVote,
    ProbabilityAverage,
    WeightedConfidence,
}

impl Strategy {
    pub fn name(self) -> &'static str {
        match self {
            Strategy::MajorityVote => "Majority Vote",
            Strategy::ProbabilityAverage => "Prob Average",
            Strategy::WeightedConfidence => "Weighted Conf",
        }
    }
}

/// Ensemble of classifiers
pub struct Ensemble {
    pub models: Vec<Classifier>,
}

impl Ensemble {
    pub fn new(models: Vec<Classifier>) -> Self {
        Self { models }
    }

    /// Predict using majority voting
    pub fn vote(&self, input: &[f32]) -> (usize, f32) {
        let mut votes = [0usize; OUTPUT_DIM];
        for model in &self.models {
            let pred = argmax(&model.predict(input));
            votes[pred] += 1;
        }
        let winner = argmax(&votes.iter().map(|&v| v as f32).collect::<Vec<_>>());
        let confidence = votes[winner] as f32 / self.models.len() as f32;
        (winner, confidence)
    }

    /// Predict using probability averaging
    pub fn average(&self, input: &[f32]) -> (usize, f32) {
        let n = self.models.len() as f32;
        let mut avg_probs = vec![0.0f32; OUTPUT_DIM];
        for model in &self.models {
            let probs = model.predict(input);
            for (avg, &p) in avg_probs.iter_mut().zip(probs.iter()) {
                *avg += p / n;
            }
        }
        let winner = argmax(&avg_probs);
        (winner, avg_probs[winner])
    }

    /// Predict using confidence-weighted combining
    pub fn weighted(&self, input: &[f32]) -> (usize, f32) {
        let mut weighted_probs = vec![0.0f32; OUTPUT_DIM];
        let mut total_weight = 0.0f32;

        for model in &self.models {
            let probs = model.predict(input);
            let confidence = probs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            total_weight += confidence;
            for (wp, &p) in weighted_probs.iter_mut().zip(probs.iter()) {
                *wp += p * confidence;
            }
        }

        for wp in &mut weighted_probs {
            *wp /= total_weight.max(1e-8);
        }

        let winner = argmax(&weighted_probs);
        (winner, weighted_probs[winner])
    }

    /// Predict using a given strategy
    pub fn predict(&self, input: &[f32], strategy: Strategy) -> (usize, f32) {
        match strategy {
            Strategy::MajorityVote => self.vote(input),
            Strategy::ProbabilityAverage => self.average(input),
            Strategy::WeightedConfidence => self.weighted(input),
        }
    }
}

/// Generate labeled data
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

/// Evaluate accuracy of a prediction function
pub fn evaluate(data: &[(Vec<f32>, usize)], predict_fn: &dyn Fn(&[f32]) -> usize) -> f64 {
    let correct = data
        .iter()
        .filter(|(input, label)| predict_fn(input) == *label)
        .count();
    correct as f64 / data.len() as f64
}
