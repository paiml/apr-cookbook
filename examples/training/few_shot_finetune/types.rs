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
use std::hash::{Hash, Hasher};
use std::time::Instant;

pub const EMBED_DIM: usize = 32;
pub const NUM_CLASSES: usize = 5;
pub const CLASS_NAMES: [&str; NUM_CLASSES] = ["cat", "dog", "bird", "fish", "frog"];

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

/// Simple embedding model (pre-trained encoder)
pub struct Encoder {
    pub w1: Vec<f32>, // EMBED_DIM x EMBED_DIM
    pub b1: Vec<f32>,
    pub w2: Vec<f32>, // EMBED_DIM x EMBED_DIM
    pub b2: Vec<f32>,
}

impl Encoder {
    pub fn new(seed: u64) -> Self {
        Self {
            w1: init_weights(EMBED_DIM * EMBED_DIM, seed),
            b1: init_weights(EMBED_DIM, seed + 1),
            w2: init_weights(EMBED_DIM * EMBED_DIM, seed + 2),
            b2: init_weights(EMBED_DIM, seed + 3),
        }
    }

    pub fn embed(&self, input: &[f32]) -> Vec<f32> {
        // Layer 1 + ReLU
        let mut hidden = self.b1.clone();
        for (o, h) in hidden.iter_mut().enumerate() {
            for (i, &x) in input.iter().enumerate() {
                *h += self.w1[o * EMBED_DIM + i] * x;
            }
            *h = h.max(0.0);
        }

        // Layer 2 (projection)
        let mut output = self.b2.clone();
        for (o, out) in output.iter_mut().enumerate() {
            for (i, &h) in hidden.iter().enumerate() {
                *out += self.w2[o * EMBED_DIM + i] * h;
            }
        }

        // L2 normalize
        let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for x in &mut output {
            *x /= norm;
        }
        output
    }
}

/// Distance metric for prototype comparison
#[derive(Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    Manhattan,
}

impl DistanceMetric {
    pub fn name(self) -> &'static str {
        match self {
            DistanceMetric::Euclidean => "Euclidean",
            DistanceMetric::Cosine => "Cosine",
            DistanceMetric::Manhattan => "Manhattan",
        }
    }

    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
            DistanceMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                1.0 - dot / (na * nb).max(1e-8)
            }
            DistanceMetric::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
        }
    }
}

/// Prototypical network for few-shot classification
pub struct PrototypicalNetwork {
    pub encoder: Encoder,
    pub prototypes: Vec<Vec<f32>>,
    pub metric: DistanceMetric,
}

impl PrototypicalNetwork {
    pub fn new(encoder: Encoder, metric: DistanceMetric) -> Self {
        Self {
            encoder,
            prototypes: Vec::new(),
            metric,
        }
    }

    /// Compute class prototypes from support set
    pub fn fit(&mut self, support_set: &[(Vec<f32>, usize)]) {
        let mut class_embeddings: Vec<Vec<Vec<f32>>> = vec![Vec::new(); NUM_CLASSES];

        for (input, label) in support_set {
            let emb = self.encoder.embed(input);
            class_embeddings[*label].push(emb);
        }

        self.prototypes = class_embeddings
            .iter()
            .map(|embeddings| {
                if embeddings.is_empty() {
                    return vec![0.0; EMBED_DIM];
                }
                let n = embeddings.len() as f32;
                let mut centroid = vec![0.0; EMBED_DIM];
                for emb in embeddings {
                    for (c, &e) in centroid.iter_mut().zip(emb.iter()) {
                        *c += e / n;
                    }
                }
                centroid
            })
            .collect();
    }

    /// Classify a query point
    pub fn predict(&self, input: &[f32]) -> (usize, f32) {
        let emb = self.encoder.embed(input);
        let mut best_class = 0;
        let mut best_dist = f32::INFINITY;

        for (class, proto) in self.prototypes.iter().enumerate() {
            let dist = self.metric.distance(&emb, proto);
            if dist < best_dist {
                best_dist = dist;
                best_class = class;
            }
        }

        // Convert distance to confidence (inverse softmax-like)
        let confidence = (-best_dist).exp();
        (best_class, confidence)
    }
}

/// Generate synthetic data with class-conditional distributions
pub fn generate_class_data(class: usize, n: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..EMBED_DIM)
                .map(|j| {
                    let mut h = DefaultHasher::new();
                    (seed, class, i, j).hash(&mut h);
                    let base = h.finish() as f32 / u64::MAX as f32 - 0.5;
                    // Add class-dependent offset for separability
                    base + (class as f32 * 0.3)
                })
                .collect()
        })
        .collect()
}

/// Build support set with K examples per class
pub fn build_support_set(k: usize, seed: u64) -> Vec<(Vec<f32>, usize)> {
    let mut support = Vec::new();
    for class in 0..NUM_CLASSES {
        for ex in generate_class_data(class, k, seed) {
            support.push((ex, class));
        }
    }
    support
}

/// Evaluate average confidence on generated test data
pub fn evaluate_avg_confidence(net: &PrototypicalNetwork, n_per_class: usize, seed: u64) -> f32 {
    let mut total_conf = 0.0f32;
    let mut count = 0usize;
    for class in 0..NUM_CLASSES {
        for query in generate_class_data(class, n_per_class, seed) {
            total_conf += net.predict(&query).1;
            count += 1;
        }
    }
    total_conf / count.max(1) as f32
}

/// Evaluate network accuracy on generated test data
pub fn evaluate_accuracy(
    net: &PrototypicalNetwork,
    n_per_class: usize,
    seed: u64,
) -> (usize, usize) {
    let mut hit = 0usize;
    let mut count = 0usize;
    for class in 0..NUM_CLASSES {
        for query in generate_class_data(class, n_per_class, seed) {
            if net.predict(&query).0 == class {
                hit += 1;
            }
            count += 1;
        }
    }
    (hit, count)
}
