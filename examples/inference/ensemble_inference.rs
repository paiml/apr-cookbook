//! Ensemble Inference Example
//!
//! Demonstrates combining predictions from multiple models using different
//! ensemble strategies: majority voting, probability averaging, and weighted
//! confidence-based combining.
//!
//! # Ensemble Strategies
//!
//! ```text
//! Voting:    argmax(count(predictions))
//! Averaging: softmax(mean(logits))
//! Weighted:  sum(confidence_i * prediction_i)
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example ensemble_inference
//! ```

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

const INPUT_DIM: usize = 16;
const HIDDEN_DIM: usize = 8;
const OUTPUT_DIM: usize = 4;
const CLASS_NAMES: [&str; OUTPUT_DIM] = ["cat", "dog", "bird", "fish"];

/// Deterministic weight init
fn init_weights(size: usize, seed: u64) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let mut h = DefaultHasher::new();
            (seed, i).hash(&mut h);
            (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.2
        })
        .collect()
}

/// Simple two-layer classifier
struct Classifier {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    name: String,
}

impl Classifier {
    fn new(name: &str, seed: u64) -> Self {
        Self {
            w1: init_weights(HIDDEN_DIM * INPUT_DIM, seed),
            b1: vec![0.0; HIDDEN_DIM],
            w2: init_weights(OUTPUT_DIM * HIDDEN_DIM, seed + 1),
            b2: vec![0.0; OUTPUT_DIM],
            name: name.to_string(),
        }
    }

    /// Forward pass returning logits
    fn logits(&self, input: &[f32]) -> Vec<f32> {
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
    fn predict(&self, input: &[f32]) -> Vec<f32> {
        softmax(&self.logits(input))
    }
}

/// Softmax normalization
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Argmax helper
fn argmax(probs: &[f32]) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map_or(0, |(i, _)| i)
}

/// Ensemble strategy
#[derive(Clone, Copy)]
enum Strategy {
    MajorityVote,
    ProbabilityAverage,
    WeightedConfidence,
}

impl Strategy {
    fn name(self) -> &'static str {
        match self {
            Strategy::MajorityVote => "Majority Vote",
            Strategy::ProbabilityAverage => "Prob Average",
            Strategy::WeightedConfidence => "Weighted Conf",
        }
    }
}

/// Ensemble of classifiers
struct Ensemble {
    models: Vec<Classifier>,
}

impl Ensemble {
    fn new(models: Vec<Classifier>) -> Self {
        Self { models }
    }

    /// Predict using majority voting
    fn vote(&self, input: &[f32]) -> (usize, f32) {
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
    fn average(&self, input: &[f32]) -> (usize, f32) {
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
    fn weighted(&self, input: &[f32]) -> (usize, f32) {
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
    fn predict(&self, input: &[f32], strategy: Strategy) -> (usize, f32) {
        match strategy {
            Strategy::MajorityVote => self.vote(input),
            Strategy::ProbabilityAverage => self.average(input),
            Strategy::WeightedConfidence => self.weighted(input),
        }
    }
}

/// Generate labeled data
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

/// Evaluate accuracy of a prediction function
fn evaluate(data: &[(Vec<f32>, usize)], predict_fn: &dyn Fn(&[f32]) -> usize) -> f64 {
    let correct = data
        .iter()
        .filter(|(input, label)| predict_fn(input) == *label)
        .count();
    correct as f64 / data.len() as f64
}

fn main() {
    println!("=== Ensemble Inference Example ===\n");

    let seed = 42;
    let test_data = generate_data(200, seed);

    // =========================================================================
    // Section 1: Individual Model Performance
    // =========================================================================
    println!("1. Individual Model Performance");
    println!("   ─────────────────────────────────────────");

    let models: Vec<Classifier> = (0..5)
        .map(|i| Classifier::new(&format!("model_{i}"), seed + i * 100))
        .collect();

    println!("   {:>10} {:>10} {:>12}", "Model", "Accuracy", "Avg Conf");
    println!("   {}", "─".repeat(34));

    for model in &models {
        let acc = evaluate(&test_data, &|input| argmax(&model.predict(input)));
        let avg_conf: f32 = test_data
            .iter()
            .map(|(input, _)| {
                let probs = model.predict(input);
                probs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
            })
            .sum::<f32>()
            / test_data.len() as f32;
        println!(
            "   {:>10} {:>9.1}% {:>11.2}%",
            model.name,
            acc * 100.0,
            avg_conf * 100.0
        );
    }
    println!();

    // =========================================================================
    // Section 2: Ensemble Strategy Comparison
    // =========================================================================
    println!("2. Ensemble Strategy Comparison");
    println!("   ─────────────────────────────────────────");

    let ensemble = Ensemble::new(
        (0..5)
            .map(|i| Classifier::new(&format!("model_{i}"), seed + i * 100))
            .collect(),
    );

    let strategies = [
        Strategy::MajorityVote,
        Strategy::ProbabilityAverage,
        Strategy::WeightedConfidence,
    ];

    println!(
        "   {:>15} {:>10} {:>12}",
        "Strategy", "Accuracy", "Avg Conf"
    );
    println!("   {}", "─".repeat(40));

    for strategy in strategies {
        let acc = evaluate(&test_data, &|input| ensemble.predict(input, strategy).0);
        let avg_conf: f32 = test_data
            .iter()
            .map(|(input, _)| ensemble.predict(input, strategy).1)
            .sum::<f32>()
            / test_data.len() as f32;
        println!(
            "   {:>15} {:>9.1}% {:>11.2}%",
            strategy.name(),
            acc * 100.0,
            avg_conf * 100.0
        );
    }
    println!();

    // =========================================================================
    // Section 3: Ensemble Size Sweep
    // =========================================================================
    println!("3. Ensemble Size Sweep");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>6} {:>10} {:>10} {:>10}",
        "Size", "Vote", "Average", "Weighted"
    );
    println!("   {}", "─".repeat(40));

    for size in [1, 3, 5, 7, 9] {
        let ens = Ensemble::new(
            (0..size)
                .map(|i| Classifier::new(&format!("m{i}"), seed + i * 100))
                .collect(),
        );

        let vote_acc = evaluate(&test_data, &|input| {
            ens.predict(input, Strategy::MajorityVote).0
        });
        let avg_acc = evaluate(&test_data, &|input| {
            ens.predict(input, Strategy::ProbabilityAverage).0
        });
        let wt_acc = evaluate(&test_data, &|input| {
            ens.predict(input, Strategy::WeightedConfidence).0
        });

        println!(
            "   {:>6} {:>9.1}% {:>9.1}% {:>9.1}%",
            size,
            vote_acc * 100.0,
            avg_acc * 100.0,
            wt_acc * 100.0
        );
    }
    println!();

    // =========================================================================
    // Section 4: Per-Class Analysis
    // =========================================================================
    println!("4. Per-Class Ensemble Analysis");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>6} {:>10} {:>10} {:>10}",
        "Class", "Individual", "Ensemble", "Lift"
    );
    println!("   {}", "─".repeat(40));

    let best_single = &models[0];
    for (class, &class_name) in CLASS_NAMES.iter().enumerate() {
        let class_data: Vec<_> = test_data
            .iter()
            .filter(|(_, l)| *l == class)
            .cloned()
            .collect();

        if class_data.is_empty() {
            continue;
        }

        let single_acc = evaluate(&class_data, &|input| argmax(&best_single.predict(input)));
        let ens_acc = evaluate(&class_data, &|input| {
            ensemble.predict(input, Strategy::ProbabilityAverage).0
        });
        let lift = ens_acc - single_acc;

        println!(
            "   {:>6} {:>9.1}% {:>9.1}% {:>+9.1}pp",
            class_name,
            single_acc * 100.0,
            ens_acc * 100.0,
            lift * 100.0
        );
    }
    println!();

    // =========================================================================
    // Section 5: Throughput
    // =========================================================================
    println!("5. Throughput Benchmark");
    println!("   ─────────────────────────────────────────");

    let n_iters: u32 = 1000;
    let input = &test_data[0].0;

    println!(
        "   {:>15} {:>12} {:>14}",
        "Method", "Time(us)", "Samples/sec"
    );
    println!("   {}", "─".repeat(44));

    // Single model
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = models[0].predict(input);
    }
    let single_time = start.elapsed().as_micros();
    println!(
        "   {:>15} {:>12} {:>14.0}",
        "Single model",
        single_time,
        f64::from(n_iters) / (single_time as f64 / 1_000_000.0)
    );

    // Ensemble (5 models)
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = ensemble.predict(input, Strategy::ProbabilityAverage);
    }
    let ens_time = start.elapsed().as_micros();
    println!(
        "   {:>15} {:>12} {:>14.0}",
        "Ensemble (5)",
        ens_time,
        f64::from(n_iters) / (ens_time as f64 / 1_000_000.0)
    );

    println!(
        "   Overhead: {:.1}x (expected ~{}.0x)",
        ens_time as f64 / single_time.max(1) as f64,
        ensemble.models.len()
    );
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmax_basic() {
        assert_eq!(argmax(&[0.1, 0.5, 0.3, 0.1]), 1);
        assert_eq!(argmax(&[0.9, 0.0, 0.0, 0.0]), 0);
    }

    #[test]
    fn test_classifier_output_size() {
        let model = Classifier::new("test", 42);
        let input = vec![0.5; INPUT_DIM];
        let probs = model.predict(&input);
        assert_eq!(probs.len(), OUTPUT_DIM);
    }

    #[test]
    fn test_classifier_deterministic() {
        let model = Classifier::new("test", 42);
        let input = vec![0.3; INPUT_DIM];
        assert_eq!(model.predict(&input), model.predict(&input));
    }

    #[test]
    fn test_ensemble_vote() {
        let ens = Ensemble::new(vec![
            Classifier::new("a", 42),
            Classifier::new("b", 43),
            Classifier::new("c", 44),
        ]);
        let input = vec![0.5; INPUT_DIM];
        let (class, conf) = ens.vote(&input);
        assert!(class < OUTPUT_DIM);
        assert!(conf > 0.0 && conf <= 1.0);
    }

    #[test]
    fn test_ensemble_average() {
        let ens = Ensemble::new(vec![Classifier::new("a", 42), Classifier::new("b", 43)]);
        let input = vec![0.5; INPUT_DIM];
        let (class, conf) = ens.average(&input);
        assert!(class < OUTPUT_DIM);
        assert!(conf > 0.0);
    }

    #[test]
    fn test_ensemble_weighted() {
        let ens = Ensemble::new(vec![Classifier::new("a", 42), Classifier::new("b", 43)]);
        let input = vec![0.5; INPUT_DIM];
        let (class, conf) = ens.weighted(&input);
        assert!(class < OUTPUT_DIM);
        assert!(conf > 0.0);
    }

    #[test]
    fn test_evaluate_function() {
        let data = generate_data(10, 42);
        let acc = evaluate(&data, &|_| 0);
        assert!(acc >= 0.0 && acc <= 1.0);
    }
}
