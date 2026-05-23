#![allow(unused_imports)]
//! Few-Shot Fine-Tuning Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Demonstrates fine-tuning a pre-trained model with very few labeled examples
//! using prototypical networks and metric learning. Compares few-shot approaches:
//! nearest-centroid, cosine-similarity, and Mahalanobis distance.
//!
//! # Few-Shot Learning
//!
//! ```text
//! Support Set (K examples per class):
//!   Class A: [x1, x2, ..., xK] → prototype_A = mean(embeddings)
//!   Class B: [y1, y2, ..., yK] → prototype_B = mean(embeddings)
//!
//! Query: embed(q) → argmin distance(q, prototype_i)
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example few_shot_finetune
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr finetune model.apr          # APR native format
//! apr finetune model.gguf         # GGUF (llama.cpp compatible)
//! apr finetune model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Few-Shot Fine-Tuning Example ===\n");

    let seed = 42;

    // =========================================================================
    // Section 1: Support Set Construction
    // =========================================================================
    println!("1. Support Set Construction");
    println!("   ─────────────────────────────────────────");

    let shots = 5;
    let support_set = build_support_set(shots, seed);

    println!("   K-shot: {} examples per class", shots);
    println!("   Classes: {}", NUM_CLASSES);
    println!("   Total support: {} examples", support_set.len());
    println!("   Embed dim: {}", EMBED_DIM);
    println!();

    // =========================================================================
    // Section 2: Prototype Computation
    // =========================================================================
    println!("2. Prototype Computation");
    println!("   ─────────────────────────────────────────");

    let mut proto_net = PrototypicalNetwork::new(Encoder::new(seed), DistanceMetric::Euclidean);
    proto_net.fit(&support_set);

    for (class, proto) in proto_net.prototypes.iter().enumerate() {
        let norm: f32 = proto.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "   {} prototype: norm={:.4}, first3=[{:.3}, {:.3}, {:.3}]",
            CLASS_NAMES[class], norm, proto[0], proto[1], proto[2]
        );
    }
    println!();

    // =========================================================================
    // Section 3: Few-Shot Classification
    // =========================================================================
    println!("3. Few-Shot Classification");
    println!("   ─────────────────────────────────────────");

    let test_queries = 20;
    let mut correct = 0usize;
    let mut total = 0usize;

    println!(
        "   {:>6} {:>10} {:>10} {:>8}",
        "True", "Predicted", "Conf", "Correct"
    );
    println!("   {}", "─".repeat(38));

    for (class, &class_name) in CLASS_NAMES.iter().enumerate() {
        let queries = generate_class_data(class, test_queries / NUM_CLASSES, seed + 1000);
        for query in &queries {
            let (pred, conf) = proto_net.predict(query);
            let is_correct = pred == class;
            if is_correct {
                correct += 1;
            }
            total += 1;
            if total <= 10 {
                println!(
                    "   {:>6} {:>10} {:>9.2}% {:>8}",
                    class_name,
                    CLASS_NAMES[pred],
                    conf * 100.0,
                    if is_correct { "yes" } else { "no" }
                );
            }
        }
    }

    println!("   ...");
    println!(
        "   Overall accuracy: {}/{} ({:.1}%)",
        correct,
        total,
        correct as f64 / total as f64 * 100.0
    );
    println!();

    // =========================================================================
    // Section 4: Distance Metric Comparison
    // =========================================================================
    println!("4. Distance Metric Comparison");
    println!("   ─────────────────────────────────────────");

    let metrics = [
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::Manhattan,
    ];

    println!("   {:>12} {:>10} {:>12}", "Metric", "Accuracy", "Avg Conf");
    println!("   {}", "─".repeat(36));

    for metric in metrics {
        let mut net = PrototypicalNetwork::new(Encoder::new(seed), metric);
        net.fit(&support_set);

        let (hit, count) = evaluate_accuracy(&net, 10, seed + 2000);
        let avg_conf = evaluate_avg_confidence(&net, 10, seed + 2000);

        println!(
            "   {:>12} {:>9.1}% {:>11.2}%",
            metric.name(),
            hit as f64 / count as f64 * 100.0,
            avg_conf * 100.0
        );
    }
    println!();

    // =========================================================================
    // Section 5: K-Shot Sweep
    // =========================================================================
    println!("5. K-Shot Sweep (accuracy vs shots)");
    println!("   ─────────────────────────────────────────");
    println!("   {:>6} {:>10} {:>12}", "K", "Accuracy", "Time(us)");
    println!("   {}", "─".repeat(30));

    for k in [1, 2, 5, 10, 20] {
        let support = build_support_set(k, seed);

        let start = Instant::now();
        let mut net = PrototypicalNetwork::new(Encoder::new(seed), DistanceMetric::Euclidean);
        net.fit(&support);

        let (hit, count) = evaluate_accuracy(&net, 10, seed + 3000);
        let elapsed = start.elapsed().as_micros();

        println!(
            "   {:>6} {:>9.1}% {:>10}",
            k,
            hit as f64 / count as f64 * 100.0,
            elapsed
        );
    }
    println!();

    // =========================================================================
    // Section 6: Fine-Tuning with Adapter
    // =========================================================================
    println!("6. Adapter Fine-Tuning");
    println!("   ─────────────────────────────────────────");

    // Simulate fine-tuning the encoder's projection layer on support set
    let mut adapted_encoder = Encoder::new(seed);
    let adapt_lr = 0.001;
    let adapt_epochs = 10;

    for epoch in 0..adapt_epochs {
        let mut epoch_loss = 0.0f32;
        for (input, target) in &support_set {
            let emb = adapted_encoder.embed(input);
            // Contrastive-like loss: push embedding toward class center
            let target_offset = *target as f32 * 0.3;
            for (d, &e) in emb.iter().enumerate() {
                let target_val = target_offset / EMBED_DIM as f32;
                let grad = e - target_val;
                epoch_loss += grad.abs();
                // Update projection layer
                for i in 0..EMBED_DIM {
                    adapted_encoder.w2[d * EMBED_DIM + i] -= adapt_lr * grad * 0.001;
                }
            }
        }
        if epoch == 0 || epoch == adapt_epochs - 1 {
            println!(
                "   Epoch {}: avg_loss={:.4}",
                epoch,
                epoch_loss / support_set.len() as f32
            );
        }
    }

    // Compare before/after adaptation
    let original_net = {
        let mut n = PrototypicalNetwork::new(Encoder::new(seed), DistanceMetric::Euclidean);
        n.fit(&support_set);
        n
    };
    let adapted_net = {
        let mut n = PrototypicalNetwork::new(adapted_encoder, DistanceMetric::Euclidean);
        n.fit(&support_set);
        n
    };

    let (before_hit, before_count) = evaluate_accuracy(&original_net, 10, seed + 5000);
    let (after_hit, after_count) = evaluate_accuracy(&adapted_net, 10, seed + 5000);

    println!(
        "   Before adaptation: {:.1}%",
        before_hit as f64 / before_count as f64 * 100.0
    );
    println!(
        "   After adaptation:  {:.1}%",
        after_hit as f64 / after_count as f64 * 100.0
    );
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_output_normalized() {
        let enc = Encoder::new(42);
        let input = vec![0.5; EMBED_DIM];
        let emb = enc.embed(&input);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Embedding should be L2 normalized"
        );
    }

    #[test]
    fn test_encoder_deterministic() {
        let enc = Encoder::new(42);
        let input = vec![0.3; EMBED_DIM];
        let e1 = enc.embed(&input);
        let e2 = enc.embed(&input);
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_euclidean_distance_zero_for_same() {
        let a = vec![1.0, 2.0, 3.0];
        let d = DistanceMetric::Euclidean.distance(&a, &a);
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_zero_for_same() {
        let a = vec![1.0, 2.0, 3.0];
        let d = DistanceMetric::Cosine.distance(&a, &a);
        assert!(d.abs() < 1e-5);
    }

    #[test]
    fn test_prototypical_network_fit() {
        let support = vec![
            (vec![1.0; EMBED_DIM], 0),
            (vec![2.0; EMBED_DIM], 0),
            (vec![-1.0; EMBED_DIM], 1),
        ];
        let mut net = PrototypicalNetwork::new(Encoder::new(42), DistanceMetric::Euclidean);
        net.fit(&support);
        assert_eq!(net.prototypes.len(), NUM_CLASSES);
    }

    #[test]
    fn test_prototypical_network_predict() {
        let support = vec![(vec![1.0; EMBED_DIM], 0), (vec![-1.0; EMBED_DIM], 1)];
        let mut net = PrototypicalNetwork::new(Encoder::new(42), DistanceMetric::Euclidean);
        net.fit(&support);
        let (class, conf) = net.predict(&[1.0; EMBED_DIM]);
        assert!(class < NUM_CLASSES);
        assert!(conf > 0.0);
    }

    #[test]
    fn test_generate_class_data_dimensions() {
        let data = generate_class_data(0, 5, 42);
        assert_eq!(data.len(), 5);
        assert_eq!(data[0].len(), EMBED_DIM);
    }

    #[test]
    fn test_different_classes_different_data() {
        let d0 = generate_class_data(0, 1, 42);
        let d1 = generate_class_data(1, 1, 42);
        assert_ne!(d0, d1);
    }
}
