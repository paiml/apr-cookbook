#![allow(unused_imports)]
//! Continuous Online Training (Defect Prediction)
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Incremental SGD and Passive-Aggressive algorithms for predicting software
//! defects from execution traces, with concept drift detection.
//!
//! ```bash
//! cargo run --example online_training_defect
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use std::collections::VecDeque;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
