#![allow(unused_imports)]
//! Model Canary Deployment Example
//! **CLI Equivalent**: `apr showcase`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Demonstrates progressive canary deployment for ML model serving with
//! gradual traffic shifting, health checks at each stage, and automatic
//! rollback when metrics degrade beyond configured thresholds.
//!
//! ```text
//! Canary Pipeline:
//!
//!   [Old Model v1.0] ←── 99% ──┐
//!                               ├── [Router] ← Requests
//!   [New Model v2.0] ←──  1% ──┘
//!         │
//!         ▼
//!   [Health Check] → latency / error_rate / accuracy
//!         │
//!     ┌───┴───┐
//!     │ Pass? │──yes──→ Promote (1% → 5% → 25% → 50% → 100%)
//!     └───────┘
//!         │no
//!         ▼
//!     [Rollback] → 100% old model
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example model_canary_deploy
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr serve model.apr          # APR native format
//! apr serve model.gguf         # GGUF (llama.cpp compatible)
//! apr serve model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Crankshaw, D. et al. (2017). *Clipper: A Low-Latency Online Prediction Serving System*. NSDI. arXiv:1612.03079

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Model Canary Deployment Example ===\n");
    let seed = 42_u64;

    // =========================================================================
    println!("1. Model Setup");
    println!("   ─────────────────────────────────────────");
    let old_model = Model::new("v1.0-stable", seed);
    let new_model = Model::new("v2.0-candidate", seed + 200);
    println!("   Old model: {} (production baseline)", old_model.version);
    println!("   New model: {} (canary candidate)", new_model.version);
    let test_input = [0.5_f32; INPUT_DIM];
    let diff: f32 = old_model
        .predict(&test_input)
        .iter()
        .zip(new_model.predict(&test_input).iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    println!("   Output divergence: {diff:.4} (confirms distinct models)\n");

    // =========================================================================
    println!("2. Canary Stage Definition");
    println!("   ─────────────────────────────────────────");
    let criteria = PromotionCriteria::default_strict();
    println!(
        "   Max error rate: {:.0}% | Max latency: {:.0}us | Min accuracy: {:.0}%",
        criteria.max_error_rate * 100.0,
        criteria.max_latency_us,
        criteria.min_accuracy * 100.0
    );
    for (i, &pct) in CANARY_STAGES.iter().enumerate() {
        println!(
            "     Stage {}: {:>3}% new / {:>3}% old",
            i + 1,
            pct,
            100 - pct
        );
    }
    println!();

    // =========================================================================
    println!("3. Health Check Evaluation");
    println!("   ─────────────────────────────────────────");
    let mut good = StageMetrics::new();
    for i in 0..100_u64 {
        good.record(150 + i, i >= 98, i % 4 != 0);
    }
    let gh = HealthCheckResult::evaluate(&good, &criteria);
    println!(
        "   Healthy:   err={:.1}% lat={:.0}us acc={:.0}% -> {}",
        gh.error_rate * 100.0,
        gh.avg_latency_us,
        gh.accuracy * 100.0,
        gh.passed
    );

    let mut bad = StageMetrics::new();
    for i in 0..100_u64 {
        bad.record(800 + i * 5, i >= 85, i % 10 == 0);
    }
    let bh = HealthCheckResult::evaluate(&bad, &criteria);
    println!(
        "   Unhealthy: err={:.1}% lat={:.0}us acc={:.0}% -> {}",
        bh.error_rate * 100.0,
        bh.avg_latency_us,
        bh.accuracy * 100.0,
        bh.passed
    );
    for v in &bh.violations {
        println!("     Violation: {v}");
    }
    println!();

    // =========================================================================
    println!("4. Progressive Rollout Simulation");
    println!("   ─────────────────────────────────────────");
    let test_data = generate_test_data(2000, seed);
    let mut deployment = CanaryDeployment::new(
        Model::new("v1.0-stable", seed),
        Model::new("v2.0-candidate", seed + 200),
        criteria,
    );
    run_deployment(&mut deployment, &test_data, seed);
    println!();
    match &deployment.outcome {
        DeploymentOutcome::Promoted => println!(
            "   PROMOTED - {} now serving 100% traffic",
            deployment.new_model.version
        ),
        DeploymentOutcome::RolledBack { stage, reason } => {
            println!("   ROLLED BACK at stage {} - {reason}", stage + 1);
        }
        DeploymentOutcome::InProgress => println!("   In progress"),
    }
    println!();

    // =========================================================================
    println!("5. Automatic Rollback Demo");
    println!("   ─────────────────────────────────────────");
    let rollback_criteria = PromotionCriteria {
        max_error_rate: 0.10,
        max_latency_us: 300.0,
        min_accuracy: 0.30,
    };
    let mut rollback_deploy = CanaryDeployment::new(
        Model::new("v1.0-stable", seed),
        Model::new("v3.0-broken", seed + 9999),
        rollback_criteria,
    );
    let rollback_data = generate_test_data(3000, seed + 777);
    run_deployment(&mut rollback_deploy, &rollback_data, seed);
    println!();
    match &rollback_deploy.outcome {
        DeploymentOutcome::RolledBack { stage, reason } => {
            println!("   ROLLED BACK at stage {}", stage + 1);
            println!("   Reason: {reason}");
            println!(
                "   Reverted to {} serving 100%",
                rollback_deploy.old_model.version
            );
        }
        other => println!("   Outcome: {other:?}"),
    }
    println!();

    // =========================================================================
    println!("6. Deployment Summary");
    println!("   ─────────────────────────────────────────");
    for (label, hist) in [
        ("Promotion", &deployment.health_history),
        ("Rollback", &rollback_deploy.health_history),
    ] {
        println!("   {label} ({} stages):", hist.len());
        for (i, h) in hist.iter().enumerate() {
            println!(
                "     Stage {}: err={:.2}% lat={:.0}us acc={:.1}% [{}]",
                i + 1,
                h.error_rate * 100.0,
                h.avg_latency_us,
                h.accuracy * 100.0,
                if h.passed { "OK" } else { "FAIL" }
            );
        }
    }
    let total: usize = deployment
        .health_history
        .iter()
        .chain(rollback_deploy.health_history.iter())
        .map(|h| h.sample_count)
        .sum();
    println!("   Total canary samples: {total}\n");
    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_predict_output_dim() {
        let model = Model::new("test", 42);
        assert_eq!(model.predict(&[0.5; INPUT_DIM]).len(), OUTPUT_DIM);
    }

    #[test]
    fn test_model_predict_probabilities_sum_to_one() {
        let probs = Model::new("test", 42).predict(&[0.3; INPUT_DIM]);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_model_deterministic() {
        let m = Model::new("test", 42);
        let input = [0.1_f32; INPUT_DIM];
        assert_eq!(m.predict(&input), m.predict(&input));
    }

    #[test]
    fn test_different_seeds_different_outputs() {
        let input = [0.5_f32; INPUT_DIM];
        assert_ne!(
            Model::new("a", 42).predict(&input),
            Model::new("b", 99).predict(&input)
        );
    }

    #[test]
    fn test_canary_routing_deterministic() {
        let stage = CanaryStage::new(50, PromotionCriteria::default_strict());
        assert_eq!(stage.route_to_new(123), stage.route_to_new(123));
    }

    #[test]
    fn test_canary_routing_distribution() {
        let stage = CanaryStage::new(30, PromotionCriteria::default_strict());
        let n = (0..10_000_u64).filter(|&id| stage.route_to_new(id)).count();
        let pct = n as f64 / 100.0;
        assert!(
            (pct - 30.0).abs() < 5.0,
            "Expected ~30% canary, got {pct:.1}%"
        );
    }

    #[test]
    fn test_stage_metrics_error_rate() {
        let mut m = StageMetrics::new();
        m.record(100, true, true);
        m.record(100, false, true);
        m.record(100, false, true);
        m.record(100, true, true);
        assert!((m.error_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_stage_metrics_accuracy() {
        let mut m = StageMetrics::new();
        m.record(100, false, true);
        m.record(100, false, true);
        m.record(100, false, false);
        assert!((m.accuracy() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_stage_metrics_empty() {
        let m = StageMetrics::new();
        assert!(m.error_rate().abs() < 1e-6);
        assert!(m.avg_latency_us().abs() < 1e-6);
        assert!(m.accuracy().abs() < 1e-6);
    }

    #[test]
    fn test_health_check_passes_good_metrics() {
        let mut m = StageMetrics::new();
        for i in 0..100_u64 {
            m.record(100 + i, false, i % 3 != 0);
        }
        let h = HealthCheckResult::evaluate(&m, &PromotionCriteria::default_strict());
        assert!(h.passed);
        assert!(h.violations.is_empty());
    }

    #[test]
    fn test_health_check_fails_high_error_rate() {
        let mut m = StageMetrics::new();
        for _ in 0..100 {
            m.record(100, true, true);
        }
        let h = HealthCheckResult::evaluate(&m, &PromotionCriteria::default_strict());
        assert!(!h.passed);
        assert!(h.violations.iter().any(|v| v.contains("error_rate")));
    }

    #[test]
    fn test_health_check_fails_low_accuracy() {
        let c = PromotionCriteria {
            max_error_rate: 1.0,
            max_latency_us: 10000.0,
            min_accuracy: 0.80,
        };
        let mut m = StageMetrics::new();
        for _ in 0..100 {
            m.record(100, false, false);
        }
        let h = HealthCheckResult::evaluate(&m, &c);
        assert!(!h.passed);
        assert!(h.violations.iter().any(|v| v.contains("accuracy")));
    }

    #[test]
    fn test_deployment_promotes_healthy_model() {
        let c = PromotionCriteria {
            max_error_rate: 0.50,
            max_latency_us: 5000.0,
            min_accuracy: 0.05,
        };
        let data = generate_test_data(5000, 42);
        let mut deploy = CanaryDeployment::new(Model::new("old", 42), Model::new("new", 43), c);
        let mut ctr = 0_u64;
        while !deploy.is_complete() {
            for _ in 0..MIN_REQUESTS_PER_STAGE * 4 {
                let (input, label) = &data[ctr as usize % data.len()];
                deploy.process_request(ctr, input, *label, 42);
                ctr += 1;
            }
            deploy.evaluate_stage();
        }
        assert_eq!(deploy.outcome, DeploymentOutcome::Promoted);
        assert_eq!(deploy.health_history.len(), NUM_CANARY_STAGES);
    }

    #[test]
    fn test_deployment_rollback_on_strict_criteria() {
        let c = PromotionCriteria {
            max_error_rate: 0.001,
            max_latency_us: 1.0,
            min_accuracy: 0.99,
        };
        let data = generate_test_data(2000, 42);
        let mut deploy = CanaryDeployment::new(Model::new("old", 42), Model::new("new", 43), c);
        let mut ctr = 0_u64;
        for _ in 0..MIN_REQUESTS_PER_STAGE * 150 {
            let (input, label) = &data[ctr as usize % data.len()];
            deploy.process_request(ctr, input, *label, 42);
            ctr += 1;
        }
        deploy.evaluate_stage();
        match &deploy.outcome {
            DeploymentOutcome::RolledBack { stage, .. } => assert_eq!(*stage, 0),
            other => panic!("Expected rollback, got {other:?}"),
        }
    }

    #[test]
    fn test_generate_data_deterministic() {
        let d1 = generate_test_data(10, 42);
        let d2 = generate_test_data(10, 42);
        for (i, (a, b)) in d1.iter().zip(d2.iter()).enumerate() {
            assert_eq!(a.0, b.0, "Inputs differ at {i}");
            assert_eq!(a.1, b.1, "Labels differ at {i}");
        }
    }

    #[test]
    fn test_generate_data_labels_in_range() {
        for (i, (_, label)) in generate_test_data(100, 42).iter().enumerate() {
            assert!(*label < OUTPUT_DIM, "Label {label} at {i} out of range");
        }
    }

    #[test]
    fn test_canary_stages_monotonic() {
        assert_eq!(CANARY_STAGES.len(), NUM_CANARY_STAGES);
        for w in CANARY_STAGES.windows(2) {
            assert!(w[0] < w[1]);
        }
        assert_eq!(*CANARY_STAGES.last().unwrap(), 100);
    }
}
