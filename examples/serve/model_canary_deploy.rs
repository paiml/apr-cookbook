//! Model Canary Deployment Example
//! **CLI Equivalent**: `apr showcase`
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

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const INPUT_DIM: usize = 16;
const OUTPUT_DIM: usize = 4;
const NUM_CANARY_STAGES: usize = 5;
const CANARY_STAGES: [u8; NUM_CANARY_STAGES] = [1, 5, 25, 50, 100];
const MIN_REQUESTS_PER_STAGE: usize = 50;

fn hash_seed(seed: u64, idx: usize) -> f32 {
    let mut h = DefaultHasher::new();
    (seed, idx).hash(&mut h);
    h.finish() as f32 / u64::MAX as f32 - 0.5
}

struct Model {
    weights: Vec<f32>,
    bias: Vec<f32>,
    version: String,
}

impl Model {
    fn new(version: &str, seed: u64) -> Self {
        Self {
            weights: (0..OUTPUT_DIM * INPUT_DIM)
                .map(|i| hash_seed(seed, i) * 0.2)
                .collect(),
            bias: (0..OUTPUT_DIM)
                .map(|i| hash_seed(seed + 1, i) * 0.2)
                .collect(),
            version: version.to_string(),
        }
    }

    fn predict(&self, input: &[f32]) -> [f32; OUTPUT_DIM] {
        let mut output = [0.0_f32; OUTPUT_DIM];
        for (o, out) in output.iter_mut().enumerate() {
            *out = self.bias[o];
            for (i, &x) in input.iter().enumerate().take(INPUT_DIM) {
                *out += self.weights[o * INPUT_DIM + i] * x;
            }
        }
        let max = output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut exps = [0.0_f32; OUTPUT_DIM];
        for (i, &val) in output.iter().enumerate() {
            exps[i] = (val - max).exp();
        }
        let sum: f32 = exps.iter().sum();
        for e in &mut exps {
            *e /= sum;
        }
        exps
    }
}

#[derive(Clone, Copy, Debug)]
struct PromotionCriteria {
    max_error_rate: f64,
    max_latency_us: f64,
    min_accuracy: f64,
}

impl PromotionCriteria {
    fn default_strict() -> Self {
        Self {
            max_error_rate: 0.10,
            max_latency_us: 500.0,
            min_accuracy: 0.15,
        }
    }
}

#[derive(Clone, Debug)]
struct HealthCheckResult {
    passed: bool,
    error_rate: f64,
    avg_latency_us: f64,
    accuracy: f64,
    sample_count: usize,
    violations: Vec<String>,
}

impl HealthCheckResult {
    fn evaluate(metrics: &StageMetrics, criteria: &PromotionCriteria) -> Self {
        let (error_rate, avg_latency_us, accuracy) = (
            metrics.error_rate(),
            metrics.avg_latency_us(),
            metrics.accuracy(),
        );
        let mut violations = Vec::new();
        if error_rate > criteria.max_error_rate {
            violations.push(format!(
                "error_rate {:.2}% > {:.2}%",
                error_rate * 100.0,
                criteria.max_error_rate * 100.0
            ));
        }
        if avg_latency_us > criteria.max_latency_us {
            violations.push(format!(
                "latency {avg_latency_us:.0}us > {:.0}us",
                criteria.max_latency_us
            ));
        }
        if accuracy < criteria.min_accuracy {
            violations.push(format!(
                "accuracy {:.2}% < {:.2}%",
                accuracy * 100.0,
                criteria.min_accuracy * 100.0
            ));
        }
        Self {
            passed: violations.is_empty(),
            error_rate,
            avg_latency_us,
            accuracy,
            sample_count: metrics.total_requests,
            violations,
        }
    }
}

struct StageMetrics {
    total_requests: usize,
    errors: usize,
    correct_predictions: usize,
    latency_sum_us: u64,
}

impl StageMetrics {
    fn new() -> Self {
        Self {
            total_requests: 0,
            errors: 0,
            correct_predictions: 0,
            latency_sum_us: 0,
        }
    }

    fn record(&mut self, latency_us: u64, is_error: bool, is_correct: bool) {
        self.total_requests += 1;
        self.latency_sum_us += latency_us;
        if is_error {
            self.errors += 1;
        }
        if is_correct {
            self.correct_predictions += 1;
        }
    }

    fn error_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.errors as f64 / self.total_requests as f64
        }
    }

    fn avg_latency_us(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.latency_sum_us as f64 / self.total_requests as f64
        }
    }

    fn accuracy(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.correct_predictions as f64 / self.total_requests as f64
        }
    }
}

struct CanaryStage {
    canary_pct: u8,
    criteria: PromotionCriteria,
    old_metrics: StageMetrics,
    new_metrics: StageMetrics,
}

impl CanaryStage {
    fn new(canary_pct: u8, criteria: PromotionCriteria) -> Self {
        Self {
            canary_pct,
            criteria,
            old_metrics: StageMetrics::new(),
            new_metrics: StageMetrics::new(),
        }
    }

    fn route_to_new(&self, request_id: u64) -> bool {
        let mut h = DefaultHasher::new();
        request_id.hash(&mut h);
        ((h.finish() % 100) as u8) < self.canary_pct
    }

    fn check_health(&self) -> HealthCheckResult {
        HealthCheckResult::evaluate(&self.new_metrics, &self.criteria)
    }

    fn has_enough_samples(&self) -> bool {
        self.new_metrics.total_requests >= MIN_REQUESTS_PER_STAGE
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum DeploymentOutcome {
    Promoted,
    RolledBack { stage: usize, reason: String },
    InProgress,
}

struct CanaryDeployment {
    old_model: Model,
    new_model: Model,
    stages: Vec<CanaryStage>,
    current_stage: usize,
    outcome: DeploymentOutcome,
    health_history: Vec<HealthCheckResult>,
}

impl CanaryDeployment {
    fn new(old_model: Model, new_model: Model, criteria: PromotionCriteria) -> Self {
        let stages = CANARY_STAGES
            .iter()
            .map(|&pct| CanaryStage::new(pct, criteria))
            .collect();
        Self {
            old_model,
            new_model,
            stages,
            current_stage: 0,
            outcome: DeploymentOutcome::InProgress,
            health_history: Vec::new(),
        }
    }

    fn current_canary_pct(&self) -> u8 {
        self.stages
            .get(self.current_stage)
            .map_or(100, |s| s.canary_pct)
    }

    fn process_request(
        &mut self,
        request_id: u64,
        input: &[f32],
        true_label: usize,
        latency_seed: u64,
    ) {
        if self.outcome != DeploymentOutcome::InProgress {
            return;
        }
        let use_new = self.stages[self.current_stage].route_to_new(request_id);

        let mut h = DefaultHasher::new();
        (latency_seed, request_id, "latency").hash(&mut h);
        let base_latency = h.finish() % 200 + 50;
        let latency_us = if use_new {
            base_latency + 10
        } else {
            base_latency
        };

        let probs = if use_new {
            self.new_model.predict(input)
        } else {
            self.old_model.predict(input)
        };
        let predicted = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);

        let mut h = DefaultHasher::new();
        (latency_seed, request_id, "error").hash(&mut h);
        let is_error = (h.finish() % 100) < 2;

        let stage = &mut self.stages[self.current_stage];
        if use_new {
            stage
                .new_metrics
                .record(latency_us, is_error, predicted == true_label);
        } else {
            stage
                .old_metrics
                .record(latency_us, is_error, predicted == true_label);
        }
    }

    fn evaluate_stage(&mut self) -> Option<HealthCheckResult> {
        if self.outcome != DeploymentOutcome::InProgress {
            return None;
        }
        if !self.stages[self.current_stage].has_enough_samples() {
            return None;
        }

        let health = self.stages[self.current_stage].check_health();
        self.health_history.push(health.clone());

        if health.passed {
            if self.current_stage + 1 < self.stages.len() {
                self.current_stage += 1;
            } else {
                self.outcome = DeploymentOutcome::Promoted;
            }
        } else {
            self.outcome = DeploymentOutcome::RolledBack {
                stage: self.current_stage,
                reason: health.violations.join("; "),
            };
        }
        Some(health)
    }

    fn is_complete(&self) -> bool {
        self.outcome != DeploymentOutcome::InProgress
    }
}

fn generate_test_data(n: usize, seed: u64) -> Vec<([f32; INPUT_DIM], usize)> {
    (0..n)
        .map(|i| {
            let mut input = [0.0_f32; INPUT_DIM];
            for (j, val) in input.iter_mut().enumerate() {
                let mut h = DefaultHasher::new();
                (seed, i, j).hash(&mut h);
                *val = h.finish() as f32 / u64::MAX as f32 - 0.5;
            }
            let mut h = DefaultHasher::new();
            (seed, "label", i).hash(&mut h);
            (input, h.finish() as usize % OUTPUT_DIM)
        })
        .collect()
}

/// Run one full deployment simulation, printing stage results
fn run_deployment(
    deploy: &mut CanaryDeployment,
    data: &[([f32; INPUT_DIM], usize)],
    latency_seed: u64,
) {
    println!(
        "   {:>6} {:>6} {:>10} {:>10} {:>10} {:>8}",
        "Stage", "New%", "Err Rate", "Latency", "Accuracy", "Result"
    );
    println!("   {}", "\u{2500}".repeat(56));
    let mut counter = 0_u64;
    while !deploy.is_complete() {
        let (stage_idx, canary_pct) = (deploy.current_stage, deploy.current_canary_pct());
        for _ in 0..MIN_REQUESTS_PER_STAGE * 3 {
            let (input, label) = &data[counter as usize % data.len()];
            deploy.process_request(counter, input, *label, latency_seed);
            counter += 1;
        }
        if let Some(health) = deploy.evaluate_stage() {
            println!(
                "   {:>6} {:>5}% {:>9.2}% {:>8.0}us {:>9.1}% {:>8}",
                stage_idx + 1,
                canary_pct,
                health.error_rate * 100.0,
                health.avg_latency_us,
                health.accuracy * 100.0,
                if health.passed { "PASS" } else { "FAIL" }
            );
        }
    }
}

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
