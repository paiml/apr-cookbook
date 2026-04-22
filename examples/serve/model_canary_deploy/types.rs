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

pub const INPUT_DIM: usize = 16;
pub const OUTPUT_DIM: usize = 4;
pub const NUM_CANARY_STAGES: usize = 5;
pub const CANARY_STAGES: [u8; NUM_CANARY_STAGES] = [1, 5, 25, 50, 100];
pub const MIN_REQUESTS_PER_STAGE: usize = 50;

pub fn hash_seed(seed: u64, idx: usize) -> f32 {
    let mut h = DefaultHasher::new();
    (seed, idx).hash(&mut h);
    h.finish() as f32 / u64::MAX as f32 - 0.5
}

pub struct Model {
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub version: String,
}

impl Model {
    pub fn new(version: &str, seed: u64) -> Self {
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

    pub fn predict(&self, input: &[f32]) -> [f32; OUTPUT_DIM] {
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
pub struct PromotionCriteria {
    pub max_error_rate: f64,
    pub max_latency_us: f64,
    pub min_accuracy: f64,
}

impl PromotionCriteria {
    pub fn default_strict() -> Self {
        Self {
            max_error_rate: 0.10,
            max_latency_us: 500.0,
            min_accuracy: 0.15,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HealthCheckResult {
    pub passed: bool,
    pub error_rate: f64,
    pub avg_latency_us: f64,
    pub accuracy: f64,
    pub sample_count: usize,
    pub violations: Vec<String>,
}

impl HealthCheckResult {
    pub fn evaluate(metrics: &StageMetrics, criteria: &PromotionCriteria) -> Self {
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

pub struct StageMetrics {
    pub total_requests: usize,
    pub errors: usize,
    pub correct_predictions: usize,
    pub latency_sum_us: u64,
}

impl StageMetrics {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            errors: 0,
            correct_predictions: 0,
            latency_sum_us: 0,
        }
    }

    pub fn record(&mut self, latency_us: u64, is_error: bool, is_correct: bool) {
        self.total_requests += 1;
        self.latency_sum_us += latency_us;
        if is_error {
            self.errors += 1;
        }
        if is_correct {
            self.correct_predictions += 1;
        }
    }

    pub fn error_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.errors as f64 / self.total_requests as f64
        }
    }

    pub fn avg_latency_us(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.latency_sum_us as f64 / self.total_requests as f64
        }
    }

    pub fn accuracy(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.correct_predictions as f64 / self.total_requests as f64
        }
    }
}

pub struct CanaryStage {
    pub canary_pct: u8,
    pub criteria: PromotionCriteria,
    pub old_metrics: StageMetrics,
    pub new_metrics: StageMetrics,
}

impl CanaryStage {
    pub fn new(canary_pct: u8, criteria: PromotionCriteria) -> Self {
        Self {
            canary_pct,
            criteria,
            old_metrics: StageMetrics::new(),
            new_metrics: StageMetrics::new(),
        }
    }

    pub fn route_to_new(&self, request_id: u64) -> bool {
        let mut h = DefaultHasher::new();
        request_id.hash(&mut h);
        ((h.finish() % 100) as u8) < self.canary_pct
    }

    pub fn check_health(&self) -> HealthCheckResult {
        HealthCheckResult::evaluate(&self.new_metrics, &self.criteria)
    }

    pub fn has_enough_samples(&self) -> bool {
        self.new_metrics.total_requests >= MIN_REQUESTS_PER_STAGE
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeploymentOutcome {
    Promoted,
    RolledBack { stage: usize, reason: String },
    InProgress,
}

pub struct CanaryDeployment {
    pub old_model: Model,
    pub new_model: Model,
    pub stages: Vec<CanaryStage>,
    pub current_stage: usize,
    pub outcome: DeploymentOutcome,
    pub health_history: Vec<HealthCheckResult>,
}

impl CanaryDeployment {
    pub fn new(old_model: Model, new_model: Model, criteria: PromotionCriteria) -> Self {
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

    pub fn current_canary_pct(&self) -> u8 {
        self.stages
            .get(self.current_stage)
            .map_or(100, |s| s.canary_pct)
    }

    pub fn process_request(
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

    pub fn evaluate_stage(&mut self) -> Option<HealthCheckResult> {
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

    pub fn is_complete(&self) -> bool {
        self.outcome != DeploymentOutcome::InProgress
    }
}

pub fn generate_test_data(n: usize, seed: u64) -> Vec<([f32; INPUT_DIM], usize)> {
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
pub fn run_deployment(
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
