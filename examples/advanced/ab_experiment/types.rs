#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use rand::Rng;

// ============================================================================
// Data Structures
// ============================================================================

/// Configuration for an A/B experiment.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    // Name of the baseline model (model A).
    pub model_a_name: String,
    // Name of the candidate model (model B).
    pub model_b_name: String,
    // Number of test samples to evaluate.
    pub n_samples: usize,
    // p-value threshold for statistical significance.
    pub significance_threshold: f64,
}

/// Per-sample prediction result from a single model run.
#[derive(Debug, Clone)]
pub struct SampleResult {
    // Zero-indexed sample identifier.
    pub sample_id: usize,
    // Model prediction (continuous score in [0, 1]).
    pub prediction: f64,
    // Confidence of the prediction (0 = uncertain, 1 = certain).
    pub confidence: f64,
    // Simulated inference latency in milliseconds.
    pub latency_ms: f64,
    // Whether the prediction matches the ground truth label.
    pub correct: bool,
}

/// Aggregated diff between model A and model B across all samples.
#[derive(Debug, Clone)]
pub struct DiffResult {
    // Fraction of samples where both models predicted the same class.
    pub match_rate: f64,
    // Mean difference in confidence (B - A); positive means B more confident.
    pub mean_confidence_delta: f64,
    // Mean difference in latency (B - A); negative means B is faster.
    pub mean_latency_delta: f64,
    // Accuracy of model A (fraction correct).
    pub accuracy_a: f64,
    // Accuracy of model B (fraction correct).
    pub accuracy_b: f64,
}

/// Final experiment verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentVerdict {
    // Model B is significantly better; promote it to production.
    Promote,
    // Model A remains better or equivalent; keep the baseline.
    Keep,
    // Results are not statistically significant; need more data.
    Inconclusive,
}

impl std::fmt::Display for ExperimentVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Promote => write!(f, "PROMOTE model_b"),
            Self::Keep => write!(f, "KEEP model_a"),
            Self::Inconclusive => write!(f, "INCONCLUSIVE"),
        }
    }
}

/// Full experiment report containing all intermediate results.
#[derive(Debug, Clone)]
pub struct ExperimentReport {
    pub config: ExperimentConfig,
    pub results_a: Vec<SampleResult>,
    pub results_b: Vec<SampleResult>,
    pub diff: DiffResult,
    pub t_statistic: f64,
    pub verdict: ExperimentVerdict,
}

// ============================================================================
// Main Entry Point
// ============================================================================
