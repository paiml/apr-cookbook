//! # Recipe: A/B Experiment Workflow
//!
//! **Category**: Advanced - End-to-End Workflow
//! **CLI Equivalent**: `apr experiment`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## Learning Objective
//! Controlled A/B experiment comparing two model versions end-to-end:
//! run model_A (baseline) x run model_B (candidate) -> diff -> eval -> verdict.
//!
//! ## Run Command
//! ```bash
//! cargo run --example ab_experiment
//! ```
//!
//! ## Toyota Way Principles
//! - **Genchi Genbutsu** (Go and see): Measure actual model performance, not assumptions
//! - **Jidoka** (Quality built-in): Statistical significance gates before promotion
//! - **Kaizen** (Continuous improvement): Systematic model iteration via A/B testing
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

use apr_cookbook::prelude::*;
use rand::Rng;

// ============================================================================
// Data Structures
// ============================================================================

/// Configuration for an A/B experiment.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// Name of the baseline model (model A).
    pub model_a_name: String,
    /// Name of the candidate model (model B).
    pub model_b_name: String,
    /// Number of test samples to evaluate.
    pub n_samples: usize,
    /// p-value threshold for statistical significance.
    pub significance_threshold: f64,
}

/// Per-sample prediction result from a single model run.
#[derive(Debug, Clone)]
pub struct SampleResult {
    /// Zero-indexed sample identifier.
    pub sample_id: usize,
    /// Model prediction (continuous score in [0, 1]).
    pub prediction: f64,
    /// Confidence of the prediction (0 = uncertain, 1 = certain).
    pub confidence: f64,
    /// Simulated inference latency in milliseconds.
    pub latency_ms: f64,
    /// Whether the prediction matches the ground truth label.
    pub correct: bool,
}

/// Aggregated diff between model A and model B across all samples.
#[derive(Debug, Clone)]
pub struct DiffResult {
    /// Fraction of samples where both models predicted the same class.
    pub match_rate: f64,
    /// Mean difference in confidence (B - A); positive means B more confident.
    pub mean_confidence_delta: f64,
    /// Mean difference in latency (B - A); negative means B is faster.
    pub mean_latency_delta: f64,
    /// Accuracy of model A (fraction correct).
    pub accuracy_a: f64,
    /// Accuracy of model B (fraction correct).
    pub accuracy_b: f64,
}

/// Final experiment verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentVerdict {
    /// Model B is significantly better; promote it to production.
    Promote,
    /// Model A remains better or equivalent; keep the baseline.
    Keep,
    /// Results are not statistically significant; need more data.
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

fn main() -> Result<()> {
    println!("========================================================");
    println!("  A/B Experiment Workflow");
    println!("  model_A x model_B -> diff -> eval -> verdict");
    println!("========================================================");
    println!();

    let mut ctx = RecipeContext::new("ab_experiment")?;
    let report = run_experiment(&mut ctx)?;
    print_report(&report);

    ctx.record_float_metric("accuracy_a", report.diff.accuracy_a);
    ctx.record_float_metric("accuracy_b", report.diff.accuracy_b);
    ctx.record_float_metric("match_rate", report.diff.match_rate);
    ctx.record_float_metric("t_statistic", report.t_statistic);
    ctx.record_string_metric("verdict", report.verdict.to_string());

    println!("\nA/B experiment complete.");
    Ok(())
}

// ============================================================================
// Section 1: Setup
// ============================================================================

/// Build the experiment configuration.
fn setup_config() -> ExperimentConfig {
    ExperimentConfig {
        model_a_name: "baseline-v1.0".to_string(),
        model_b_name: "candidate-v1.1".to_string(),
        n_samples: 200,
        significance_threshold: 0.05,
    }
}

/// Generate synthetic ground truth labels for `n` samples.
///
/// Labels are binary (0.0 or 1.0), roughly balanced.
fn generate_ground_truth(rng: &mut impl Rng, n: usize) -> Vec<f64> {
    (0..n)
        .map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 })
        .collect()
}

// ============================================================================
// Section 2 & 3: Run Models A and B
// ============================================================================

/// Simulate running a model on `n_samples` test inputs.
///
/// `base_accuracy` controls how often the model is correct.
/// `base_latency` is the mean latency in ms; actual values are perturbed.
/// `confidence_bias` shifts the mean confidence score.
fn run_model(
    rng: &mut impl Rng,
    ground_truth: &[f64],
    base_accuracy: f64,
    base_latency: f64,
    confidence_bias: f64,
) -> Vec<SampleResult> {
    ground_truth
        .iter()
        .enumerate()
        .map(|(i, &label)| {
            let correct = rng.gen_bool(base_accuracy.clamp(0.0, 1.0));
            let prediction = if correct { label } else { 1.0 - label };
            let raw_confidence = rng.gen_range(0.5..1.0) + confidence_bias;
            let confidence = raw_confidence.clamp(0.0, 1.0);
            let latency_ms = base_latency + rng.gen_range(-2.0..2.0);

            SampleResult {
                sample_id: i,
                prediction,
                confidence,
                latency_ms: latency_ms.max(0.1),
                correct,
            }
        })
        .collect()
}

// ============================================================================
// Section 4: Diff
// ============================================================================

/// Compute per-sample and aggregate differences between model A and B results.
fn compute_diff(results_a: &[SampleResult], results_b: &[SampleResult]) -> DiffResult {
    let n = results_a.len() as f64;

    let mut matches = 0u64;
    let mut confidence_delta_sum = 0.0_f64;
    let mut latency_delta_sum = 0.0_f64;
    let mut correct_a = 0u64;
    let mut correct_b = 0u64;

    for (a, b) in results_a.iter().zip(results_b.iter()) {
        let prediction_match = (a.prediction - b.prediction).abs() < 1e-9;
        if prediction_match {
            matches += 1;
        }
        confidence_delta_sum += b.confidence - a.confidence;
        latency_delta_sum += b.latency_ms - a.latency_ms;
        if a.correct {
            correct_a += 1;
        }
        if b.correct {
            correct_b += 1;
        }
    }

    DiffResult {
        match_rate: f64::from(matches as u32) / n,
        mean_confidence_delta: confidence_delta_sum / n,
        mean_latency_delta: latency_delta_sum / n,
        accuracy_a: f64::from(correct_a as u32) / n,
        accuracy_b: f64::from(correct_b as u32) / n,
    }
}

// ============================================================================
// Section 5: Eval (Statistical Tests)
// ============================================================================

/// Paired t-test approximation for the confidence differences.
///
/// Returns the t-statistic: mean(diff) / (std(diff) / sqrt(n)).
fn paired_t_statistic(results_a: &[SampleResult], results_b: &[SampleResult]) -> f64 {
    let n = results_a.len();
    if n < 2 {
        return 0.0;
    }

    let diffs: Vec<f64> = results_a
        .iter()
        .zip(results_b.iter())
        .map(|(a, b)| b.confidence - a.confidence)
        .collect();

    let n_f = n as f64;
    let mean = diffs.iter().copied().sum::<f64>() / n_f;
    let variance = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n_f - 1.0);
    let std_dev = variance.sqrt();

    if std_dev < 1e-12 {
        return 0.0;
    }

    mean / (std_dev / n_f.sqrt())
}

/// Determine the experiment verdict based on accuracy, latency, and significance.
///
/// Approximate p-value check: for n >= 30 and two-sided test, |t| > 1.96
/// corresponds roughly to p < 0.05 under the normal approximation.
fn determine_verdict(
    diff: &DiffResult,
    t_stat: f64,
    config: &ExperimentConfig,
) -> ExperimentVerdict {
    // Critical t-value for approximate two-sided test at alpha = 0.05
    // For n >= 30, 1.96 is the standard normal critical value.
    let critical_t = if config.significance_threshold <= 0.01 {
        2.576
    } else if config.significance_threshold <= 0.05 {
        1.96
    } else {
        1.645
    };

    let significant = t_stat.abs() > critical_t;
    let accuracy_delta = diff.accuracy_b - diff.accuracy_a;
    let b_faster = diff.mean_latency_delta < 0.0;

    if accuracy_delta > 0.02 && significant {
        // Model B has meaningfully higher accuracy with statistical significance
        return ExperimentVerdict::Promote;
    }

    if accuracy_delta.abs() <= 0.02 && b_faster && significant {
        // Same accuracy but model B is significantly faster
        return ExperimentVerdict::Promote;
    }

    if accuracy_delta < -0.02 && significant {
        // Model A is significantly better
        return ExperimentVerdict::Keep;
    }

    if !significant {
        return ExperimentVerdict::Inconclusive;
    }

    ExperimentVerdict::Keep
}

// ============================================================================
// Pipeline Orchestrator
// ============================================================================

/// Run the full A/B experiment pipeline.
fn run_experiment(ctx: &mut RecipeContext) -> Result<ExperimentReport> {
    // Section 1: Setup
    let config = setup_config();
    println!(
        "[1/6] Setup: {} vs {}, {} samples, p < {}",
        config.model_a_name, config.model_b_name, config.n_samples, config.significance_threshold
    );

    let ground_truth = generate_ground_truth(ctx.rng(), config.n_samples);

    // Section 2: Run Model A (baseline)
    // base_accuracy = 0.72, latency ~10ms, no confidence bias
    let results_a = run_model(ctx.rng(), &ground_truth, 0.72, 10.0, 0.0);
    let acc_a = results_a.iter().filter(|r| r.correct).count() as f64 / config.n_samples as f64;
    println!(
        "[2/6] Run model_a: accuracy={:.1}%, mean_latency={:.2}ms",
        acc_a * 100.0,
        mean_latency(&results_a)
    );

    // Section 3: Run Model B (candidate -- slightly better)
    // base_accuracy = 0.82, latency ~9.5ms, confidence boost
    let results_b = run_model(ctx.rng(), &ground_truth, 0.82, 9.5, 0.06);
    let acc_b = results_b.iter().filter(|r| r.correct).count() as f64 / config.n_samples as f64;
    println!(
        "[3/6] Run model_b: accuracy={:.1}%, mean_latency={:.2}ms",
        acc_b * 100.0,
        mean_latency(&results_b)
    );

    // Section 4: Diff
    let diff = compute_diff(&results_a, &results_b);
    println!(
        "[4/6] Diff: match_rate={:.1}%, conf_delta={:+.4}, latency_delta={:+.2}ms",
        diff.match_rate * 100.0,
        diff.mean_confidence_delta,
        diff.mean_latency_delta
    );

    // Section 5: Eval
    let t_stat = paired_t_statistic(&results_a, &results_b);
    println!(
        "[5/6] Eval: accuracy_a={:.1}%, accuracy_b={:.1}%, t_stat={:.3}",
        diff.accuracy_a * 100.0,
        diff.accuracy_b * 100.0,
        t_stat
    );

    // Section 6: Verdict
    let verdict = determine_verdict(&diff, t_stat, &config);
    println!("[6/6] Verdict: {}", verdict);

    Ok(ExperimentReport {
        config,
        results_a,
        results_b,
        diff,
        t_statistic: t_stat,
        verdict,
    })
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute mean latency across a set of sample results.
fn mean_latency(results: &[SampleResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    results.iter().map(|r| r.latency_ms).sum::<f64>() / results.len() as f64
}

/// Print a formatted experiment report.
fn print_report(report: &ExperimentReport) {
    println!();
    println!("+-----------------------------------------------------------+");
    println!("| A/B Experiment Report                                     |");
    println!("+-----------------------------------------------------------+");
    println!("| Baseline : {:<46} |", report.config.model_a_name);
    println!("| Candidate: {:<46} |", report.config.model_b_name);
    println!("| Samples  : {:<46} |", report.config.n_samples);
    println!("+-----------------------------------------------------------+");
    println!("| Metric              | Model A     | Model B     | Delta   |");
    println!("+---------------------+-------------+-------------+---------+");
    println!(
        "| Accuracy            | {:>10.1}% | {:>10.1}% | {:>+6.1}% |",
        report.diff.accuracy_a * 100.0,
        report.diff.accuracy_b * 100.0,
        (report.diff.accuracy_b - report.diff.accuracy_a) * 100.0
    );
    println!(
        "| Mean Latency (ms)   | {:>11.2} | {:>11.2} | {:>+7.2} |",
        mean_latency(&report.results_a),
        mean_latency(&report.results_b),
        report.diff.mean_latency_delta
    );
    println!(
        "| Mean Confidence      | {:>11.4} | {:>11.4} | {:>+7.4} |",
        mean_confidence(&report.results_a),
        mean_confidence(&report.results_b),
        report.diff.mean_confidence_delta
    );
    println!("+---------------------+-------------+-------------+---------+");
    println!(
        "| Match Rate: {:.1}%    t-statistic: {:.3}    Verdict: {} |",
        report.diff.match_rate * 100.0,
        report.t_statistic,
        report.verdict
    );
    println!("+-----------------------------------------------------------+");
}

/// Compute mean confidence across a set of sample results.
fn mean_confidence(results: &[SampleResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    results.iter().map(|r| r.confidence).sum::<f64>() / results.len() as f64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_experiment_succeeds() {
        let mut ctx = RecipeContext::new("test_ab_full").expect("context");
        let report = run_experiment(&mut ctx).expect("experiment");
        assert_eq!(report.results_a.len(), 200);
        assert_eq!(report.results_b.len(), 200);
    }

    #[test]
    fn test_experiment_deterministic() {
        let mut ctx1 = RecipeContext::new("test_ab_determinism").expect("ctx1");
        let mut ctx2 = RecipeContext::new("test_ab_determinism").expect("ctx2");
        let r1 = run_experiment(&mut ctx1).expect("r1");
        let r2 = run_experiment(&mut ctx2).expect("r2");
        assert_eq!(r1.diff.accuracy_a, r2.diff.accuracy_a);
        assert_eq!(r1.diff.accuracy_b, r2.diff.accuracy_b);
        assert_eq!(r1.t_statistic, r2.t_statistic);
        assert_eq!(r1.verdict, r2.verdict);
    }

    #[test]
    fn test_verdict_is_promote() {
        let mut ctx = RecipeContext::new("test_ab_promote").expect("context");
        let report = run_experiment(&mut ctx).expect("experiment");
        // Model B is configured to be better (0.80 vs 0.72 accuracy)
        assert_eq!(
            report.verdict,
            ExperimentVerdict::Promote,
            "candidate should be promoted (higher accuracy)"
        );
    }

    #[test]
    fn test_ground_truth_balanced() {
        let mut ctx = RecipeContext::new("test_ab_gt").expect("context");
        let gt = generate_ground_truth(ctx.rng(), 1000);
        let ones = gt.iter().filter(|&&v| v > 0.5).count();
        // Should be roughly balanced (within 10% of 500)
        assert!(ones > 400, "too few positives: {}", ones);
        assert!(ones < 600, "too many positives: {}", ones);
    }

    #[test]
    fn test_sample_results_valid_ranges() {
        let mut ctx = RecipeContext::new("test_ab_ranges").expect("context");
        let gt = generate_ground_truth(ctx.rng(), 50);
        let results = run_model(ctx.rng(), &gt, 0.75, 10.0, 0.0);
        for r in &results {
            assert!(
                r.confidence >= 0.0 && r.confidence <= 1.0,
                "confidence out of range: {}",
                r.confidence
            );
            assert!(
                r.latency_ms > 0.0,
                "latency must be positive: {}",
                r.latency_ms
            );
            assert!(
                r.prediction == 0.0 || r.prediction == 1.0,
                "prediction must be binary: {}",
                r.prediction
            );
        }
    }

    #[test]
    fn test_diff_match_rate_range() {
        let mut ctx = RecipeContext::new("test_ab_diff_range").expect("context");
        let gt = generate_ground_truth(ctx.rng(), 100);
        let ra = run_model(ctx.rng(), &gt, 0.75, 10.0, 0.0);
        let rb = run_model(ctx.rng(), &gt, 0.75, 10.0, 0.0);
        let diff = compute_diff(&ra, &rb);
        assert!(
            diff.match_rate >= 0.0 && diff.match_rate <= 1.0,
            "match_rate out of range: {}",
            diff.match_rate
        );
    }

    #[test]
    fn test_paired_t_statistic_identical() {
        // When two result sets are identical, confidence diffs are all zero
        let mut ctx = RecipeContext::new("test_ab_t_identical").expect("context");
        let gt = generate_ground_truth(ctx.rng(), 50);
        let results = run_model(ctx.rng(), &gt, 0.75, 10.0, 0.0);
        let t = paired_t_statistic(&results, &results);
        assert!(
            t.abs() < 1e-9,
            "t-statistic for identical sets should be ~0, got {}",
            t
        );
    }

    #[test]
    fn test_determine_verdict_keep_when_a_better() {
        let diff = DiffResult {
            match_rate: 0.5,
            mean_confidence_delta: -0.1,
            mean_latency_delta: 0.0,
            accuracy_a: 0.90,
            accuracy_b: 0.70,
        };
        let config = setup_config();
        // Large negative t-stat => A is significantly better
        let verdict = determine_verdict(&diff, -5.0, &config);
        assert_eq!(verdict, ExperimentVerdict::Keep);
    }

    #[test]
    fn test_determine_verdict_inconclusive() {
        let diff = DiffResult {
            match_rate: 0.9,
            mean_confidence_delta: 0.001,
            mean_latency_delta: 0.0,
            accuracy_a: 0.80,
            accuracy_b: 0.81,
        };
        let config = setup_config();
        // Small t-stat => not significant
        let verdict = determine_verdict(&diff, 0.5, &config);
        assert_eq!(verdict, ExperimentVerdict::Inconclusive);
    }

    #[test]
    fn test_mean_latency_empty() {
        assert!((mean_latency(&[]) - 0.0).abs() < 1e-12);
    }
}
