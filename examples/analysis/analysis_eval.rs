//! # APR Model Evaluation — Perplexity and Cross-Entropy
//!
//! CLI equivalent: `apr eval model.apr --dataset test.jsonl`
//!
//! Evaluates an APR language model by computing perplexity and cross-entropy
//! on synthetic test data. Uses the log-sum-exp trick for numerical stability.
//!
//!
//! ## Format Variants
//! ```bash
//! apr eval model.apr          # APR native format
//! apr eval model.gguf         # GGUF (llama.cpp compatible)
//! apr eval model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use rand::Rng;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct EvalConfig {
    vocab_size: usize,
    threshold_ppl: f64,
    dataset_name: String,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            threshold_ppl: 20.0,
            dataset_name: "synthetic-test".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
struct EvalResult {
    perplexity: f64,
    cross_entropy: f64,
    tokens_evaluated: usize,
    passed: bool,
}

impl EvalResult {
    fn verdict(&self) -> &str {
        if self.passed {
            "PASS"
        } else {
            "FAIL"
        }
    }
}

#[derive(Debug, Clone)]
struct BucketResult {
    label: String,
    cross_entropy: f64,
    perplexity: f64,
    count: usize,
}

// ---------------------------------------------------------------------------
// Numerically stable softmax helpers
// ---------------------------------------------------------------------------

/// Compute log-sum-exp of a slice using the max-shift trick for stability.
fn log_sum_exp(logits: &[f64]) -> f64 {
    let max_val = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_infinite() {
        return f64::NEG_INFINITY;
    }
    let sum_exp: f64 = logits.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

/// Compute log-softmax for a single position, returning log P(target).
fn log_prob_at(logits: &[f64], target: usize) -> f64 {
    let lse = log_sum_exp(logits);
    logits[target] - lse
}

// ---------------------------------------------------------------------------
// Logit generation
// ---------------------------------------------------------------------------

/// Generate synthetic logits for `n_positions` over a vocabulary.
///
/// Each position gets `vocab_size` logit values drawn from the RNG.
/// The target token for each position is also drawn from the RNG.
fn generate_logits_and_targets(
    rng: &mut impl Rng,
    vocab_size: usize,
    n_positions: usize,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut logits = Vec::with_capacity(n_positions);
    let mut targets = Vec::with_capacity(n_positions);

    for _ in 0..n_positions {
        let row: Vec<f64> = (0..vocab_size).map(|_| rng.gen_range(-5.0..5.0)).collect();
        let target = rng.gen_range(0..vocab_size);
        logits.push(row);
        targets.push(target);
    }

    (logits, targets)
}

// ---------------------------------------------------------------------------
// Evaluation logic
// ---------------------------------------------------------------------------

/// Evaluate cross-entropy and perplexity over all positions.
#[cfg(test)]
fn evaluate(logits: &[Vec<f64>], targets: &[usize]) -> EvalResult {
    evaluate_with_threshold(logits, targets, f64::INFINITY)
}

/// Evaluate cross-entropy and perplexity, comparing against a threshold.
fn evaluate_with_threshold(
    logits: &[Vec<f64>],
    targets: &[usize],
    threshold_ppl: f64,
) -> EvalResult {
    let n = logits.len();
    if n == 0 {
        return EvalResult {
            perplexity: f64::INFINITY,
            cross_entropy: f64::INFINITY,
            tokens_evaluated: 0,
            passed: false,
        };
    }

    let total_neg_log_prob: f64 = logits
        .iter()
        .zip(targets.iter())
        .map(|(row, &t)| -log_prob_at(row, t))
        .sum();

    let cross_entropy = total_neg_log_prob / n as f64;
    let perplexity = cross_entropy.exp();
    let passed = perplexity < threshold_ppl;

    EvalResult {
        perplexity,
        cross_entropy,
        tokens_evaluated: n,
        passed,
    }
}

/// Evaluate a slice (bucket) of positions.
fn evaluate_bucket(
    logits: &[Vec<f64>],
    targets: &[usize],
    start: usize,
    end: usize,
    label: &str,
    threshold_ppl: f64,
) -> BucketResult {
    let end = end.min(logits.len());
    let start = start.min(end);
    let slice_logits = &logits[start..end];
    let slice_targets = &targets[start..end];
    let result = evaluate_with_threshold(slice_logits, slice_targets, threshold_ppl);

    BucketResult {
        label: label.to_string(),
        cross_entropy: result.cross_entropy,
        perplexity: result.perplexity,
        count: end - start,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("analysis_eval")?;

    println!("=== APR Model Evaluation ===\n");

    let config = EvalConfig::default();
    let n_positions = 1000;

    // --- Section 1: Generate synthetic evaluation data ---
    println!("--- Dataset ---");
    println!("Dataset:    {}", config.dataset_name);
    println!("Vocab size: {}", config.vocab_size);
    println!("Positions:  {}", n_positions);
    println!("Threshold:  perplexity < {:.1}\n", config.threshold_ppl);

    let (logits, targets) = generate_logits_and_targets(ctx.rng(), config.vocab_size, n_positions);

    // --- Section 2: Full evaluation ---
    println!("--- Full Evaluation ---");
    let result = evaluate_with_threshold(&logits, &targets, config.threshold_ppl);

    println!("Tokens evaluated: {}", result.tokens_evaluated);
    println!("Cross-entropy:    {:.6}", result.cross_entropy);
    println!("Perplexity:       {:.4}", result.perplexity);
    println!("Verdict:          {}\n", result.verdict());

    // --- Section 3: Bucket breakdown ---
    println!("--- Position Bucket Breakdown ---\n");
    let mid_start = n_positions / 2 - 50;
    let mid_end = n_positions / 2 + 50;
    let buckets = [
        evaluate_bucket(&logits, &targets, 0, 100, "first-100", config.threshold_ppl),
        evaluate_bucket(
            &logits,
            &targets,
            mid_start,
            mid_end,
            "middle-100",
            config.threshold_ppl,
        ),
        evaluate_bucket(
            &logits,
            &targets,
            n_positions - 100,
            n_positions,
            "last-100",
            config.threshold_ppl,
        ),
    ];

    println!(
        "{:<12} {:>6} {:>14} {:>14}",
        "Bucket", "Count", "Cross-Entropy", "Perplexity"
    );
    println!("{}", "-".repeat(50));
    for b in &buckets {
        println!(
            "{:<12} {:>6} {:>14.6} {:>14.4}",
            b.label, b.count, b.cross_entropy, b.perplexity,
        );
    }

    // --- Section 4: Evaluation report ---
    println!("\n--- Evaluation Report ---");
    println!(
        "Model evaluation on '{}': {}",
        config.dataset_name,
        result.verdict()
    );
    println!(
        "  {} tokens | CE={:.6} | PPL={:.4} | threshold={:.1}",
        result.tokens_evaluated, result.cross_entropy, result.perplexity, config.threshold_ppl,
    );

    // --- Section 5: Sensitivity analysis ---
    println!("\n--- Threshold Sensitivity ---\n");
    let thresholds = [5.0, 10.0, 20.0, 50.0, 100.0];
    for &t in &thresholds {
        let r = evaluate_with_threshold(&logits, &targets, t);
        println!(
            "  threshold={:<6.1} perplexity={:<12.4} verdict={}",
            t,
            r.perplexity,
            r.verdict(),
        );
    }

    // Record metrics
    ctx.record_float_metric("cross_entropy", result.cross_entropy);
    ctx.record_float_metric("perplexity", result.perplexity);
    ctx.record_metric("tokens_evaluated", result.tokens_evaluated as i64);

    println!("\nEvaluation complete.");
    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform_logits(vocab_size: usize, n: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
        // Uniform logits: all zeros => softmax gives 1/V for each token.
        let logits: Vec<Vec<f64>> = (0..n).map(|_| vec![0.0; vocab_size]).collect();
        let targets: Vec<usize> = (0..n).map(|i| i % vocab_size).collect();
        (logits, targets)
    }

    #[test]
    fn test_log_sum_exp_single_element() {
        let result = log_sum_exp(&[3.0]);
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_stability() {
        // Large values that would overflow naive exp()
        let large = vec![1000.0, 1001.0, 1002.0];
        let result = log_sum_exp(&large);
        assert!(result.is_finite(), "log_sum_exp should handle large values");
        assert!(
            result > 1001.0,
            "result should be > max(inputs) due to extra terms"
        );
    }

    #[test]
    fn test_log_prob_uniform() {
        // Uniform logits: log P(any token) = -ln(V)
        let vocab = 100;
        let logits = vec![0.0; vocab];
        let lp = log_prob_at(&logits, 0);
        let expected = -(vocab as f64).ln();
        assert!(
            (lp - expected).abs() < 1e-10,
            "Uniform logits: log_prob={lp}, expected={expected}",
        );
    }

    #[test]
    fn test_log_prob_peaked() {
        // One logit much larger => probability near 1 => log_prob near 0
        let mut logits = vec![0.0; 10];
        logits[3] = 100.0;
        let lp = log_prob_at(&logits, 3);
        assert!(
            lp > -1e-6,
            "Peaked logit should yield log_prob near 0, got {lp}",
        );
    }

    #[test]
    fn test_evaluate_empty() {
        let result = evaluate(&[], &[]);
        assert!(result.perplexity.is_infinite());
        assert_eq!(result.tokens_evaluated, 0);
        assert!(!result.passed);
    }

    #[test]
    fn test_evaluate_uniform_perplexity() {
        // Uniform logits over V tokens => perplexity = V
        let vocab = 50;
        let (logits, targets) = make_uniform_logits(vocab, 200);
        let result = evaluate(&logits, &targets);
        let expected_ppl = vocab as f64;
        assert!(
            (result.perplexity - expected_ppl).abs() < 0.1,
            "Uniform logits should give PPL={expected_ppl}, got {:.4}",
            result.perplexity,
        );
        assert_eq!(result.tokens_evaluated, 200);
    }

    #[test]
    fn test_evaluate_threshold_pass() {
        let (logits, targets) = make_uniform_logits(10, 50);
        // PPL=10, threshold=20 => PASS
        let result = evaluate_with_threshold(&logits, &targets, 20.0);
        assert!(
            result.passed,
            "PPL={:.2} should be under 20.0",
            result.perplexity
        );
    }

    #[test]
    fn test_evaluate_threshold_fail() {
        let (logits, targets) = make_uniform_logits(100, 50);
        // PPL=100, threshold=20 => FAIL
        let result = evaluate_with_threshold(&logits, &targets, 20.0);
        assert!(
            !result.passed,
            "PPL={:.2} should exceed 20.0",
            result.perplexity
        );
    }

    #[test]
    fn test_bucket_subset() {
        let (logits, targets) = make_uniform_logits(10, 100);
        let bucket = evaluate_bucket(&logits, &targets, 10, 30, "mid", f64::INFINITY);
        assert_eq!(bucket.count, 20);
        assert_eq!(bucket.label, "mid");
        assert!(bucket.perplexity.is_finite());
    }

    #[test]
    fn test_deterministic_eval() {
        let mut ctx1 = RecipeContext::new("analysis_eval").expect("ctx1");
        let mut ctx2 = RecipeContext::new("analysis_eval").expect("ctx2");
        let (l1, t1) = generate_logits_and_targets(ctx1.rng(), 100, 50);
        let (l2, t2) = generate_logits_and_targets(ctx2.rng(), 100, 50);
        let r1 = evaluate(&l1, &t1);
        let r2 = evaluate(&l2, &t2);
        assert!(
            (r1.perplexity - r2.perplexity).abs() < 1e-10,
            "Same seed must yield identical perplexity",
        );
    }
}
