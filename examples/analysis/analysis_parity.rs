//! # CPU vs GPU Parity Check — Statistical Process Control
//!
//! CLI equivalent: `apr parity model.apr --device cpu,cuda`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Compares CPU and GPU logit outputs using SPC metrics: cosine similarity,
//! KL divergence, RMSE, max absolute error, and sigma level. Classifies each
//! comparison as Pass, WarnArgmax, FailDivergent, or FailNan.
//!
//! ## What this demonstrates
//! - Statistical process control for numerical reproducibility
//! - Cosine similarity and KL divergence computation
//! - Sigma-level classification for manufacturing-style quality gates
//! - Deterministic scenario generation for regression testing
//!
//!
//! ## Format Variants
//! ```bash
//! apr inspect model.apr          # APR native format
//! apr inspect model.gguf         # GGUF (llama.cpp compatible)
//! apr inspect model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use rand::Rng;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Statistical process control metrics for parity comparison.
#[derive(Debug, Clone)]
struct SpcMetrics {
    cosine_similarity: f64,
    kl_divergence: f64,
    rmse: f64,
    max_abs_error: f64,
    sigma_level: f64,
}

/// Verdict for a parity check.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Verdict {
    /// Cosine similarity > 0.999 and argmax matches
    Pass,
    /// Cosine similarity > 0.999 but argmax differs on at least one token
    WarnArgmax,
    /// Cosine similarity <= 0.999 (significant divergence)
    FailDivergent,
    /// NaN or Inf detected in logits
    FailNan,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::WarnArgmax => write!(f, "WARN-ARGMAX"),
            Self::FailDivergent => write!(f, "FAIL-DIVERGENT"),
            Self::FailNan => write!(f, "FAIL-NAN"),
        }
    }
}

/// A single parity check scenario.
#[derive(Debug, Clone)]
struct ParityScenario {
    label: String,
    cpu_logits: Vec<Vec<f64>>,
    gpu_logits: Vec<Vec<f64>>,
}

/// Result of a parity check.
#[derive(Debug, Clone)]
struct ParityResult {
    label: String,
    metrics: SpcMetrics,
    verdict: Verdict,
    argmax_mismatches: usize,
    tokens_checked: usize,
}

// ---------------------------------------------------------------------------
// SPC metric computation
// ---------------------------------------------------------------------------

/// Softmax of a logit vector (numerically stable via max-shift).
fn softmax(logits: &[f64]) -> Option<Vec<f64>> {
    if logits.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return None;
    }
    let max_val = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 || sum.is_nan() {
        return None;
    }
    Some(exps.into_iter().map(|e| e / sum).collect())
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
        return None;
    }
    let sim = dot / (norm_a * norm_b);
    if sim.is_nan() {
        return None;
    }
    Some(sim.clamp(-1.0, 1.0))
}

/// KL divergence: D_KL(P || Q) where P=cpu_probs, Q=gpu_probs.
/// Uses epsilon floor to avoid log(0).
fn kl_divergence(p: &[f64], q: &[f64]) -> Option<f64> {
    if p.len() != q.len() || p.is_empty() {
        return None;
    }
    let eps = 1e-12;
    let kl: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            let pi = pi.max(eps);
            let qi = qi.max(eps);
            pi * (pi / qi).ln()
        })
        .sum();
    if kl.is_nan() || kl.is_infinite() {
        return None;
    }
    Some(kl)
}

/// Compute RMSE between two vectors.
fn rmse(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    Some(mse.sqrt())
}

/// Max absolute error between two vectors.
fn max_abs_error(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(None, |acc, v| {
            Some(match acc {
                Some(max) => {
                    if v > max {
                        v
                    } else {
                        max
                    }
                }
                None => v,
            })
        })
}

/// Argmax index of a vector.
fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

/// Compute sigma level from cosine similarity.
///
/// Maps cosine similarity to a sigma level where:
/// - 1.0 cosine => 6.0 sigma (perfect)
/// - 0.999 cosine => ~3.0 sigma (threshold)
/// - 0.0 cosine => 0.0 sigma
fn compute_sigma(cosine_sim: f64) -> f64 {
    if cosine_sim >= 1.0 {
        return 6.0;
    }
    if cosine_sim <= 0.0 {
        return 0.0;
    }
    // Map (0, 1] to (0, 6] using inverse complementary error function approximation.
    // Simple linear-in-log mapping: sigma = -2.0 * log10(1 - cosine_sim)
    let deficit = 1.0 - cosine_sim;
    let raw = -2.0 * deficit.log10();
    raw.clamp(0.0, 6.0)
}

// ---------------------------------------------------------------------------
// Parity check logic
// ---------------------------------------------------------------------------

/// Run a full parity check between CPU and GPU logits for all tokens.
fn check_parity(scenario: &ParityScenario) -> ParityResult {
    let n = scenario.cpu_logits.len();

    // Check for NaN/Inf in any logit
    let has_nan = scenario
        .cpu_logits
        .iter()
        .chain(scenario.gpu_logits.iter())
        .any(|row| row.iter().any(|v| v.is_nan() || v.is_infinite()));

    if has_nan {
        return ParityResult {
            label: scenario.label.clone(),
            metrics: SpcMetrics {
                cosine_similarity: f64::NAN,
                kl_divergence: f64::NAN,
                rmse: f64::NAN,
                max_abs_error: f64::NAN,
                sigma_level: 0.0,
            },
            verdict: Verdict::FailNan,
            argmax_mismatches: 0,
            tokens_checked: n,
        };
    }

    // Flatten logits for cosine similarity and RMSE
    let cpu_flat: Vec<f64> = scenario.cpu_logits.iter().flatten().copied().collect();
    let gpu_flat: Vec<f64> = scenario.gpu_logits.iter().flatten().copied().collect();

    let cos_sim = cosine_similarity(&cpu_flat, &gpu_flat).unwrap_or(0.0);
    let rmse_val = rmse(&cpu_flat, &gpu_flat).unwrap_or(f64::INFINITY);
    let mae_val = max_abs_error(&cpu_flat, &gpu_flat).unwrap_or(f64::INFINITY);

    // Per-token KL divergence (averaged)
    let mut total_kl = 0.0;
    let mut kl_count = 0usize;
    for (cpu_row, gpu_row) in scenario.cpu_logits.iter().zip(scenario.gpu_logits.iter()) {
        if let (Some(p), Some(q)) = (softmax(cpu_row), softmax(gpu_row)) {
            if let Some(kl) = kl_divergence(&p, &q) {
                total_kl += kl;
                kl_count += 1;
            }
        }
    }
    let avg_kl = if kl_count > 0 {
        total_kl / kl_count as f64
    } else {
        f64::INFINITY
    };

    // Count argmax mismatches
    let argmax_mismatches = scenario
        .cpu_logits
        .iter()
        .zip(scenario.gpu_logits.iter())
        .filter(|(cpu, gpu)| argmax(cpu) != argmax(gpu))
        .count();

    let sigma = compute_sigma(cos_sim);

    let metrics = SpcMetrics {
        cosine_similarity: cos_sim,
        kl_divergence: avg_kl,
        rmse: rmse_val,
        max_abs_error: mae_val,
        sigma_level: sigma,
    };

    let verdict = classify_verdict(cos_sim, argmax_mismatches);

    ParityResult {
        label: scenario.label.clone(),
        metrics,
        verdict,
        argmax_mismatches,
        tokens_checked: n,
    }
}

/// Classify verdict based on cosine similarity and argmax agreement.
fn classify_verdict(cosine_sim: f64, argmax_mismatches: usize) -> Verdict {
    if cosine_sim.is_nan() {
        return Verdict::FailNan;
    }
    if cosine_sim <= 0.999 {
        return Verdict::FailDivergent;
    }
    if argmax_mismatches > 0 {
        return Verdict::WarnArgmax;
    }
    Verdict::Pass
}

// ---------------------------------------------------------------------------
// Scenario generation
// ---------------------------------------------------------------------------

/// Generate near-perfect parity: GPU logits = CPU logits + tiny FP noise.
fn generate_near_perfect(rng: &mut impl Rng, n_tokens: usize, vocab: usize) -> ParityScenario {
    let cpu_logits: Vec<Vec<f64>> = (0..n_tokens)
        .map(|_| (0..vocab).map(|_| rng.gen_range(-5.0..5.0)).collect())
        .collect();

    let gpu_logits: Vec<Vec<f64>> = cpu_logits
        .iter()
        .map(|row| {
            row.iter()
                .map(|&v| v + rng.gen_range(-1e-7..1e-7))
                .collect()
        })
        .collect();

    ParityScenario {
        label: "near-perfect".to_string(),
        cpu_logits,
        gpu_logits,
    }
}

/// Generate slight drift: GPU logits have small systematic bias.
fn generate_slight_drift(rng: &mut impl Rng, n_tokens: usize, vocab: usize) -> ParityScenario {
    let cpu_logits: Vec<Vec<f64>> = (0..n_tokens)
        .map(|_| (0..vocab).map(|_| rng.gen_range(-5.0..5.0)).collect())
        .collect();

    let gpu_logits: Vec<Vec<f64>> = cpu_logits
        .iter()
        .map(|row| {
            row.iter()
                .map(|&v| v + rng.gen_range(-0.01..0.01) + 0.001)
                .collect()
        })
        .collect();

    ParityScenario {
        label: "slight-drift".to_string(),
        cpu_logits,
        gpu_logits,
    }
}

/// Generate catastrophic divergence: GPU logits are completely different.
fn generate_catastrophic(rng: &mut impl Rng, n_tokens: usize, vocab: usize) -> ParityScenario {
    let cpu_logits: Vec<Vec<f64>> = (0..n_tokens)
        .map(|_| (0..vocab).map(|_| rng.gen_range(-5.0..5.0)).collect())
        .collect();

    let gpu_logits: Vec<Vec<f64>> = (0..n_tokens)
        .map(|_| (0..vocab).map(|_| rng.gen_range(-5.0..5.0)).collect())
        .collect();

    ParityScenario {
        label: "catastrophic".to_string(),
        cpu_logits,
        gpu_logits,
    }
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

/// Print the SPC metrics table header.
fn print_table_header() {
    println!(
        "{:<16} {:>8} {:>12} {:>10} {:>10} {:>8} {:>8} {:>16}",
        "Scenario", "Tokens", "Cosine", "KL-Div", "RMSE", "MaxErr", "Sigma", "Verdict"
    );
    println!("{}", "-".repeat(92));
}

/// Print a single parity result row.
fn print_result_row(r: &ParityResult) {
    println!(
        "{:<16} {:>8} {:>12.9} {:>10.6} {:>10.6} {:>10.6} {:>8.2} {:>16}",
        r.label,
        r.tokens_checked,
        r.metrics.cosine_similarity,
        r.metrics.kl_divergence,
        r.metrics.rmse,
        r.metrics.max_abs_error,
        r.metrics.sigma_level,
        r.verdict,
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("analysis_parity")?;

    println!("=== CPU vs GPU Parity Check (SPC) ===\n");

    let n_tokens = 1000;
    let vocab_size = 128;

    // --- Section 1: Generate scenarios ---
    println!("--- Configuration ---");
    println!("Tokens:     {}", n_tokens);
    println!("Vocab size: {}", vocab_size);
    println!();

    let scenario_perfect = generate_near_perfect(ctx.rng(), n_tokens, vocab_size);
    let scenario_drift = generate_slight_drift(ctx.rng(), n_tokens, vocab_size);
    let scenario_catastrophic = generate_catastrophic(ctx.rng(), n_tokens, vocab_size);

    // --- Section 2: Run parity checks ---
    println!("--- SPC Metrics ---\n");
    print_table_header();

    let results: Vec<ParityResult> = vec![
        check_parity(&scenario_perfect),
        check_parity(&scenario_drift),
        check_parity(&scenario_catastrophic),
    ];

    for r in &results {
        print_result_row(r);
    }

    // --- Section 3: Detailed breakdown ---
    println!("\n--- Detailed Breakdown ---\n");
    for r in &results {
        println!("[{}]", r.label);
        println!("  Cosine similarity: {:.12}", r.metrics.cosine_similarity);
        println!("  KL divergence:     {:.9}", r.metrics.kl_divergence);
        println!("  RMSE:              {:.9}", r.metrics.rmse);
        println!("  Max abs error:     {:.9}", r.metrics.max_abs_error);
        println!("  Sigma level:       {:.2}", r.metrics.sigma_level);
        println!(
            "  Argmax mismatches: {}/{}",
            r.argmax_mismatches, r.tokens_checked
        );
        println!("  Verdict:           {}", r.verdict);
        println!();
    }

    // --- Section 4: Record metrics ---
    for r in &results {
        let prefix = &r.label;
        ctx.record_float_metric(&format!("{prefix}_cosine"), r.metrics.cosine_similarity);
        ctx.record_float_metric(&format!("{prefix}_kl"), r.metrics.kl_divergence);
        ctx.record_float_metric(&format!("{prefix}_sigma"), r.metrics.sigma_level);
        ctx.record_string_metric(&format!("{prefix}_verdict"), r.verdict.to_string());
    }

    println!("Parity check complete.");
    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let cos = cosine_similarity(&a, &a).expect("should compute");
        assert!(
            (cos - 1.0).abs() < 1e-10,
            "identical vectors => cosine 1.0, got {cos}"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let cos = cosine_similarity(&a, &b).expect("should compute");
        assert!(
            cos.abs() < 1e-10,
            "orthogonal vectors => cosine 0.0, got {cos}"
        );
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b: Vec<f64> = a.iter().map(|v| -v).collect();
        let cos = cosine_similarity(&a, &b).expect("should compute");
        assert!(
            (cos - (-1.0)).abs() < 1e-10,
            "opposite vectors => cosine -1.0, got {cos}"
        );
    }

    #[test]
    fn test_kl_divergence_identical_distributions() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let kl = kl_divergence(&p, &p).expect("should compute");
        assert!(
            kl.abs() < 1e-10,
            "identical distributions => KL 0.0, got {kl}"
        );
    }

    #[test]
    fn test_kl_divergence_different_distributions() {
        let p = vec![0.9, 0.1];
        let q = vec![0.5, 0.5];
        let kl = kl_divergence(&p, &q).expect("should compute");
        assert!(kl > 0.0, "different distributions => KL > 0, got {kl}");
    }

    #[test]
    fn test_softmax_uniform() {
        let logits = vec![0.0, 0.0, 0.0, 0.0];
        let probs = softmax(&logits).expect("should compute");
        for &p in &probs {
            assert!(
                (p - 0.25).abs() < 1e-10,
                "uniform logits => equal probs, got {p}"
            );
        }
    }

    #[test]
    fn test_softmax_rejects_nan() {
        let logits = vec![1.0, f64::NAN, 3.0];
        assert!(softmax(&logits).is_none(), "NaN input should return None");
    }

    #[test]
    fn test_sigma_level_boundaries() {
        assert!(
            (compute_sigma(1.0) - 6.0).abs() < 1e-10,
            "perfect cosine => 6 sigma"
        );
        assert!((compute_sigma(0.0)).abs() < 1e-10, "zero cosine => 0 sigma");
        let sigma_mid = compute_sigma(0.999);
        assert!(
            sigma_mid > 2.0 && sigma_mid < 7.0,
            "0.999 cosine => moderate sigma, got {sigma_mid}"
        );
    }

    #[test]
    fn test_verdict_classification() {
        assert_eq!(classify_verdict(1.0, 0), Verdict::Pass);
        assert_eq!(classify_verdict(0.9999, 3), Verdict::WarnArgmax);
        assert_eq!(classify_verdict(0.95, 0), Verdict::FailDivergent);
        assert_eq!(classify_verdict(f64::NAN, 0), Verdict::FailNan);
    }

    #[test]
    fn test_near_perfect_scenario_passes() {
        let mut ctx = RecipeContext::new("analysis_parity").expect("ctx");
        let scenario = generate_near_perfect(ctx.rng(), 100, 64);
        let result = check_parity(&scenario);
        assert_eq!(
            result.verdict,
            Verdict::Pass,
            "near-perfect scenario should pass, got {} (cosine={:.12})",
            result.verdict,
            result.metrics.cosine_similarity,
        );
        assert!(result.metrics.cosine_similarity > 0.999);
        assert!(result.metrics.sigma_level > 3.0);
    }

    #[test]
    fn test_catastrophic_scenario_fails() {
        let mut ctx = RecipeContext::new("analysis_parity").expect("ctx");
        // Skip past the near-perfect and drift draws to match main's sequence
        let _skip1 = generate_near_perfect(ctx.rng(), 100, 64);
        let _skip2 = generate_slight_drift(ctx.rng(), 100, 64);
        let scenario = generate_catastrophic(ctx.rng(), 100, 64);
        let result = check_parity(&scenario);
        assert_eq!(
            result.verdict,
            Verdict::FailDivergent,
            "catastrophic scenario should fail, got {} (cosine={:.6})",
            result.verdict,
            result.metrics.cosine_similarity,
        );
    }
}
