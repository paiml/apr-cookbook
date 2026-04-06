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

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Statistical process control metrics for parity comparison.
#[derive(Debug, Clone)]
pub struct SpcMetrics {
    pub cosine_similarity: f64,
    pub kl_divergence: f64,
    pub rmse: f64,
    pub max_abs_error: f64,
    pub sigma_level: f64,
}

/// Verdict for a parity check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    // Cosine similarity > 0.999 and argmax matches
    Pass,
    // Cosine similarity > 0.999 but argmax differs on at least one token
    WarnArgmax,
    // Cosine similarity <= 0.999 (significant divergence)
    FailDivergent,
    // NaN or Inf detected in logits
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
pub struct ParityScenario {
    pub label: String,
    pub cpu_logits: Vec<Vec<f64>>,
    pub gpu_logits: Vec<Vec<f64>>,
}

/// Result of a parity check.
#[derive(Debug, Clone)]
pub struct ParityResult {
    pub label: String,
    pub metrics: SpcMetrics,
    pub verdict: Verdict,
    pub argmax_mismatches: usize,
    pub tokens_checked: usize,
}

// ---------------------------------------------------------------------------
// SPC metric computation
// ---------------------------------------------------------------------------

/// Softmax of a logit vector (numerically stable via max-shift).
pub fn softmax(logits: &[f64]) -> Option<Vec<f64>> {
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
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> Option<f64> {
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

// KL divergence: D_KL(P || Q) where P=cpu_probs, Q=gpu_probs.
/// Uses epsilon floor to avoid log(0).
pub fn kl_divergence(p: &[f64], q: &[f64]) -> Option<f64> {
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
pub fn rmse(a: &[f64], b: &[f64]) -> Option<f64> {
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
pub fn max_abs_error(a: &[f64], b: &[f64]) -> Option<f64> {
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
pub fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

// Compute sigma level from cosine similarity.
//
// Maps cosine similarity to a sigma level where:
// - 1.0 cosine => 6.0 sigma (perfect)
// - 0.999 cosine => ~3.0 sigma (threshold)
/// - 0.0 cosine => 0.0 sigma
pub fn compute_sigma(cosine_sim: f64) -> f64 {
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
pub fn check_parity(scenario: &ParityScenario) -> ParityResult {
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
pub fn classify_verdict(cosine_sim: f64, argmax_mismatches: usize) -> Verdict {
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
pub fn generate_near_perfect(rng: &mut impl Rng, n_tokens: usize, vocab: usize) -> ParityScenario {
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
pub fn generate_slight_drift(rng: &mut impl Rng, n_tokens: usize, vocab: usize) -> ParityScenario {
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
pub fn generate_catastrophic(rng: &mut impl Rng, n_tokens: usize, vocab: usize) -> ParityScenario {
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
pub fn print_table_header() {
    println!(
        "{:<16} {:>8} {:>12} {:>10} {:>10} {:>8} {:>8} {:>16}",
        "Scenario", "Tokens", "Cosine", "KL-Div", "RMSE", "MaxErr", "Sigma", "Verdict"
    );
    println!("{}", "-".repeat(92));
}

/// Print a single parity result row.
pub fn print_result_row(r: &ParityResult) {
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
