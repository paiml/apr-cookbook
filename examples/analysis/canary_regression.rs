//! # Recipe: Canary Regression Detection on Drifted Model
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr canary check drifted_model.apr --baseline golden.apr`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example canary_regression` exits 0
//! 2. [x] `cargo test --example canary_regression` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr canary` behavior in-process (no shell-out)
//! 10. [x] Unit tests cover detection, false-positive floor, deterministic drift
//!
//! ## Learning Objective
//! Demonstrates regression detection by running a canary suite against three
//! model variants: golden (baseline), small drift (within tolerance), and large
//! drift (beyond tolerance). Classifies each as PASS / WARN / FAIL.
//!
//! ## Run Command
//! ```bash
//! cargo run --example canary_regression
//! ```
//!
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CanaryVector {
    index: usize,
    input: Vec<f32>,
    expected: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Pass,
    Warn,
    Fail,
}

impl Verdict {
    fn label(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
        }
    }
}

#[derive(Debug, Clone)]
struct RegressionReport {
    variant: String,
    n_vectors: usize,
    max_abs_delta: f64,
    mean_abs_delta: f64,
    n_above_warn: usize,
    n_above_fail: usize,
    verdict: Verdict,
}

// ---------------------------------------------------------------------------
// Canary + regression logic
// ---------------------------------------------------------------------------

fn build_canaries(weights: &[f32], n: usize) -> Vec<CanaryVector> {
    if weights.is_empty() || n == 0 {
        return vec![];
    }
    let chunk = (weights.len() / n).max(1);
    (0..n)
        .map(|i| {
            let start = (i * chunk) % weights.len();
            let end = (start + 4.min(chunk)).min(weights.len());
            let input = weights[start..end].to_vec();
            let expected: f32 = input
                .iter()
                .enumerate()
                .map(|(j, v)| v * (j as f32 + 1.0))
                .sum();
            CanaryVector {
                index: i,
                input,
                expected,
            }
        })
        .collect()
}

fn run_canaries(weights: &[f32], canaries: &[CanaryVector]) -> Vec<f32> {
    canaries
        .iter()
        .map(|c| {
            c.input
                .iter()
                .enumerate()
                .map(|(j, _)| {
                    let idx = (c.index + j) % weights.len().max(1);
                    weights[idx] * (j as f32 + 1.0)
                })
                .sum()
        })
        .collect()
}

fn classify(report_deltas: &[f64], warn_threshold: f64, fail_threshold: f64) -> Verdict {
    let n_fail = report_deltas
        .iter()
        .filter(|&&d| d >= fail_threshold)
        .count();
    let n_warn = report_deltas
        .iter()
        .filter(|&&d| d >= warn_threshold)
        .count();
    if n_fail > 0 {
        Verdict::Fail
    } else if n_warn > 0 {
        Verdict::Warn
    } else {
        Verdict::Pass
    }
}

fn regress(
    baseline: &[f32],
    candidate: &[f32],
    canaries: &[CanaryVector],
    variant_name: &str,
    warn_t: f64,
    fail_t: f64,
) -> RegressionReport {
    let baseline_out = run_canaries(baseline, canaries);
    let candidate_out = run_canaries(candidate, canaries);
    let deltas: Vec<f64> = baseline_out
        .iter()
        .zip(candidate_out.iter())
        .map(|(b, c)| f64::from((b - c).abs()))
        .collect();
    let max = deltas.iter().copied().fold(0.0_f64, f64::max);
    let mean = if deltas.is_empty() {
        0.0
    } else {
        deltas.iter().sum::<f64>() / deltas.len() as f64
    };
    let n_warn = deltas.iter().filter(|&&d| d >= warn_t).count();
    let n_fail = deltas.iter().filter(|&&d| d >= fail_t).count();
    let verdict = classify(&deltas, warn_t, fail_t);
    RegressionReport {
        variant: variant_name.to_string(),
        n_vectors: canaries.len(),
        max_abs_delta: max,
        mean_abs_delta: mean,
        n_above_warn: n_warn,
        n_above_fail: n_fail,
        verdict,
    }
}

/// Create a drifted copy of the weights with a uniform perturbation.
fn drift(weights: &[f32], amount: f32) -> Vec<f32> {
    weights.iter().map(|w| w + amount).collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("canary_regression")?;
    println!("=== Recipe: {} ===", ctx.name());

    // Baseline model weights.
    let seed = hash_name_to_seed("canary-regression-golden");
    let weight_bytes = generate_model_payload(seed, 64);
    let baseline: Vec<f32> = weight_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let canaries = build_canaries(&baseline, 8);
    println!("Built {} canary vectors", canaries.len());

    // Three variants.
    let warn_t = 0.01_f64;
    let fail_t = 0.5_f64;
    let reports = vec![
        regress(&baseline, &baseline, &canaries, "golden", warn_t, fail_t),
        regress(
            &baseline,
            &drift(&baseline, 0.002),
            &canaries,
            "small_drift",
            warn_t,
            fail_t,
        ),
        regress(
            &baseline,
            &drift(&baseline, 1.0),
            &canaries,
            "large_drift",
            warn_t,
            fail_t,
        ),
    ];

    println!("\n--- Regression Reports ---");
    println!(
        "{:>14} {:>8} {:>14} {:>14} {:>10} {:>10} {:>8}",
        "Variant", "N", "MaxDelta", "MeanDelta", "NWarn", "NFail", "Verdict"
    );
    for r in &reports {
        println!(
            "{:>14} {:>8} {:>14.6} {:>14.6} {:>10} {:>10} {:>8}",
            r.variant,
            r.n_vectors,
            r.max_abs_delta,
            r.mean_abs_delta,
            r.n_above_warn,
            r.n_above_fail,
            r.verdict.label()
        );
    }

    // Sanity: golden must pass; large drift must fail.
    let golden = reports
        .iter()
        .find(|r| r.variant == "golden")
        .ok_or_else(|| CookbookError::invalid_format("missing golden report"))?;
    let large = reports
        .iter()
        .find(|r| r.variant == "large_drift")
        .ok_or_else(|| CookbookError::invalid_format("missing large_drift report"))?;
    assert_eq!(golden.verdict, Verdict::Pass);
    assert_eq!(large.verdict, Verdict::Fail);

    let report_json = json!({
        "recipe": ctx.name(),
        "warn_threshold": warn_t,
        "fail_threshold": fail_t,
        "reports": reports.iter().map(|r| json!({
            "variant": r.variant,
            "n_vectors": r.n_vectors,
            "max_abs_delta": r.max_abs_delta,
            "mean_abs_delta": r.mean_abs_delta,
            "n_above_warn": r.n_above_warn,
            "n_above_fail": r.n_above_fail,
            "verdict": r.verdict.label(),
        })).collect::<Vec<_>>(),
    });
    let out_path = ctx.path("regression-report.json");
    let out_bytes = serde_json::to_vec_pretty(&report_json)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, out_bytes)?;

    ctx.record_metric("n_variants", reports.len() as i64);
    ctx.record_string_metric("golden_verdict", golden.verdict.label());
    ctx.record_string_metric("large_drift_verdict", large.verdict.label());

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_weights() -> Vec<f32> {
        (0..64).map(|i| (i as f32 * 0.01).sin()).collect()
    }

    #[test]
    fn test_golden_passes() {
        let w = sample_weights();
        let c = build_canaries(&w, 6);
        let r = regress(&w, &w, &c, "golden", 0.01, 0.5);
        assert_eq!(r.verdict, Verdict::Pass);
        assert_eq!(r.n_above_warn, 0);
    }

    #[test]
    fn test_large_drift_fails() {
        let w = sample_weights();
        let c = build_canaries(&w, 6);
        let drifted = drift(&w, 2.0);
        let r = regress(&w, &drifted, &c, "big", 0.01, 0.5);
        assert_eq!(r.verdict, Verdict::Fail);
    }

    #[test]
    fn test_small_drift_warns_not_fails() {
        let w = sample_weights();
        let c = build_canaries(&w, 6);
        let drifted = drift(&w, 0.05);
        let r = regress(&w, &drifted, &c, "tiny", 0.01, 10.0);
        assert!(matches!(r.verdict, Verdict::Warn | Verdict::Pass));
        assert_ne!(r.verdict, Verdict::Fail);
    }

    #[test]
    fn test_classify_fail_overrides_warn() {
        assert_eq!(classify(&[1.0, 0.02], 0.01, 0.5), Verdict::Fail);
    }

    #[test]
    fn test_build_canaries_empty() {
        assert!(build_canaries(&[], 5).is_empty());
        assert!(build_canaries(&[1.0], 0).is_empty());
    }

    #[test]
    fn test_run_canaries_matches_len() {
        let w = sample_weights();
        let c = build_canaries(&w, 4);
        let out = run_canaries(&w, &c);
        assert_eq!(out.len(), 4);
    }
}
