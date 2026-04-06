#![allow(unused_imports)]
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

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
