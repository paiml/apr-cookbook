//! # Recipe: Cross-Quantization Output Parity (FP32 vs Int8 vs Int4)
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr parity --ref fp32.apr --cmp int8.apr,int4.apr --tolerance 0.05`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example parity_quantization` exits 0
//! 2. [x] `cargo test --example parity_quantization` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr parity` in-process (no shell-out)
//! 10. [x] Unit tests cover cosine similarity, error decomposition, budget checks
//!
//! ## Learning Objective
//! Demonstrates numeric parity verification across FP32, INT8, and INT4
//! quantization regimes. Generates a reference FP32 output, synthesizes
//! quantization-induced drift, and reports absolute / relative / cosine-sim
//! drift versus a configurable tolerance band.
//!
//! ## Run Command
//! ```bash
//! cargo run --example parity_quantization
//! ```
//!
//! ## References
//! - Dettmers, T. et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS. arXiv:2208.07339

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use rand::Rng;
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    Fp32,
    Int8,
    Int4,
}

impl Quant {
    pub fn label(&self) -> &'static str {
        match self {
            Quant::Fp32 => "fp32",
            Quant::Int8 => "int8",
            Quant::Int4 => "int4",
        }
    }

    /// Expected drift magnitude per element for a well-calibrated quantizer.
    fn drift_scale(self) -> f64 {
        match self {
            Quant::Fp32 => 0.0,
            Quant::Int8 => 0.005,
            Quant::Int4 => 0.03,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParityReport {
    pub quant: Quant,
    pub max_abs: f64,
    pub mean_abs: f64,
    pub relative_error: f64,
    pub cosine_sim: f64,
    pub within_tolerance: bool,
}

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        0.0
    } else {
        dot / (na * nb)
    }
}

pub fn compute_parity(
    reference: &[f64],
    candidate: &[f64],
    quant: Quant,
    tol: f64,
) -> ParityReport {
    assert_eq!(reference.len(), candidate.len());
    let n = reference.len() as f64;
    let diffs: Vec<f64> = reference
        .iter()
        .zip(candidate.iter())
        .map(|(r, c)| (r - c).abs())
        .collect();
    let max_abs = diffs.iter().copied().fold(0.0, f64::max);
    let mean_abs = if n > 0.0 {
        diffs.iter().sum::<f64>() / n
    } else {
        0.0
    };
    let ref_norm: f64 = reference.iter().map(|x| x * x).sum::<f64>().sqrt();
    let relative_error = if ref_norm < 1e-12 {
        0.0
    } else {
        diffs.iter().map(|d| d * d).sum::<f64>().sqrt() / ref_norm
    };
    let cosine_sim = cosine_similarity(reference, candidate);
    ParityReport {
        quant,
        max_abs,
        mean_abs,
        relative_error,
        cosine_sim,
        within_tolerance: relative_error <= tol,
    }
}

fn synthesize_reference<R: Rng>(rng: &mut R, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn apply_quantization_drift<R: Rng>(rng: &mut R, reference: &[f64], quant: Quant) -> Vec<f64> {
    let scale = quant.drift_scale();
    reference
        .iter()
        .map(|x| x + rng.gen_range(-scale..=scale))
        .collect()
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("parity_quantization")?;
    println!("=== Recipe: {} ===", ctx.name());

    let tol = 0.05;
    let reference = synthesize_reference(ctx.rng(), 4096);

    let int8 = apply_quantization_drift(ctx.rng(), &reference, Quant::Int8);
    let int4 = apply_quantization_drift(ctx.rng(), &reference, Quant::Int4);

    let reports = vec![
        compute_parity(&reference, &reference, Quant::Fp32, tol),
        compute_parity(&reference, &int8, Quant::Int8, tol),
        compute_parity(&reference, &int4, Quant::Int4, tol),
    ];

    for r in &reports {
        println!(
            "{:<6} max_abs={:.6} mean_abs={:.6} rel_err={:.6} cosine={:.6} [{}]",
            r.quant.label(),
            r.max_abs,
            r.mean_abs,
            r.relative_error,
            r.cosine_sim,
            if r.within_tolerance { "PASS" } else { "FAIL" },
        );
    }

    let pass_count = reports.iter().filter(|r| r.within_tolerance).count();
    let report = json!({
        "recipe": ctx.name(),
        "tolerance": tol,
        "pass_count": pass_count,
        "reports": reports.iter().map(|r| json!({
            "quant": r.quant.label(),
            "max_abs": r.max_abs,
            "mean_abs": r.mean_abs,
            "relative_error": r.relative_error,
            "cosine_sim": r.cosine_sim,
            "within_tolerance": r.within_tolerance,
        })).collect::<Vec<_>>(),
    });
    let path = ctx.path("parity-quant.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    ctx.record_metric("pass_count", pass_count as i64);
    ctx.record_float_metric("tolerance", tol);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn identical_vectors_have_perfect_parity() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let r = compute_parity(&v, &v, Quant::Fp32, 0.01);
        assert_eq!(r.max_abs, 0.0);
        assert_eq!(r.mean_abs, 0.0);
        assert!(r.within_tolerance);
        assert!((r.cosine_sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_of_orthogonal_vectors_is_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn int4_drift_exceeds_tight_tolerance() {
        let mut rng = StdRng::seed_from_u64(1);
        let r = synthesize_reference(&mut rng, 2048);
        let d = apply_quantization_drift(&mut rng, &r, Quant::Int4);
        let rep = compute_parity(&r, &d, Quant::Int4, 0.001);
        assert!(!rep.within_tolerance);
    }

    #[test]
    fn int8_drift_is_small() {
        let mut rng = StdRng::seed_from_u64(2);
        let r = synthesize_reference(&mut rng, 2048);
        let d = apply_quantization_drift(&mut rng, &r, Quant::Int8);
        let rep = compute_parity(&r, &d, Quant::Int8, 0.05);
        assert!(rep.within_tolerance);
    }
}
