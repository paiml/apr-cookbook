//! # apr rosetta — Per-Tensor Statistical Fingerprint
//!
//! `apr rosetta fingerprint <FILE>` emits per-tensor (mean, std, min, max,
//! l2_norm) — a JAX-stat-001 compatible fingerprint that downstream
//! validation can compare across formats (PMAT-201). This recipe builds
//! the stats computation as a pure function so a CI pipeline can preview
//! the fingerprint that would be emitted.
//!
//! Demonstrates the **ROSETTA.5** recipe for PMAT-094 (apr rosetta coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-201 + JAX-STAT-001
//!
//! Run with: cargo run --example cli_rosetta_fingerprint_stats
//!
//! Added by PMAT-094 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct TensorFingerprint {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub l2_norm: f64,
}

pub fn fingerprint(values: &[f64]) -> Option<TensorFingerprint> {
    if values.is_empty() {
        return None;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let l2_norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
    Some(TensorFingerprint {
        mean,
        std,
        min,
        max,
        l2_norm,
    })
}

pub fn fingerprints_close(a: &TensorFingerprint, b: &TensorFingerprint, abs_tol: f64) -> bool {
    (a.mean - b.mean).abs() <= abs_tol
        && (a.std - b.std).abs() <= abs_tol
        && (a.min - b.min).abs() <= abs_tol
        && (a.max - b.max).abs() <= abs_tol
        && (a.l2_norm - b.l2_norm).abs() <= abs_tol
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_fingerprint_stats")?;

    let small: Vec<f64> = vec![-1.0, 0.0, 1.0, 2.0, 3.0];
    let big: Vec<f64> = (0..1024).map(|i| (i as f64 - 512.0) / 512.0).collect();

    println!("small:  {:?}", fingerprint(&small));
    if let Some(fp) = fingerprint(&big) {
        println!(
            "big:    mean={:.4}  std={:.4}  l2_norm={:.4}",
            fp.mean, fp.std, fp.l2_norm
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_returns_none() {
        assert!(fingerprint(&[]).is_none());
    }

    #[test]
    fn single_element_has_zero_std() {
        let fp = fingerprint(&[5.0]).unwrap();
        assert_eq!(fp.mean, 5.0);
        assert_eq!(fp.std, 0.0);
        assert_eq!(fp.min, 5.0);
        assert_eq!(fp.max, 5.0);
        assert!((fp.l2_norm - 5.0).abs() < 1e-12);
    }

    #[test]
    fn symmetric_distribution_has_zero_mean() {
        let fp = fingerprint(&[-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        assert!((fp.mean - 0.0).abs() < 1e-12);
        // l2_norm = sqrt(4+1+0+1+4) = sqrt(10)
        assert!((fp.l2_norm - 10.0_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn min_max_are_actual_extrema() {
        let fp = fingerprint(&[3.0, -7.0, 2.0, 5.0, -1.0]).unwrap();
        assert_eq!(fp.min, -7.0);
        assert_eq!(fp.max, 5.0);
    }

    #[test]
    fn fingerprints_close_within_tolerance() {
        let a = fingerprint(&[1.0, 2.0, 3.0]).unwrap();
        let b = fingerprint(&[1.0001, 2.0001, 3.0001]).unwrap();
        assert!(fingerprints_close(&a, &b, 1e-3));
        assert!(!fingerprints_close(&a, &b, 1e-9));
    }

    #[test]
    fn round_trip_fingerprint_matches_within_fp32_noise() {
        // Same data, same fingerprint — sanity that there's no nondeterminism.
        let v: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let a = fingerprint(&v).unwrap();
        let b = fingerprint(&v).unwrap();
        assert_eq!(a, b);
    }
}
