//! # Acceleration Arithmetic Intensity Roofline Classifier
//!
//! Roofline model: AI = FLOPs / bytes_moved. If AI > peak_flops /
//! peak_bandwidth → compute-bound; else memory-bound. Place workload
//! on roofline + recommend optimization. This recipe builds the
//! classifier + ridge-point calculator.
//!
//! Demonstrates the **ACCEL.5** recipe for PMAT-126 (acceleration coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Williams, Waterman & Patterson (2009). Roofline: an insightful visual performance model. CACM 52(4).
//!
//! Run with: cargo run --example acceleration_arithmetic_intensity
//!
//! Added by PMAT-126 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BoundClass {
    MemoryBound { ai: f64, ridge: f64 },
    ComputeBound { ai: f64, ridge: f64 },
    AtRidgePoint { ai: f64 },
    InvalidInputs,
}

pub fn arithmetic_intensity(flops: u64, bytes: u64) -> Option<f64> {
    if bytes == 0 {
        return None;
    }
    Some(flops as f64 / bytes as f64)
}

pub fn ridge_point(peak_gflops: f64, peak_bandwidth_gbs: f64) -> Option<f64> {
    if peak_bandwidth_gbs <= 0.0 || !peak_bandwidth_gbs.is_finite() {
        return None;
    }
    if peak_gflops <= 0.0 || !peak_gflops.is_finite() {
        return None;
    }
    Some(peak_gflops / peak_bandwidth_gbs)
}

pub fn classify(flops: u64, bytes: u64, peak_gflops: f64, peak_bandwidth_gbs: f64) -> BoundClass {
    let Some(ai) = arithmetic_intensity(flops, bytes) else {
        return BoundClass::InvalidInputs;
    };
    let Some(ridge) = ridge_point(peak_gflops, peak_bandwidth_gbs) else {
        return BoundClass::InvalidInputs;
    };
    if (ai - ridge).abs() < 1e-9 {
        BoundClass::AtRidgePoint { ai }
    } else if ai < ridge {
        BoundClass::MemoryBound { ai, ridge }
    } else {
        BoundClass::ComputeBound { ai, ridge }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("acceleration_arithmetic_intensity")?;

    let peak_gflops = 1000.0; // RTX 3090 FP32
    let peak_bw = 936.0; // GB/s

    println!("ridge: {:?}", ridge_point(peak_gflops, peak_bw));
    let cases = [
        (1_000_000u64, 4_000_000u64), // SGEMV-like, low AI
        (1_000_000_000, 4_000_000),   // GEMM, high AI
        (936, 1000),                  // exactly at ridge
    ];
    for (f, b) in cases {
        println!(
            "flops={f} bytes={b}  AI={:?}  →  {:?}",
            arithmetic_intensity(f, b),
            classify(f, b, peak_gflops, peak_bw)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn arithmetic_intensity_basic_math() {
        // 100 FLOPs / 4 bytes = 25 FLOPs/byte.
        let ai = arithmetic_intensity(100, 4).unwrap();
        assert!((ai - 25.0).abs() < 1e-9);
    }

    #[test]
    fn zero_bytes_invalid() {
        assert!(arithmetic_intensity(100, 0).is_none());
    }

    #[test]
    fn ridge_point_basic() {
        // 1 TFLOPS / 100 GB/s = 10 FLOPs/byte.
        let r = ridge_point(1000.0, 100.0).unwrap();
        assert!((r - 10.0).abs() < 1e-9);
    }

    #[test]
    fn ridge_point_zero_bandwidth_invalid() {
        assert!(ridge_point(1000.0, 0.0).is_none());
    }

    #[test]
    fn ridge_point_zero_flops_invalid() {
        assert!(ridge_point(0.0, 100.0).is_none());
    }

    #[test]
    fn low_ai_memory_bound() {
        // AI = 1/4 = 0.25 < ridge 10 → memory-bound.
        let v = classify(1, 4, 1000.0, 100.0);
        assert!(matches!(v, BoundClass::MemoryBound { .. }));
    }

    #[test]
    fn high_ai_compute_bound() {
        // AI = 100 > ridge 10 → compute-bound.
        let v = classify(1000, 10, 1000.0, 100.0);
        assert!(matches!(v, BoundClass::ComputeBound { .. }));
    }

    #[test]
    fn at_ridge_point_classified() {
        // AI = 10 exactly = ridge.
        let v = classify(100, 10, 1000.0, 100.0);
        assert!(matches!(v, BoundClass::AtRidgePoint { .. }));
    }

    #[test]
    fn invalid_inputs_rejected() {
        assert_eq!(classify(100, 0, 1000.0, 100.0), BoundClass::InvalidInputs);
        assert_eq!(classify(100, 10, 1000.0, 0.0), BoundClass::InvalidInputs);
    }
}
