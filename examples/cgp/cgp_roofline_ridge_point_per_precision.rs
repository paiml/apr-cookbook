//! # CGP — Roofline Ridge Points per Precision
//!
//! Compute the ridge point (peak compute / peak bandwidth = arithmetic
//! intensity at the compute/memory transition) for an RTX 4090 across
//! all 5 precisions: FP32, FP16, TF32, INT8, BF16. Same DRAM bandwidth
//! across precisions, so ridge scales linearly with peak compute.
//!
//! Useful when picking a precision for a kernel: pick the precision whose
//! ridge point is just *below* your kernel's arithmetic intensity to
//! maximize hardware utilization without wasted bandwidth.
//!
//! Demonstrates the **CGP.3** recipe per
//! `docs/specifications/expand-cookbooks/subcrate-coverage.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Williams, S., Waterman, A., Patterson, D. (2009). Roofline: An Insightful Visual Performance Model. CACM 52(4). DOI: 10.1145/1498765.1498785
//!
//! Run with: cargo run --example cgp_roofline_ridge_point_per_precision
//!
//! Added by PMAT-083 (expand-cookbooks: aprender-cgp coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use cgp::analysis::roofline::{MemoryLevel, Precision, RooflineModel};
use std::collections::HashMap;

fn rtx_4090() -> RooflineModel {
    let mut peak_compute = HashMap::new();
    peak_compute.insert(Precision::Fp32, 82.6e12);
    peak_compute.insert(Precision::Tf32, 82.6e12);
    peak_compute.insert(Precision::Bf16, 165.2e12);
    peak_compute.insert(Precision::Fp16, 330.3e12); // tensor with sparsity
    peak_compute.insert(Precision::Int8, 660.6e12); // tensor int8 with sparsity

    let mut peak_bandwidth = HashMap::new();
    peak_bandwidth.insert(MemoryLevel::Dram, 1008e9); // 1008 GB/s

    RooflineModel {
        target: "RTX 4090 (synthetic, with sparsity)".to_string(),
        peak_compute,
        peak_bandwidth,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cgp_roofline_ridge_point_per_precision")?;

    let model = rtx_4090();
    let precisions = [
        Precision::Fp32,
        Precision::Tf32,
        Precision::Bf16,
        Precision::Fp16,
        Precision::Int8,
    ];

    println!("RTX 4090 (synthetic) ridge points vs DRAM (1008 GB/s):");
    for p in &precisions {
        let ridge = model
            .ridge_point(*p, MemoryLevel::Dram)
            .expect("ridge point must compute");
        println!("  {p:>14}: {:>8.2} FLOP/byte", ridge);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn ridge_scales_with_peak_compute() {
        // INT8 has 2× the FP16 peak compute (in the model above), so the
        // INT8 ridge should be 2× the FP16 ridge at the same DRAM bandwidth.
        let model = rtx_4090();
        let ridge_fp16 = model
            .ridge_point(Precision::Fp16, MemoryLevel::Dram)
            .unwrap();
        let ridge_int8 = model
            .ridge_point(Precision::Int8, MemoryLevel::Dram)
            .unwrap();
        let ratio = ridge_int8 / ridge_fp16;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "INT8 ridge / FP16 ridge should be 2.0, got {ratio:.4}"
        );
    }

    #[test]
    fn missing_precision_returns_none() {
        // FP32 only in compute table; ask for a precision not in the map.
        let mut m = HashMap::new();
        m.insert(Precision::Fp32, 82.6e12);
        let mut bw = HashMap::new();
        bw.insert(MemoryLevel::Dram, 1008e9);
        let model = RooflineModel {
            target: "partial".into(),
            peak_compute: m,
            peak_bandwidth: bw,
        };
        assert!(model
            .ridge_point(Precision::Fp16, MemoryLevel::Dram)
            .is_none());
    }
}
