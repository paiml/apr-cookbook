//! # CGP — Roofline Model: Classify Kernel as Compute- or Memory-Bound
//!
//! Build a synthetic `RooflineModel` for a hypothetical RTX 4090
//! (peak FP16 Tensor compute = 165 TFLOP/s, peak DRAM bandwidth =
//! 1008 GB/s), then classify two kernels:
//! - **Memory-bound**: low arithmetic intensity (0.5 FLOP/byte) →
//!   bandwidth-bottlenecked, achieved throughput well below peak.
//! - **Compute-bound**: high arithmetic intensity (256 FLOP/byte) →
//!   compute-bottlenecked, achieved throughput close to peak.
//!
//! The ridge point separates the two regimes:
//!   ridge = peak_compute / peak_bandwidth = 165e12 / 1008e9 ≈ 163.7
//!
//! Demonstrates the **CGP.2** recipe per
//! `docs/specifications/expand-cookbooks/subcrate-coverage.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Williams, S., Waterman, A., Patterson, D. (2009). Roofline: An Insightful Visual Performance Model. CACM 52(4). DOI: 10.1145/1498765.1498785
//!
//! Run with: cargo run --example cgp_roofline_classify_kernel
//!
//! Added by PMAT-083 (expand-cookbooks: aprender-cgp coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use cgp::analysis::roofline::{MemoryLevel, Precision, RooflineModel};
use std::collections::HashMap;

fn rtx_4090_model() -> RooflineModel {
    let mut peak_compute = HashMap::new();
    peak_compute.insert(Precision::Fp32, 82.6e12); // 82.6 TFLOPS
    peak_compute.insert(Precision::Fp16, 165.2e12); // 165 TFLOP/s tensor

    let mut peak_bandwidth = HashMap::new();
    peak_bandwidth.insert(MemoryLevel::Dram, 1008e9); // 1008 GB/s GDDR6X
    peak_bandwidth.insert(MemoryLevel::L2Cache, 5300e9); // estimated

    RooflineModel {
        target: "RTX 4090 (synthetic)".to_string(),
        peak_compute,
        peak_bandwidth,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cgp_roofline_classify_kernel")?;

    let model = rtx_4090_model();
    let ridge = model
        .ridge_point(Precision::Fp16, MemoryLevel::Dram)
        .expect("ridge point must compute for FP16/DRAM");
    println!("RTX 4090 FP16/DRAM ridge point: {:.2} FLOP/byte", ridge);

    // Memory-bound kernel: low AI, achieved throughput is BW-limited.
    let kernel_mb = model
        .classify(0.5, 400e9, Precision::Fp16, MemoryLevel::Dram)
        .expect("classify failed");
    // Compute-bound kernel: high AI, achieved throughput is compute-limited.
    let kernel_cb = model
        .classify(256.0, 130e12, Precision::Fp16, MemoryLevel::Dram)
        .expect("classify failed");

    println!(
        "memory-bound kernel (AI=0.5 FLOP/byte): {:?}",
        kernel_mb.bound
    );
    println!(
        "compute-bound kernel (AI=256 FLOP/byte): {:?}",
        kernel_cb.bound
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgp::analysis::roofline::Bound;

    #[test]
    fn classify_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn low_ai_classifies_memory_bound() {
        let model = rtx_4090_model();
        let p = model
            .classify(0.5, 400e9, Precision::Fp16, MemoryLevel::Dram)
            .unwrap();
        assert!(
            matches!(p.bound, Bound::Memory { .. }),
            "AI=0.5 should be memory-bound, got {:?}",
            p.bound
        );
    }

    #[test]
    fn high_ai_classifies_compute_bound() {
        let model = rtx_4090_model();
        let p = model
            .classify(256.0, 130e12, Precision::Fp16, MemoryLevel::Dram)
            .unwrap();
        assert!(
            matches!(p.bound, Bound::Compute { .. }),
            "AI=256 should be compute-bound, got {:?}",
            p.bound
        );
    }
}
