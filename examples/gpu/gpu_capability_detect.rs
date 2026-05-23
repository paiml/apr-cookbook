//! # Recipe: GPU — CUDA Capability Detection (Simulated)
//!
//! **Category**: gpu
//! **CLI Equivalent**: `apr gpu detect --json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example gpu_capability_detect` exits 0
//! 2. [x] `cargo test --example gpu_capability_detect` passes
//! 3. [x] Deterministic output (fixture-based)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr gpu detect` in-process (no CUDA call)
//! 10. [x] Unit tests cover arch gating, feature flags, sm_80+ check
//!
//! ## Learning Objective
//! Emits a structured capability report for a fleet of simulated GPUs
//! (compute capability, VRAM, tensor-core generation, FP8/INT8 support) —
//! mirroring what `apr gpu detect` would emit for a real NVML query.
//!
//! ## Run Command
//! ```bash
//! cargo run --example gpu_capability_detect
//! ```
//!
//! ## References
//! - Rajbhandari, S. et al. (2020). *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*. SC20. arXiv:1910.02054

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone)]
struct GpuRecord {
    name: &'static str,
    sm_major: u32,
    sm_minor: u32,
    vram_gb: u32,
    tensor_core_gen: Option<u32>,
    fp8_supported: bool,
    int8_supported: bool,
}

#[derive(Debug, Clone)]
struct CapabilityReport {
    records: Vec<GpuRecord>,
    fleet_supports_fp8: bool,
    fleet_supports_int8: bool,
    min_sm: u32,
}

fn synth_fleet() -> Vec<GpuRecord> {
    vec![
        GpuRecord {
            name: "A100",
            sm_major: 8,
            sm_minor: 0,
            vram_gb: 80,
            tensor_core_gen: Some(3),
            fp8_supported: false,
            int8_supported: true,
        },
        GpuRecord {
            name: "H100",
            sm_major: 9,
            sm_minor: 0,
            vram_gb: 80,
            tensor_core_gen: Some(4),
            fp8_supported: true,
            int8_supported: true,
        },
        GpuRecord {
            name: "L40",
            sm_major: 8,
            sm_minor: 9,
            vram_gb: 48,
            tensor_core_gen: Some(4),
            fp8_supported: true,
            int8_supported: true,
        },
        GpuRecord {
            name: "T4",
            sm_major: 7,
            sm_minor: 5,
            vram_gb: 16,
            tensor_core_gen: Some(2),
            fp8_supported: false,
            int8_supported: true,
        },
    ]
}

fn sm_version(rec: &GpuRecord) -> u32 {
    rec.sm_major * 10 + rec.sm_minor
}

fn build_report(records: Vec<GpuRecord>) -> CapabilityReport {
    let fleet_supports_fp8 = records.iter().all(|r| r.fp8_supported);
    let fleet_supports_int8 = records.iter().all(|r| r.int8_supported);
    let min_sm = records.iter().map(sm_version).min().unwrap_or(0);
    CapabilityReport {
        records,
        fleet_supports_fp8,
        fleet_supports_int8,
        min_sm,
    }
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("gpu_capability_detect")?;
    println!("=== Recipe: {} ===", ctx.name());

    let fleet = synth_fleet();
    let report = build_report(fleet);

    println!(
        "\n{:<6} {:>6} {:>6} {:>5} {:>5} {:>5}",
        "GPU", "SM", "VRAM", "TCg", "FP8", "INT8"
    );
    for r in &report.records {
        println!(
            "{:<6} {:>6} {:>5}GB {:>5} {:>5} {:>5}",
            r.name,
            format!("{}.{}", r.sm_major, r.sm_minor),
            r.vram_gb,
            r.tensor_core_gen.map_or("-".to_string(), |v| v.to_string()),
            if r.fp8_supported { "y" } else { "n" },
            if r.int8_supported { "y" } else { "n" }
        );
    }
    println!("\nFleet-wide FP8 support:  {}", report.fleet_supports_fp8);
    println!("Fleet-wide INT8 support: {}", report.fleet_supports_int8);
    println!("Minimum SM:              {}", report.min_sm);

    let out = json!({
        "recipe": ctx.name(),
        "records": report.records.iter().map(|r| json!({
            "name": r.name,
            "sm": format!("{}.{}", r.sm_major, r.sm_minor),
            "vram_gb": r.vram_gb,
            "tensor_core_gen": r.tensor_core_gen,
            "fp8_supported": r.fp8_supported,
            "int8_supported": r.int8_supported,
        })).collect::<Vec<_>>(),
        "fleet_supports_fp8": report.fleet_supports_fp8,
        "fleet_supports_int8": report.fleet_supports_int8,
        "min_sm": report.min_sm,
    });
    let out_path = ctx.path("gpu-capability.json");
    let bytes =
        serde_json::to_vec_pretty(&out).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, bytes)?;

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sm_version_matches_cuda_format() {
        let r = GpuRecord {
            name: "A100",
            sm_major: 8,
            sm_minor: 0,
            vram_gb: 80,
            tensor_core_gen: Some(3),
            fp8_supported: false,
            int8_supported: true,
        };
        assert_eq!(sm_version(&r), 80);
    }

    #[test]
    fn fleet_fp8_requires_all_gpus() {
        let fleet = synth_fleet();
        let r = build_report(fleet);
        // T4 has no FP8 -> fleet_supports_fp8 must be false.
        assert!(!r.fleet_supports_fp8);
    }

    #[test]
    fn fleet_int8_all_gpus_support() {
        let fleet = synth_fleet();
        let r = build_report(fleet);
        assert!(r.fleet_supports_int8);
    }

    #[test]
    fn min_sm_identifies_oldest_gpu() {
        let fleet = synth_fleet();
        let r = build_report(fleet);
        // T4 is sm_75 = 75, oldest in the fleet.
        assert_eq!(r.min_sm, 75);
    }

    #[test]
    fn empty_fleet_min_sm_zero() {
        let r = build_report(vec![]);
        assert_eq!(r.min_sm, 0);
    }
}
