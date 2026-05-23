//! # Recipe: Time-Slice Roofline Profile
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr profile --mode roofline --slice 100ms model.apr`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example profile_roofline` exits 0
//! 2. [x] `cargo test --example profile_roofline` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr profile --mode roofline` in-process (no shell-out)
//! 10. [x] Unit tests cover arithmetic intensity, ridge point, mem-bound classification
//!
//! ## Learning Objective
//! Demonstrates Williams et al. Roofline modeling: compute arithmetic intensity
//! (FLOPs/byte) per kernel, compare against the machine roofline (peak FLOPs,
//! peak bandwidth), and classify each kernel as memory-bound or compute-bound.
//!
//! ## Run Command
//! ```bash
//! cargo run --example profile_roofline
//! ```
//!
//! ## References
//! - Williams, S. et al. (2009). *Roofline: An Insightful Visual Performance Model for Multicore Architectures*. CACM. DOI: 10.1145/1498765.1498785

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    MemoryBound,
    ComputeBound,
}

impl Regime {
    pub fn label(&self) -> &'static str {
        match self {
            Regime::MemoryBound => "memory-bound",
            Regime::ComputeBound => "compute-bound",
        }
    }
}

#[derive(Debug, Clone)]
pub struct KernelSample {
    pub name: String,
    pub flops: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone)]
pub struct RooflineMachine {
    pub peak_gflops: f64,
    pub peak_gbs: f64,
}

impl RooflineMachine {
    /// Ridge-point intensity = peak FLOPS / peak bandwidth.
    pub fn ridge_point(&self) -> f64 {
        if self.peak_gbs < 1e-9 {
            0.0
        } else {
            self.peak_gflops / self.peak_gbs
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RooflineResult {
    pub name: String,
    pub intensity: f64,
    pub attainable_gflops: f64,
    pub regime: Regime,
}

pub fn arithmetic_intensity(sample: &KernelSample) -> f64 {
    if sample.bytes == 0 {
        0.0
    } else {
        sample.flops as f64 / sample.bytes as f64
    }
}

pub fn evaluate(sample: &KernelSample, machine: &RooflineMachine) -> RooflineResult {
    let intensity = arithmetic_intensity(sample);
    let attainable = (intensity * machine.peak_gbs).min(machine.peak_gflops);
    let regime = if intensity < machine.ridge_point() {
        Regime::MemoryBound
    } else {
        Regime::ComputeBound
    };
    RooflineResult {
        name: sample.name.clone(),
        intensity,
        attainable_gflops: attainable,
        regime,
    }
}

fn build_samples() -> Vec<KernelSample> {
    vec![
        KernelSample {
            name: "gemm_4096".into(),
            flops: 2 * 4096u64.pow(3),
            bytes: 3 * 4096u64.pow(2) * 2,
        },
        KernelSample {
            name: "memcpy_1GB".into(),
            flops: 0,
            bytes: 1 << 30,
        },
        KernelSample {
            name: "elementwise_relu".into(),
            flops: 1 << 20,
            bytes: 1 << 22,
        },
        KernelSample {
            name: "softmax_stream".into(),
            flops: 4 * (1 << 16),
            bytes: 8 * (1 << 16),
        },
    ]
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("profile_roofline")?;
    println!("=== Recipe: {} ===", ctx.name());

    let machine = RooflineMachine {
        peak_gflops: 312.0, // A100 FP16 tensor core effective
        peak_gbs: 1555.0,
    };
    println!(
        "Machine: peak_gflops={} peak_gbs={} ridge={:.4} FLOPs/byte",
        machine.peak_gflops,
        machine.peak_gbs,
        machine.ridge_point()
    );

    let samples = build_samples();
    let results: Vec<RooflineResult> = samples.iter().map(|s| evaluate(s, &machine)).collect();
    for r in &results {
        println!(
            "{:<20} intensity={:>10.3} attainable_gflops={:>10.3} {}",
            r.name,
            r.intensity,
            r.attainable_gflops,
            r.regime.label()
        );
    }

    let mem_bound = results
        .iter()
        .filter(|r| r.regime == Regime::MemoryBound)
        .count();

    let report = json!({
        "recipe": ctx.name(),
        "machine": {
            "peak_gflops": machine.peak_gflops,
            "peak_gbs": machine.peak_gbs,
            "ridge_point": machine.ridge_point(),
        },
        "mem_bound_kernels": mem_bound,
        "kernels": results.iter().map(|r| json!({
            "name": r.name,
            "intensity": r.intensity,
            "attainable_gflops": r.attainable_gflops,
            "regime": r.regime.label(),
        })).collect::<Vec<_>>(),
    });
    let path = ctx.path("profile-roofline.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    ctx.record_metric("mem_bound_kernels", mem_bound as i64);
    ctx.record_float_metric("ridge_point", machine.ridge_point());
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intensity_zero_for_zero_bytes() {
        let s = KernelSample {
            name: "z".into(),
            flops: 1000,
            bytes: 0,
        };
        assert_eq!(arithmetic_intensity(&s), 0.0);
    }

    #[test]
    fn ridge_point_is_peak_ratio() {
        let m = RooflineMachine {
            peak_gflops: 100.0,
            peak_gbs: 10.0,
        };
        assert!((m.ridge_point() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn memcpy_is_mem_bound() {
        let m = RooflineMachine {
            peak_gflops: 312.0,
            peak_gbs: 1555.0,
        };
        let s = KernelSample {
            name: "c".into(),
            flops: 0,
            bytes: 1_000_000,
        };
        let r = evaluate(&s, &m);
        assert_eq!(r.regime, Regime::MemoryBound);
    }

    #[test]
    fn high_intensity_is_compute_bound() {
        let m = RooflineMachine {
            peak_gflops: 100.0,
            peak_gbs: 10.0,
        };
        let s = KernelSample {
            name: "g".into(),
            flops: 100_000,
            bytes: 10,
        };
        let r = evaluate(&s, &m);
        assert_eq!(r.regime, Regime::ComputeBound);
    }

    #[test]
    fn attainable_bounded_by_peak() {
        let m = RooflineMachine {
            peak_gflops: 100.0,
            peak_gbs: 10.0,
        };
        let s = KernelSample {
            name: "g".into(),
            flops: 1_000_000_000,
            bytes: 1,
        };
        let r = evaluate(&s, &m);
        assert!((r.attainable_gflops - 100.0).abs() < 1e-9);
    }
}
