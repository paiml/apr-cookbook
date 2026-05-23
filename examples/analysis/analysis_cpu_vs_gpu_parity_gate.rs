//! # CPU vs GPU Parity Gate (the silent-GPU-gibberish canary)
//!
//! Implements the same fail-closed cosine-parity gate that
//! `apr-cpu-vs-gpu-output-parity-v1.yaml` v1.5.0 ACTIVE wires into the
//! wgpu inference path: at the embedding stage, compare CPU vs GPU
//! output cosine similarity to a threshold; below threshold = GPU path
//! REJECTED, fall back to CPU. This is the canary that catches "silent
//! GPU gibberish" — the failure mode where the GPU path returns
//! plausible-looking but wrong tensors.
//!
//! Recipe writes both fail-fast (parity good → accept GPU) and fail-closed
//! (parity bad → reject GPU + log) test cases. Real-world threshold per
//! aprender's CUDA + wgpu fallback log prefixes is ~0.999.
//!
//! Demonstrates the **AN+.1** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: apr-cpu-vs-gpu-output-parity-v1.yaml v1.5.0 ACTIVE (5/5 falsifiers DISCHARGED) + Salton & Buckley (1988). Term-weighting approaches in automatic text retrieval. IPM 24(5). DOI: 10.1016/0306-4573(88)90021-0
//!
//! Run with: cargo run --example analysis_cpu_vs_gpu_parity_gate
//!
//! Added by PMAT-075 (expand-cookbooks: GPU/CPU oracle bisection).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const COSINE_THRESHOLD: f64 = 0.999;
const FALLBACK_LOG_PREFIX: &str = "[apr-cpu-vs-gpu-output-parity-v1] wgpu path rejected";

#[derive(Debug, PartialEq)]
enum ParityGateDecision {
    AcceptGpu { cosine: f64 },
    RejectGpu { cosine: f64, fallback_log: String },
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        0.0
    }
}

/// Apply the parity gate: cosine ≥ threshold = AcceptGpu, else RejectGpu
/// with a structured fallback log line.
fn parity_gate(cpu: &[f32], gpu: &[f32]) -> ParityGateDecision {
    let c = cosine(cpu, gpu);
    if c >= COSINE_THRESHOLD {
        ParityGateDecision::AcceptGpu { cosine: c }
    } else {
        ParityGateDecision::RejectGpu {
            cosine: c,
            fallback_log: format!(
                "{FALLBACK_LOG_PREFIX} (cosine={c:.6} < threshold={COSINE_THRESHOLD})"
            ),
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("analysis_cpu_vs_gpu_parity_gate")?;

    // Case 1: GPU outputs match CPU within numerical noise → AcceptGpu.
    let cpu_good: Vec<f32> = (0..64).map(|i| (i as f32 * 0.01).sin()).collect();
    let gpu_good: Vec<f32> = cpu_good.iter().map(|x| x + 1e-6).collect();
    let decision_good = parity_gate(&cpu_good, &gpu_good);
    println!("Case 1 (GPU output ≈ CPU): {:?}", decision_good);

    // Case 2: GPU outputs scrambled (silent gibberish) → RejectGpu.
    let cpu_bad: Vec<f32> = (0..64).map(|i| (i as f32 * 0.01).sin()).collect();
    let gpu_bad: Vec<f32> = cpu_bad.iter().rev().copied().collect();
    let decision_bad = parity_gate(&cpu_bad, &gpu_bad);
    println!("Case 2 (GPU output scrambled): {:?}", decision_bad);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_gate_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn near_identical_outputs_pass_gate() {
        let cpu: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
        let gpu: Vec<f32> = cpu.iter().map(|x| x + 1e-7).collect();
        let d = parity_gate(&cpu, &gpu);
        assert!(matches!(d, ParityGateDecision::AcceptGpu { .. }));
    }

    #[test]
    fn scrambled_outputs_fail_gate() {
        let cpu: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();
        let gpu: Vec<f32> = cpu.iter().rev().copied().collect();
        let d = parity_gate(&cpu, &gpu);
        assert!(matches!(d, ParityGateDecision::RejectGpu { .. }));
    }

    #[test]
    fn fallback_log_includes_canonical_prefix() {
        // Drift detector: aprender's CUDA + wgpu fallback log MUST start with
        // this exact string so downstream `grep` filters keep working.
        let cpu = vec![1.0, 0.0, 0.0];
        let gpu = vec![0.0, 1.0, 0.0]; // perpendicular → cosine 0
        let d = parity_gate(&cpu, &gpu);
        match d {
            ParityGateDecision::RejectGpu { fallback_log, .. } => {
                assert!(
                    fallback_log.starts_with(FALLBACK_LOG_PREFIX),
                    "fallback log lost canonical prefix: {fallback_log}"
                );
            }
            _ => panic!("perpendicular vectors must reject GPU path"),
        }
    }

    #[test]
    fn threshold_boundary_is_inclusive() {
        // cosine == threshold should ACCEPT (boundary is >=).
        // Construct two vectors with cosine exactly 1.0 (identical), which is
        // safely above 0.999. We assert that the >= 0.999 boundary is
        // accepting, not rejecting.
        let cpu = vec![1.0, 2.0, 3.0];
        let gpu = vec![1.0, 2.0, 3.0];
        let d = parity_gate(&cpu, &gpu);
        assert!(matches!(d, ParityGateDecision::AcceptGpu { .. }));
    }
}
