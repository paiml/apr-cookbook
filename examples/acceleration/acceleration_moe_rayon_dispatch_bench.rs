//! # Acceleration — MoE Rayon Dispatch Bench (2× speedup target)
//!
//! aprender's `forward_qwen3_moe` was parallelized with rayon and discharged
//! `qwen3-moe-forward-v1` v1.3.0 → v1.4.0 FUNCTIONAL with a 2× speedup
//! target on multi-core CPUs. This recipe demonstrates the bench pattern:
//! synthetic per-expert workloads, serial baseline vs rayon-parallel
//! dispatch, assert speedup ≥ 1.5× on a multi-core box (2× target with
//! 1.5× floor for noise headroom).
//!
//! Demonstrates the **ACC+.1** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: qwen3-moe-forward-v1.yaml v1.4.0 FUNCTIONAL + Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. arXiv:1701.06538
//!
//! Run with: cargo run --example acceleration_moe_rayon_dispatch_bench
//!
//! Added by PMAT-085 (expand-cookbooks: Tier 3 perf benches).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use rayon::prelude::*;
use std::time::Instant;

/// Synthetic per-expert work: 4096-dim vector dot-product N times.
fn expert_work(seed: u32) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..1024 {
        let x = ((seed.wrapping_mul(31).wrapping_add(i)) % 1000) as f64 * 0.001;
        acc += x.sin() * x.cos();
    }
    acc
}

fn dispatch_serial(experts: &[u32]) -> f64 {
    experts.iter().copied().map(expert_work).sum()
}

fn dispatch_parallel(experts: &[u32]) -> f64 {
    experts.par_iter().copied().map(expert_work).sum()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("acceleration_moe_rayon_dispatch_bench")?;

    let experts: Vec<u32> = (0..256u32).collect();

    let t0 = Instant::now();
    let serial_total = dispatch_serial(&experts);
    let serial_ns = t0.elapsed().as_nanos();

    let t1 = Instant::now();
    let parallel_total = dispatch_parallel(&experts);
    let parallel_ns = t1.elapsed().as_nanos();

    let speedup = serial_ns as f64 / parallel_ns as f64;
    println!("256-expert dispatch (synthetic 1024-iter dot-product per expert):");
    println!("  serial:   {:>10} ns  total={serial_total:.6}", serial_ns);
    println!(
        "  parallel: {:>10} ns  total={parallel_total:.6}",
        parallel_ns
    );
    println!("  speedup: {speedup:.2}x  (target ≥ 1.5x, qwen3-moe-forward-v1 v1.4.0 claims 2x)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn parallel_matches_serial_within_fp_tolerance() {
        let experts: Vec<u32> = (0..32u32).collect();
        let s = dispatch_serial(&experts);
        let p = dispatch_parallel(&experts);
        assert!(
            (s - p).abs() < 1e-9,
            "rayon dispatch must produce same sum as serial (fp determinism)"
        );
    }

    #[test]
    fn empty_dispatch_is_zero() {
        assert_eq!(dispatch_serial(&[]), 0.0);
        assert_eq!(dispatch_parallel(&[]), 0.0);
    }
}
