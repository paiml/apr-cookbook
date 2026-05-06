//! # GPU Warp-Divergence Cost Classifier
//!
//! A CUDA warp executes 32 threads in lockstep. Branch divergence
//! serializes the divergent paths: cost ≈ paths_taken × per-path
//! latency. Tiers: 0% divergence = ideal, ≤ 25% = mild, ≤ 75% = bad,
//! > 75% = pathological. This recipe builds the classifier.
//!
//! Demonstrates the **GPU.7** recipe for PMAT-130 (gpu coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA CUDA C++ Best Practices Guide §11.2.
//!
//! Run with: cargo run --example gpu_warp_divergence_classifier
//!
//! Added by PMAT-130 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const WARP_SIZE: u32 = 32;

#[derive(Debug, PartialEq)]
pub enum DivergenceTier {
    Ideal,
    Mild { divergent_threads: u32 },
    Bad { divergent_threads: u32 },
    Pathological { divergent_threads: u32 },
    InvalidShape,
}

pub fn classify(divergent_threads: u32, warp_size: u32) -> DivergenceTier {
    if warp_size == 0 || divergent_threads > warp_size {
        return DivergenceTier::InvalidShape;
    }
    if divergent_threads == 0 {
        return DivergenceTier::Ideal;
    }
    let pct = f64::from(divergent_threads) / f64::from(warp_size);
    if pct <= 0.25 {
        DivergenceTier::Mild { divergent_threads }
    } else if pct <= 0.75 {
        DivergenceTier::Bad { divergent_threads }
    } else {
        DivergenceTier::Pathological { divergent_threads }
    }
}

pub fn estimated_slowdown_factor(num_paths: u32) -> f64 {
    // Each divergent path costs ~1 path-execution; ideal = 1 path.
    f64::from(num_paths.max(1))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_warp_divergence_classifier")?;

    for d in [0u32, 5, 10, 20, 28, 32, 40] {
        println!("divergent={d:>2} → {:?}", classify(d, WARP_SIZE));
    }
    for paths in [1u32, 2, 4, 8] {
        println!(
            "{paths} paths → slowdown ≈ {}",
            estimated_slowdown_factor(paths)
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
    fn zero_divergent_ideal() {
        assert_eq!(classify(0, WARP_SIZE), DivergenceTier::Ideal);
    }

    #[test]
    fn small_divergence_mild() {
        let v = classify(5, WARP_SIZE);
        assert!(matches!(v, DivergenceTier::Mild { .. }));
    }

    #[test]
    fn medium_divergence_bad() {
        // 10/32 = 31% → Bad.
        let v = classify(10, WARP_SIZE);
        assert!(matches!(v, DivergenceTier::Bad { .. }));
    }

    #[test]
    fn large_divergence_pathological() {
        // 28/32 = 87% → Pathological.
        let v = classify(28, WARP_SIZE);
        assert!(matches!(v, DivergenceTier::Pathological { .. }));
    }

    #[test]
    fn full_divergence_pathological() {
        let v = classify(32, WARP_SIZE);
        assert!(matches!(v, DivergenceTier::Pathological { .. }));
    }

    #[test]
    fn over_warp_size_invalid() {
        assert_eq!(classify(40, WARP_SIZE), DivergenceTier::InvalidShape);
    }

    #[test]
    fn zero_warp_size_invalid() {
        assert_eq!(classify(0, 0), DivergenceTier::InvalidShape);
    }

    #[test]
    fn boundary_at_25pct_mild() {
        // 8/32 = 25% exactly → Mild (≤ 25 inclusive).
        let v = classify(8, WARP_SIZE);
        assert!(matches!(v, DivergenceTier::Mild { .. }));
    }

    #[test]
    fn boundary_at_75pct_bad() {
        // 24/32 = 75% exactly → Bad (≤ 75 inclusive).
        let v = classify(24, WARP_SIZE);
        assert!(matches!(v, DivergenceTier::Bad { .. }));
    }

    #[test]
    fn slowdown_proportional_to_paths() {
        assert_eq!(estimated_slowdown_factor(1), 1.0);
        assert_eq!(estimated_slowdown_factor(8), 8.0);
        // Zero clamps to 1 (defensive).
        assert_eq!(estimated_slowdown_factor(0), 1.0);
    }
}
