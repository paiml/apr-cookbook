//! # GPU Memory Bandwidth Roofline
//!
//! Achieved bandwidth = bytes_moved / elapsed_secs. Compare against
//! peak_bw to find efficiency: > 80% = excellent, 60-80% = good,
//! 30-60% = mediocre, < 30% = bandwidth-starved (look for memory
//! coalescing or compute-bound logic). This recipe builds the
//! calculator + tier classifier.
//!
//! Demonstrates the **GPU.9** recipe for PMAT-130 (gpu coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Williams et al. (2009). Roofline. CACM 52(4).
//!
//! Run with: cargo run --example gpu_memory_bandwidth_roofline
//!
//! Added by PMAT-130 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BandwidthTier {
    Excellent { efficiency_pct: f64 },
    Good { efficiency_pct: f64 },
    Mediocre { efficiency_pct: f64 },
    Starved { efficiency_pct: f64 },
    InvalidInputs,
}

pub fn achieved_gbs(bytes_moved: u64, elapsed_ms: f64) -> Option<f64> {
    if elapsed_ms <= 0.0 || !elapsed_ms.is_finite() {
        return None;
    }
    let elapsed_s = elapsed_ms / 1000.0;
    Some(bytes_moved as f64 / elapsed_s / 1e9)
}

pub fn classify(bytes_moved: u64, elapsed_ms: f64, peak_gbs: f64) -> BandwidthTier {
    if !peak_gbs.is_finite() || peak_gbs <= 0.0 {
        return BandwidthTier::InvalidInputs;
    }
    let Some(achieved) = achieved_gbs(bytes_moved, elapsed_ms) else {
        return BandwidthTier::InvalidInputs;
    };
    let efficiency_pct = achieved / peak_gbs * 100.0;
    if efficiency_pct >= 80.0 {
        BandwidthTier::Excellent { efficiency_pct }
    } else if efficiency_pct >= 60.0 {
        BandwidthTier::Good { efficiency_pct }
    } else if efficiency_pct >= 30.0 {
        BandwidthTier::Mediocre { efficiency_pct }
    } else {
        BandwidthTier::Starved { efficiency_pct }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_memory_bandwidth_roofline")?;

    let peak = 936.0; // RTX 3090
    let cases = [
        (10_000_000_000u64, 12.0_f64),
        (10_000_000_000, 20.0),
        (10_000_000_000, 50.0),
        (10_000_000_000, 200.0),
        (1_000, 0.0),
    ];
    for (bytes, ms) in cases {
        println!(
            "bytes={bytes:>14} ms={ms:>5}  →  {:?}",
            classify(bytes, ms, peak)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PEAK_RTX_3090: f64 = 936.0;

    #[test]
    fn roofline_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn achieved_gbs_basic_math() {
        // 1 GB moved in 1 ms = 1000 GB/s.
        let bw = achieved_gbs(1_000_000_000, 1.0).unwrap();
        assert!((bw - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn zero_elapsed_invalid() {
        assert!(achieved_gbs(1000, 0.0).is_none());
    }

    #[test]
    fn excellent_tier_at_high_efficiency() {
        // 10 GB in 12ms → 833 GB/s ≈ 89% of 936 GB/s peak.
        let v = classify(10_000_000_000, 12.0, PEAK_RTX_3090);
        assert!(matches!(v, BandwidthTier::Excellent { .. }));
    }

    #[test]
    fn good_tier_at_moderate_efficiency() {
        // 10 GB in 20ms → 500 GB/s ≈ 53% — actually Mediocre.
        // Use 16ms → 625 GB/s ≈ 67% → Good.
        let v = classify(10_000_000_000, 16.0, PEAK_RTX_3090);
        assert!(matches!(v, BandwidthTier::Good { .. }));
    }

    #[test]
    fn mediocre_tier_at_low_efficiency() {
        // 10 GB in 30ms → 333 GB/s ≈ 35% → Mediocre.
        let v = classify(10_000_000_000, 30.0, PEAK_RTX_3090);
        assert!(matches!(v, BandwidthTier::Mediocre { .. }));
    }

    #[test]
    fn starved_tier_at_very_low_efficiency() {
        // 10 GB in 200ms → 50 GB/s ≈ 5% → Starved.
        let v = classify(10_000_000_000, 200.0, PEAK_RTX_3090);
        assert!(matches!(v, BandwidthTier::Starved { .. }));
    }

    #[test]
    fn invalid_peak_rejected() {
        assert_eq!(classify(1000, 1.0, 0.0), BandwidthTier::InvalidInputs);
        assert_eq!(classify(1000, 1.0, -1.0), BandwidthTier::InvalidInputs);
    }

    #[test]
    fn zero_elapsed_invalid_in_classify() {
        assert_eq!(
            classify(1000, 0.0, PEAK_RTX_3090),
            BandwidthTier::InvalidInputs
        );
    }

    #[test]
    fn boundary_at_80pct_excellent() {
        // 80% of 936 = 748.8 GB/s. 10 GB in 13.36ms gives ~ 748.5 GB/s.
        // Use exact math: 936 × 0.8 GB/s × 13.36 ms = ...
        // Easier: bytes=748_800_000, elapsed=1ms → 748.8 GB/s → 80.0%.
        let v = classify(748_800_000, 1.0, PEAK_RTX_3090);
        assert!(matches!(v, BandwidthTier::Excellent { .. }));
    }
}
