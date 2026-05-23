//! # Analysis Memory Leak Detector
//!
//! Series of memory snapshots over time → detect monotonic growth as
//! a leak signal. Heuristic: ≥ 80% of consecutive samples increasing
//! AND total growth > tolerance_pct → Leak. Stable (no trend) and
//! Recovering (decreasing) tiers also classified.
//!
//! Demonstrates the **ANL.55** recipe for PMAT-131 (analysis coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Goldberg & Wilson (1989). Tracing leaks in long-running programs.
//!
//! Run with: cargo run --example analysis_memory_leak_detector
//!
//! Added by PMAT-131 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MIN_SAMPLES: usize = 5;
const MONOTONIC_FRACTION: f64 = 0.8;
const GROWTH_TOLERANCE_PCT: f64 = 5.0;

#[derive(Debug, PartialEq)]
pub enum LeakVerdict {
    Stable {
        growth_pct: f64,
    },
    Leak {
        growth_pct: f64,
        monotonic_fraction: f64,
    },
    Recovering {
        shrinkage_pct: f64,
    },
    InsufficientSamples,
    InvalidSample {
        at_index: usize,
    },
}

pub fn detect(samples_bytes: &[u64]) -> LeakVerdict {
    if samples_bytes.len() < MIN_SAMPLES {
        return LeakVerdict::InsufficientSamples;
    }
    for (i, &v) in samples_bytes.iter().enumerate() {
        if v == 0 {
            return LeakVerdict::InvalidSample { at_index: i };
        }
    }
    let first = samples_bytes[0] as f64;
    let last = *samples_bytes.last().unwrap() as f64;
    let growth_pct = (last - first) / first * 100.0;
    let increasing = samples_bytes.windows(2).filter(|w| w[1] > w[0]).count();
    let frac = increasing as f64 / (samples_bytes.len() - 1) as f64;
    if growth_pct < -GROWTH_TOLERANCE_PCT {
        return LeakVerdict::Recovering {
            shrinkage_pct: growth_pct.abs(),
        };
    }
    if frac >= MONOTONIC_FRACTION && growth_pct > GROWTH_TOLERANCE_PCT {
        return LeakVerdict::Leak {
            growth_pct,
            monotonic_fraction: frac,
        };
    }
    LeakVerdict::Stable { growth_pct }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("analysis_memory_leak_detector")?;

    let monotonic: Vec<u64> = (100..=200u64).collect();
    println!("monotonic 100→200: {:?}", detect(&monotonic));

    let stable = vec![100u64, 102, 99, 101, 100];
    println!("stable: {:?}", detect(&stable));

    let recovering = vec![1000u64, 800, 600, 400, 200];
    println!("recovering: {:?}", detect(&recovering));

    println!("too few: {:?}", detect(&[1, 2]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn monotonic_growth_classified_leak() {
        let s: Vec<u64> = (100..=200).collect();
        let v = detect(&s);
        assert!(matches!(v, LeakVerdict::Leak { .. }));
    }

    #[test]
    fn flat_within_tolerance_stable() {
        let s = vec![100u64, 102, 99, 101, 100];
        let v = detect(&s);
        assert!(matches!(v, LeakVerdict::Stable { .. }));
    }

    #[test]
    fn decreasing_classified_recovering() {
        let s = vec![1000u64, 800, 600, 400, 200];
        let v = detect(&s);
        assert!(matches!(v, LeakVerdict::Recovering { .. }));
    }

    #[test]
    fn too_few_samples_rejected() {
        assert_eq!(detect(&[1, 2]), LeakVerdict::InsufficientSamples);
    }

    #[test]
    fn zero_sample_rejected() {
        let s = vec![100u64, 200, 0, 400, 500];
        let v = detect(&s);
        assert!(matches!(v, LeakVerdict::InvalidSample { at_index: 2 }));
    }

    #[test]
    fn small_growth_within_tolerance_stable() {
        // 100 → 102 = 2% growth, under 5% tolerance.
        let s = vec![100u64, 100, 101, 101, 102];
        let v = detect(&s);
        assert!(matches!(v, LeakVerdict::Stable { .. }));
    }

    #[test]
    fn large_growth_classified_leak() {
        // Strict monotonic 100 → 1000.
        let s = vec![100u64, 200, 400, 600, 800, 1000];
        let v = detect(&s);
        assert!(matches!(v, LeakVerdict::Leak { .. }));
    }

    #[test]
    fn growth_with_low_monotonic_fraction_stable() {
        // Net growth but oscillating (low monotonic frac).
        let s = vec![100u64, 200, 100, 200, 130];
        let v = detect(&s);
        // 30% net growth, 50% increasing (2/4) — under 80% threshold → Stable not Leak.
        assert!(matches!(v, LeakVerdict::Stable { .. }));
    }

    #[test]
    fn boundary_at_minimum_samples_handled() {
        let s = vec![100u64, 110, 120, 130, 140];
        let v = detect(&s);
        // 40% growth, 100% monotonic → Leak.
        assert!(matches!(v, LeakVerdict::Leak { .. }));
    }

    #[test]
    fn at_min_samples_minus_one_rejected() {
        let s: Vec<u64> = (1u64..=(MIN_SAMPLES as u64 - 1)).collect();
        assert_eq!(detect(&s), LeakVerdict::InsufficientSamples);
    }
}
