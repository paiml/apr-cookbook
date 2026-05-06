//! # Serverless Cold-Start Latency Classifier
//!
//! Cold start latency: container init + runtime init + handler init.
//! Tiers: Warm (< 50 ms, container reused), Lukewarm (50-500 ms,
//! handler init only), Cold (500 ms - 5 s, full container start),
//! Frozen (> 5 s, custom runtime / large image). This recipe builds
//! the classifier + p99 target validator.
//!
//! Demonstrates the **SVL.5** recipe for PMAT-126 (serverless coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS Lambda whitepaper §Cold Starts.
//!
//! Run with: cargo run --example serverless_cold_start_classifier
//!
//! Added by PMAT-126 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum StartTier {
    Warm,
    Lukewarm,
    Cold,
    Frozen,
    InvalidLatency,
}

const WARM_CEIL_MS: u64 = 50;
const LUKEWARM_CEIL_MS: u64 = 500;
const COLD_CEIL_MS: u64 = 5_000;

pub fn classify(latency_ms: u64) -> StartTier {
    if latency_ms <= WARM_CEIL_MS {
        StartTier::Warm
    } else if latency_ms <= LUKEWARM_CEIL_MS {
        StartTier::Lukewarm
    } else if latency_ms <= COLD_CEIL_MS {
        StartTier::Cold
    } else {
        StartTier::Frozen
    }
}

#[derive(Debug, PartialEq)]
pub enum P99Verdict {
    Acceptable { p99_ms: u64, target_ms: u64 },
    Exceeds { p99_ms: u64, target_ms: u64 },
    InvalidSamples,
}

pub fn p99_check(samples_ms: &[u64], target_ms: u64) -> P99Verdict {
    if samples_ms.is_empty() {
        return P99Verdict::InvalidSamples;
    }
    let mut sorted = samples_ms.to_vec();
    sorted.sort_unstable();
    let idx = (sorted.len() as f64 * 0.99).ceil() as usize - 1;
    let p99 = sorted[idx.min(sorted.len() - 1)];
    if p99 <= target_ms {
        P99Verdict::Acceptable {
            p99_ms: p99,
            target_ms,
        }
    } else {
        P99Verdict::Exceeds {
            p99_ms: p99,
            target_ms,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("serverless_cold_start_classifier")?;

    for ms in [10u64, 100, 1_000, 10_000] {
        println!("{ms}ms → {:?}", classify(ms));
    }
    let samples = vec![10, 20, 30, 40, 50, 60, 100, 200, 800, 1_500];
    println!("p99 vs 1000ms: {:?}", p99_check(&samples, 1_000));
    println!("p99 vs 500ms: {:?}", p99_check(&samples, 500));
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
    fn under_50ms_warm() {
        assert_eq!(classify(10), StartTier::Warm);
        assert_eq!(classify(50), StartTier::Warm);
    }

    #[test]
    fn lukewarm_range_50_to_500() {
        assert_eq!(classify(51), StartTier::Lukewarm);
        assert_eq!(classify(500), StartTier::Lukewarm);
    }

    #[test]
    fn cold_range_500_to_5000() {
        assert_eq!(classify(501), StartTier::Cold);
        assert_eq!(classify(5_000), StartTier::Cold);
    }

    #[test]
    fn frozen_over_5000() {
        assert_eq!(classify(5_001), StartTier::Frozen);
        assert_eq!(classify(60_000), StartTier::Frozen);
    }

    #[test]
    fn p99_within_target_acceptable() {
        let samples = vec![10, 20, 30, 40, 50, 60, 100, 200, 800, 999];
        let v = p99_check(&samples, 1000);
        assert!(matches!(v, P99Verdict::Acceptable { .. }));
    }

    #[test]
    fn p99_exceeds_target_rejected() {
        let samples = vec![10, 20, 30, 40, 50, 60, 100, 200, 800, 1500];
        let v = p99_check(&samples, 1000);
        assert!(matches!(v, P99Verdict::Exceeds { .. }));
    }

    #[test]
    fn empty_samples_invalid() {
        assert_eq!(p99_check(&[], 1000), P99Verdict::InvalidSamples);
    }

    #[test]
    fn single_sample_returned_as_p99() {
        let v = p99_check(&[500], 1000);
        if let P99Verdict::Acceptable { p99_ms, .. } = v {
            assert_eq!(p99_ms, 500);
        }
    }

    #[test]
    fn boundary_at_warm_ceiling_is_warm() {
        assert_eq!(classify(WARM_CEIL_MS), StartTier::Warm);
    }
}
