//! # Monitoring p99 Tail-Latency Outlier
//!
//! Flag tail latency outliers: when `p99 > p50 × ratio_threshold`, the
//! tail is unhealthy. Default ratio = 3.0 (Google SRE workbook).
//!
//! Demonstrates the **MON.36** recipe for PMAT-155 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Google SRE Workbook ch. 4 (Service Level Objectives).
//!
//! Run with: cargo run --example monitor_p99_outlier
//!
//! Added by PMAT-155 (catalog 1018→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum OutlierVerdict {
    Healthy { ratio: f64 },
    TailOutlier { ratio: f64, threshold: f64 },
    InvalidLatencies,
    EmptyLatencies,
}

pub fn detect(latencies_ms: &[f64], threshold: f64) -> OutlierVerdict {
    if latencies_ms.is_empty() {
        return OutlierVerdict::EmptyLatencies;
    }
    if latencies_ms.iter().any(|l| !l.is_finite() || *l < 0.0) {
        return OutlierVerdict::InvalidLatencies;
    }
    if !threshold.is_finite() || threshold <= 1.0 {
        return OutlierVerdict::InvalidLatencies;
    }
    let mut sorted = latencies_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = percentile(&sorted, 0.50);
    let p99 = percentile(&sorted, 0.99);
    if p50 <= 0.0 {
        return OutlierVerdict::InvalidLatencies;
    }
    let ratio = p99 / p50;
    if ratio > threshold {
        OutlierVerdict::TailOutlier { ratio, threshold }
    } else {
        OutlierVerdict::Healthy { ratio }
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len();
    let idx = ((n as f64) * p) as usize;
    sorted[idx.min(n - 1)]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_p99_outlier")?;

    let healthy: Vec<f64> = (0..100).map(f64::from).collect();
    let unhealthy: Vec<f64> = {
        let mut v: Vec<f64> = (0..99).map(f64::from).collect();
        v.push(10_000.0);
        v
    };
    println!("healthy: {:?}", detect(&healthy, 3.0));
    println!("unhealthy: {:?}", detect(&unhealthy, 3.0));
    println!("empty: {:?}", detect(&[], 3.0));
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
    fn flat_distribution_healthy() {
        let v = detect(&vec![10.0; 100], 3.0);
        assert!(matches!(v, OutlierVerdict::Healthy { .. }));
    }

    #[test]
    fn extreme_tail_flagged() {
        let mut latencies: Vec<f64> = vec![10.0; 99];
        latencies.push(1000.0);
        let v = detect(&latencies, 3.0);
        assert!(matches!(v, OutlierVerdict::TailOutlier { .. }));
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(detect(&[], 3.0), OutlierVerdict::EmptyLatencies);
    }

    #[test]
    fn negative_latency_rejected() {
        assert_eq!(detect(&[-1.0, 1.0], 3.0), OutlierVerdict::InvalidLatencies);
    }

    #[test]
    fn nan_rejected() {
        assert_eq!(detect(&[f64::NAN], 3.0), OutlierVerdict::InvalidLatencies);
    }

    #[test]
    fn threshold_below_one_rejected() {
        assert_eq!(detect(&[1.0, 2.0], 0.5), OutlierVerdict::InvalidLatencies);
    }

    #[test]
    fn ratio_carries_value() {
        let v = detect(&vec![10.0; 100], 3.0);
        if let OutlierVerdict::Healthy { ratio } = v {
            assert!((ratio - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn ratio_in_outlier_carries_threshold() {
        let mut l: Vec<f64> = vec![1.0; 99];
        l.push(100.0);
        let v = detect(&l, 3.0);
        if let OutlierVerdict::TailOutlier { threshold, .. } = v {
            assert!((threshold - 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn unsorted_input_works() {
        let unsorted = vec![5.0, 1.0, 100.0, 3.0, 2.0];
        let _v = detect(&unsorted, 3.0);
        // Just verify no panic; test logic is simple.
    }

    #[test]
    fn deterministic() {
        let l = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let a = detect(&l, 3.0);
        let b = detect(&l, 3.0);
        assert_eq!(a, b);
    }

    #[test]
    fn p50_zero_invalid() {
        // All zeros → p50 = 0 → divide-by-zero protection.
        assert_eq!(
            detect(&vec![0.0; 100], 3.0),
            OutlierVerdict::InvalidLatencies
        );
    }
}
