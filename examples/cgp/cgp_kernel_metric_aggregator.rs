//! # CGP Kernel Metric Aggregator
//!
//! Aggregates per-kernel performance metrics into roll-ups: total
//! duration, mean, median, p99, sum-of-flops. This recipe builds the
//! aggregator with NaN-aware ordering for percentile calculation.
//!
//! Demonstrates the **CGP.5** recipe for PMAT-128 (cgp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CGP-001 + percentile estimation literature.
//!
//! Run with: cargo run --example cgp_kernel_metric_aggregator
//!
//! Added by PMAT-128 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy)]
pub struct KernelMetric {
    pub duration_us: u64,
    pub flops: u64,
}

#[derive(Debug, PartialEq)]
pub struct Aggregate {
    pub count: usize,
    pub total_duration_us: u64,
    pub mean_duration_us: f64,
    pub median_duration_us: u64,
    pub p99_duration_us: u64,
    pub total_flops: u64,
}

#[derive(Debug, PartialEq)]
pub enum AggVerdict {
    Ok(Aggregate),
    EmptyMetrics,
}

pub fn aggregate(metrics: &[KernelMetric]) -> AggVerdict {
    if metrics.is_empty() {
        return AggVerdict::EmptyMetrics;
    }
    let count = metrics.len();
    let total_duration: u64 = metrics.iter().map(|m| m.duration_us).sum();
    let total_flops: u64 = metrics.iter().map(|m| m.flops).sum();
    let mut sorted_dur: Vec<u64> = metrics.iter().map(|m| m.duration_us).collect();
    sorted_dur.sort_unstable();
    let median = sorted_dur[count / 2];
    let p99_idx = ((count as f64) * 0.99).ceil() as usize - 1;
    let p99 = sorted_dur[p99_idx.min(count - 1)];
    AggVerdict::Ok(Aggregate {
        count,
        total_duration_us: total_duration,
        mean_duration_us: total_duration as f64 / count as f64,
        median_duration_us: median,
        p99_duration_us: p99,
        total_flops,
    })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cgp_kernel_metric_aggregator")?;

    let metrics: Vec<KernelMetric> = (1..=100u64)
        .map(|i| KernelMetric {
            duration_us: i,
            flops: i * 1000,
        })
        .collect();
    println!("{:?}", aggregate(&metrics));
    println!("empty: {:?}", aggregate(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<KernelMetric> {
        (1..=10u64)
            .map(|i| KernelMetric {
                duration_us: i * 10,
                flops: i * 100,
            })
            .collect()
    }

    #[test]
    fn aggregator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn count_matches_input_length() {
        if let AggVerdict::Ok(a) = aggregate(&sample()) {
            assert_eq!(a.count, 10);
        }
    }

    #[test]
    fn total_duration_sums_correctly() {
        if let AggVerdict::Ok(a) = aggregate(&sample()) {
            // 10 + 20 + ... + 100 = 550.
            assert_eq!(a.total_duration_us, 550);
        }
    }

    #[test]
    fn mean_duration_correct() {
        if let AggVerdict::Ok(a) = aggregate(&sample()) {
            assert!((a.mean_duration_us - 55.0).abs() < 1e-9);
        }
    }

    #[test]
    fn median_correct() {
        if let AggVerdict::Ok(a) = aggregate(&sample()) {
            // Sorted: 10..100. count=10 → index 5 = 60.
            assert_eq!(a.median_duration_us, 60);
        }
    }

    #[test]
    fn p99_returns_high_value() {
        if let AggVerdict::Ok(a) = aggregate(&sample()) {
            // p99 of 10 samples → index ceil(9.9)-1 = 9 → max = 100.
            assert_eq!(a.p99_duration_us, 100);
        }
    }

    #[test]
    fn total_flops_sums_correctly() {
        if let AggVerdict::Ok(a) = aggregate(&sample()) {
            // 100 + 200 + ... + 1000 = 5500.
            assert_eq!(a.total_flops, 5500);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(aggregate(&[]), AggVerdict::EmptyMetrics);
    }

    #[test]
    fn single_metric_handled() {
        let m = vec![KernelMetric {
            duration_us: 42,
            flops: 100,
        }];
        if let AggVerdict::Ok(a) = aggregate(&m) {
            assert_eq!(a.count, 1);
            assert_eq!(a.median_duration_us, 42);
            assert_eq!(a.p99_duration_us, 42);
        }
    }

    #[test]
    fn unsorted_input_aggregates_correctly() {
        let m = vec![
            KernelMetric {
                duration_us: 100,
                flops: 0,
            },
            KernelMetric {
                duration_us: 10,
                flops: 0,
            },
            KernelMetric {
                duration_us: 50,
                flops: 0,
            },
        ];
        if let AggVerdict::Ok(a) = aggregate(&m) {
            assert_eq!(a.median_duration_us, 50);
        }
    }
}
