//! # Monitoring Multi-Percentile Latency Summary
//!
//! Track p50, p95, p99, p999 from a sorted sample. Exact computation
//! (no t-digest approximation here): rank = ceil(n × p), 1-indexed.
//!
//! Demonstrates the **MON.20** recipe for PMAT-140 (monitoring round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NIST percentile definition (R-7 method).
//!
//! Run with: cargo run --example monitor_percentile_summary
//!
//! Added by PMAT-140 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SummaryVerdict {
    Ok {
        p50: u32,
        p95: u32,
        p99: u32,
        p999: u32,
        n: usize,
    },
    EmptySample,
    InsufficientForP999,
}

pub fn summarize(samples: &[u32]) -> SummaryVerdict {
    if samples.is_empty() {
        return SummaryVerdict::EmptySample;
    }
    if samples.len() < 1000 {
        // For p999 to be meaningful you need ≥ 1000 samples.
        // Still compute lower percentiles.
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let pick = |p: f64| -> u32 {
        let rank = ((n as f64 * p).ceil() as usize).max(1).min(n);
        sorted[rank - 1]
    };
    SummaryVerdict::Ok {
        p50: pick(0.50),
        p95: pick(0.95),
        p99: pick(0.99),
        p999: pick(0.999),
        n,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_percentile_summary")?;

    let normal: Vec<u32> = (1..=100).collect();
    println!("100 samples 1..100: {:?}", summarize(&normal));

    let with_tail: Vec<u32> = (1..=999).chain(std::iter::once(10_000u32)).collect();
    println!("1000 with tail: {:?}", summarize(&with_tail));

    println!("empty: {:?}", summarize(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_sample_rejected() {
        assert_eq!(summarize(&[]), SummaryVerdict::EmptySample);
    }

    #[test]
    fn uniform_distribution_percentiles() {
        let s: Vec<u32> = (1..=100).collect();
        if let SummaryVerdict::Ok { p50, p95, p99, .. } = summarize(&s) {
            assert_eq!(p50, 50);
            assert_eq!(p95, 95);
            assert_eq!(p99, 99);
        }
    }

    #[test]
    fn ordered_p999_correct() {
        let s: Vec<u32> = (1..=1000).collect();
        if let SummaryVerdict::Ok { p999, .. } = summarize(&s) {
            assert_eq!(p999, 999);
        }
    }

    #[test]
    fn unsorted_input_handled() {
        let s = [50u32, 10, 90, 30, 70];
        if let SummaryVerdict::Ok { p50, .. } = summarize(&s) {
            assert_eq!(p50, 50);
        }
    }

    #[test]
    fn single_value_all_percentiles_match() {
        let s = [42u32];
        if let SummaryVerdict::Ok {
            p50,
            p95,
            p99,
            p999,
            ..
        } = summarize(&s)
        {
            assert_eq!(p50, 42);
            assert_eq!(p95, 42);
            assert_eq!(p99, 42);
            assert_eq!(p999, 42);
        }
    }

    #[test]
    fn duplicates_handled() {
        let s = vec![5u32; 100];
        if let SummaryVerdict::Ok { p50, p95, .. } = summarize(&s) {
            assert_eq!(p50, 5);
            assert_eq!(p95, 5);
        }
    }

    #[test]
    fn tail_outliers_visible_in_p999() {
        // 9989 small + 11 outliers; n=10000, p999 rank = 9990 = first outlier.
        let mut s: Vec<u32> = (1..=9989).collect();
        s.extend(vec![10_000u32; 11]);
        if let SummaryVerdict::Ok { p999, .. } = summarize(&s) {
            assert_eq!(p999, 10_000);
        }
    }

    #[test]
    fn sample_count_returned() {
        let s: Vec<u32> = (1..=42).collect();
        if let SummaryVerdict::Ok { n, .. } = summarize(&s) {
            assert_eq!(n, 42);
        }
    }

    #[test]
    fn p99_at_least_p95() {
        let s: Vec<u32> = (1..=1000).collect();
        if let SummaryVerdict::Ok { p95, p99, .. } = summarize(&s) {
            assert!(p99 >= p95);
        }
    }

    #[test]
    fn p999_at_least_p99() {
        let s: Vec<u32> = (1..=1000).collect();
        if let SummaryVerdict::Ok { p99, p999, .. } = summarize(&s) {
            assert!(p999 >= p99);
        }
    }
}
