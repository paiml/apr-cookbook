//! # Monitoring Request Size Histogram
//!
//! Bucket request sizes (bytes) into power-of-2 ranges. Returns
//! P50/P90/P99 size estimates from buckets and total request count.
//!
//! Demonstrates the **MON.37** recipe for PMAT-155 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Prometheus histogram with exponential bucket boundaries.
//!
//! Run with: cargo run --example monitor_request_size_histogram
//!
//! Added by PMAT-155 (catalog 1018→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum HistVerdict {
    Ok {
        bucket_counts: Vec<u32>,
        p50_bucket: usize,
        p90_bucket: usize,
        p99_bucket: usize,
    },
    EmptyRequests,
}

pub fn build(request_sizes_bytes: &[u64], num_buckets: usize) -> HistVerdict {
    if request_sizes_bytes.is_empty() {
        return HistVerdict::EmptyRequests;
    }
    let n_buckets = num_buckets.max(2);
    let mut counts = vec![0u32; n_buckets];
    for &size in request_sizes_bytes {
        let bucket = bucket_index(size, n_buckets);
        counts[bucket] += 1;
    }
    let total: u32 = counts.iter().sum();
    HistVerdict::Ok {
        p50_bucket: percentile_bucket(&counts, total, 0.50),
        p90_bucket: percentile_bucket(&counts, total, 0.90),
        p99_bucket: percentile_bucket(&counts, total, 0.99),
        bucket_counts: counts,
    }
}

fn bucket_index(size: u64, n_buckets: usize) -> usize {
    if size == 0 {
        return 0;
    }
    let log = 64 - size.leading_zeros();
    (log as usize).min(n_buckets - 1)
}

fn percentile_bucket(counts: &[u32], total: u32, p: f64) -> usize {
    let target = (f64::from(total) * p) as u32;
    let mut cumsum = 0u32;
    for (i, &c) in counts.iter().enumerate() {
        cumsum += c;
        if cumsum >= target.max(1) {
            return i;
        }
    }
    counts.len() - 1
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_request_size_histogram")?;

    let sizes = vec![100, 200, 1000, 5000, 100_000];
    println!("typical: {:?}", build(&sizes, 24));
    println!("empty: {:?}", build(&[], 24));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(build(&[], 24), HistVerdict::EmptyRequests);
    }

    #[test]
    fn bucket_count_correct() {
        let v = build(&[100, 200, 300], 24);
        if let HistVerdict::Ok { bucket_counts, .. } = v {
            let total: u32 = bucket_counts.iter().sum();
            assert_eq!(total, 3);
        }
    }

    #[test]
    fn p50_below_p99() {
        let sizes: Vec<u64> = (1..=1000).collect();
        let v = build(&sizes, 24);
        if let HistVerdict::Ok {
            p50_bucket,
            p99_bucket,
            ..
        } = v
        {
            assert!(p50_bucket <= p99_bucket);
        }
    }

    #[test]
    fn small_sizes_low_bucket() {
        let v = build(&[10], 24);
        if let HistVerdict::Ok { p50_bucket, .. } = v {
            // log2(10) ≈ 4.
            assert!(p50_bucket <= 5);
        }
    }

    #[test]
    fn large_sizes_high_bucket() {
        let v = build(&[1_000_000_000], 24);
        if let HistVerdict::Ok { p50_bucket, .. } = v {
            // log2(1B) ≈ 30 → clamped to 23.
            assert_eq!(p50_bucket, 23);
        }
    }

    #[test]
    fn zero_size_in_bucket_0() {
        let v = build(&[0], 24);
        if let HistVerdict::Ok { p50_bucket, .. } = v {
            assert_eq!(p50_bucket, 0);
        }
    }

    #[test]
    fn min_buckets_two() {
        let v = build(&[100], 1);
        if let HistVerdict::Ok { bucket_counts, .. } = v {
            assert!(bucket_counts.len() >= 2);
        }
    }

    #[test]
    fn distribution_symmetry() {
        let sizes: Vec<u64> = vec![100, 100, 100, 100, 100];
        let v = build(&sizes, 24);
        if let HistVerdict::Ok {
            p50_bucket,
            p99_bucket,
            ..
        } = v
        {
            // All same size → all percentiles in same bucket.
            assert_eq!(p50_bucket, p99_bucket);
        }
    }

    #[test]
    fn large_distribution_stable() {
        let sizes: Vec<u64> = (1..=10_000).collect();
        let v = build(&sizes, 24);
        if let HistVerdict::Ok { bucket_counts, .. } = v {
            let total: u32 = bucket_counts.iter().sum();
            assert_eq!(total, 10_000);
        }
    }

    #[test]
    fn deterministic() {
        let a = build(&[100, 200, 300], 24);
        let b = build(&[100, 200, 300], 24);
        assert_eq!(a, b);
    }
}
