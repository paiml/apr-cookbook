//! # Data Row-Hash Deduplication Strategy
//!
//! Deduplicate dataset rows by hash. Three strategies: ExactMatch
//! (hash full row, fastest), MinHash (locality-sensitive, near-dup
//! detection), Sliding (n-gram fingerprint for partial overlap). This
//! recipe builds the picker + collision rate estimator.
//!
//! Demonstrates the **DATA.21** recipe for PMAT-132 (data-loading coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Broder (1997). On the resemblance and containment of documents.
//!
//! Run with: cargo run --example data_row_dedup_strategy
//!
//! Added by PMAT-132 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupStrategy {
    ExactMatch,
    MinHash,
    Sliding,
}

#[derive(Debug, PartialEq)]
pub enum PickerVerdict {
    Ok(DedupStrategy),
    InvalidRowCount,
}

pub fn pick_strategy(num_rows: u64, near_dup_pct_target: f64) -> PickerVerdict {
    if num_rows == 0 {
        return PickerVerdict::InvalidRowCount;
    }
    if !near_dup_pct_target.is_finite() || !(0.0..=100.0).contains(&near_dup_pct_target) {
        return PickerVerdict::Ok(DedupStrategy::ExactMatch);
    }
    if near_dup_pct_target < 1.0 {
        // Exact-only OK when no near-dup tolerance.
        PickerVerdict::Ok(DedupStrategy::ExactMatch)
    } else if num_rows > 1_000_000 {
        // Big dataset + want near-dup detection → MinHash.
        PickerVerdict::Ok(DedupStrategy::MinHash)
    } else {
        // Smaller dataset: sliding n-gram is feasible.
        PickerVerdict::Ok(DedupStrategy::Sliding)
    }
}

#[derive(Debug, PartialEq)]
pub enum ExactDedupVerdict {
    Ok { unique_count: usize, dup_pct: f64 },
    EmptyDataset,
}

pub fn exact_dedup(row_hashes: &[u64]) -> ExactDedupVerdict {
    if row_hashes.is_empty() {
        return ExactDedupVerdict::EmptyDataset;
    }
    let set: HashSet<u64> = row_hashes.iter().copied().collect();
    let dup = row_hashes.len() - set.len();
    let dup_pct = (dup as f64 / row_hashes.len() as f64) * 100.0;
    ExactDedupVerdict::Ok {
        unique_count: set.len(),
        dup_pct,
    }
}

pub fn estimated_collision_rate(num_unique: u64, hash_bits: u32) -> Option<f64> {
    if hash_bits == 0 || num_unique == 0 {
        return None;
    }
    // Birthday approximation: P_collide ≈ n² / (2 × 2^bits).
    let n = num_unique as f64;
    let space = 2f64.powi(hash_bits as i32);
    Some(n * n / (2.0 * space))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("data_row_dedup_strategy")?;

    for (rows, target) in [(1000u64, 0.0), (1000, 5.0), (10_000_000, 5.0)] {
        println!(
            "rows={rows} target={target}%  →  {:?}",
            pick_strategy(rows, target)
        );
    }

    let hashes = [1u64, 2, 3, 1, 2, 4, 5];
    println!("dedup: {:?}", exact_dedup(&hashes));

    println!(
        "collision (1M unique, 64-bit): {:?}",
        estimated_collision_rate(1_000_000, 64)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_dataset_picks_sliding() {
        // Want 5% near-dup tolerance + 1000 rows → Sliding.
        let v = pick_strategy(1000, 5.0);
        assert_eq!(v, PickerVerdict::Ok(DedupStrategy::Sliding));
    }

    #[test]
    fn huge_dataset_picks_minhash() {
        let v = pick_strategy(10_000_000, 5.0);
        assert_eq!(v, PickerVerdict::Ok(DedupStrategy::MinHash));
    }

    #[test]
    fn no_near_dup_target_picks_exact() {
        let v = pick_strategy(1000, 0.0);
        assert_eq!(v, PickerVerdict::Ok(DedupStrategy::ExactMatch));
    }

    #[test]
    fn zero_rows_rejected() {
        assert_eq!(pick_strategy(0, 5.0), PickerVerdict::InvalidRowCount);
    }

    #[test]
    fn exact_dedup_basic() {
        // [1, 2, 3, 1, 2, 4, 5] → 5 unique, 2 dup, dup_pct ≈ 28.6%.
        let v = exact_dedup(&[1, 2, 3, 1, 2, 4, 5]);
        if let ExactDedupVerdict::Ok {
            unique_count,
            dup_pct,
        } = v
        {
            assert_eq!(unique_count, 5);
            assert!((dup_pct - 28.571_428_571_428_57).abs() < 1e-9);
        }
    }

    #[test]
    fn no_duplicates_zero_pct() {
        let v = exact_dedup(&[1, 2, 3, 4]);
        if let ExactDedupVerdict::Ok { dup_pct, .. } = v {
            assert!(dup_pct.abs() < 1e-9);
        }
    }

    #[test]
    fn all_duplicates_high_pct() {
        let v = exact_dedup(&[1, 1, 1, 1]);
        if let ExactDedupVerdict::Ok {
            unique_count,
            dup_pct,
        } = v
        {
            assert_eq!(unique_count, 1);
            assert!((dup_pct - 75.0).abs() < 1e-9);
        }
    }

    #[test]
    fn empty_rows_rejected() {
        assert_eq!(exact_dedup(&[]), ExactDedupVerdict::EmptyDataset);
    }

    #[test]
    fn collision_rate_tiny_for_64bit() {
        // 1M unique in 2^64 space → very low collision probability.
        let p = estimated_collision_rate(1_000_000, 64).unwrap();
        assert!(p < 1e-7);
    }

    #[test]
    fn collision_rate_high_for_32bit() {
        // 1M in 2^32 → ~0.115 collision probability.
        let p = estimated_collision_rate(1_000_000, 32).unwrap();
        assert!(p > 0.1);
    }

    #[test]
    fn collision_rate_zero_inputs_invalid() {
        assert!(estimated_collision_rate(0, 64).is_none());
        assert!(estimated_collision_rate(1000, 0).is_none());
    }
}
