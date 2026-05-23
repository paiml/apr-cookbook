//! # Data Train/Val/Test Splitter
//!
//! Split a dataset into 70/15/15 (default) train/val/test partitions.
//! Constraints: ratios sum to 1.0 ± 1e-9; minimum 1 sample per
//! partition; index ranges must not overlap. This recipe builds the
//! splitter.
//!
//! Demonstrates the **DATA.22** recipe for PMAT-135 (data-loading coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: scikit-learn train_test_split documentation.
//!
//! Run with: cargo run --example data_train_val_test_split
//!
//! Added by PMAT-135 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitRanges {
    pub train: (usize, usize),
    pub val: (usize, usize),
    pub test: (usize, usize),
}

#[derive(Debug, PartialEq)]
pub enum SplitVerdict {
    Ok(SplitRanges),
    DatasetTooSmall { n: usize, min: usize },
    InvalidRatios,
}

const MIN_SAMPLES_PER_SPLIT: usize = 1;

pub fn split(n: usize, train_ratio: f64, val_ratio: f64, test_ratio: f64) -> SplitVerdict {
    if !train_ratio.is_finite() || !val_ratio.is_finite() || !test_ratio.is_finite() {
        return SplitVerdict::InvalidRatios;
    }
    if train_ratio < 0.0 || val_ratio < 0.0 || test_ratio < 0.0 {
        return SplitVerdict::InvalidRatios;
    }
    let sum = train_ratio + val_ratio + test_ratio;
    if (sum - 1.0).abs() > 1e-9 {
        return SplitVerdict::InvalidRatios;
    }
    let min_required = MIN_SAMPLES_PER_SPLIT * 3;
    if n < min_required {
        return SplitVerdict::DatasetTooSmall {
            n,
            min: min_required,
        };
    }
    let train_n = ((n as f64) * train_ratio).floor() as usize;
    let val_n = ((n as f64) * val_ratio).floor() as usize;
    let train_n = train_n.max(MIN_SAMPLES_PER_SPLIT);
    let val_n = val_n.max(MIN_SAMPLES_PER_SPLIT);
    let test_n = n.saturating_sub(train_n + val_n).max(MIN_SAMPLES_PER_SPLIT);
    let train = (0, train_n);
    let val = (train.1, train.1 + val_n);
    let test = (val.1, val.1 + test_n);
    SplitVerdict::Ok(SplitRanges { train, val, test })
}

pub fn split_default(n: usize) -> SplitVerdict {
    split(n, 0.70, 0.15, 0.15)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("data_train_val_test_split")?;

    println!("n=1000: {:?}", split_default(1000));
    println!("n=20: {:?}", split_default(20));
    println!("n=2 (too small): {:?}", split_default(2));
    println!("invalid sum: {:?}", split(100, 0.5, 0.3, 0.3));
    println!("80/10/10: {:?}", split(1000, 0.8, 0.1, 0.1));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn default_70_15_15_typical() {
        if let SplitVerdict::Ok(r) = split_default(1000) {
            assert_eq!(r.train, (0, 700));
            assert_eq!(r.val, (700, 850));
            assert_eq!(r.test, (850, 1000));
        }
    }

    #[test]
    fn custom_ratios_correct() {
        if let SplitVerdict::Ok(r) = split(1000, 0.8, 0.1, 0.1) {
            assert_eq!(r.train, (0, 800));
            assert_eq!(r.val, (800, 900));
            assert_eq!(r.test, (900, 1000));
        }
    }

    #[test]
    fn ranges_dont_overlap() {
        if let SplitVerdict::Ok(r) = split_default(1000) {
            assert_eq!(r.train.1, r.val.0);
            assert_eq!(r.val.1, r.test.0);
        }
    }

    #[test]
    fn ranges_cover_full_dataset() {
        if let SplitVerdict::Ok(r) = split_default(1000) {
            assert_eq!(r.train.0, 0);
            assert_eq!(r.test.1, 1000);
        }
    }

    #[test]
    fn dataset_too_small_rejected() {
        let v = split_default(2);
        assert!(matches!(v, SplitVerdict::DatasetTooSmall { .. }));
    }

    #[test]
    fn ratios_not_summing_to_one_rejected() {
        assert_eq!(split(1000, 0.5, 0.3, 0.3), SplitVerdict::InvalidRatios);
    }

    #[test]
    fn negative_ratio_rejected() {
        assert_eq!(split(1000, 0.7, -0.1, 0.4), SplitVerdict::InvalidRatios);
    }

    #[test]
    fn nan_ratio_rejected() {
        assert_eq!(
            split(1000, f64::NAN, 0.15, 0.15),
            SplitVerdict::InvalidRatios
        );
    }

    #[test]
    fn small_dataset_minimum_per_split() {
        // n=3, smallest valid: 1/1/1.
        if let SplitVerdict::Ok(r) = split_default(3) {
            assert!(r.train.1 - r.train.0 >= MIN_SAMPLES_PER_SPLIT);
            assert!(r.val.1 - r.val.0 >= MIN_SAMPLES_PER_SPLIT);
            assert!(r.test.1 - r.test.0 >= MIN_SAMPLES_PER_SPLIT);
        }
    }

    #[test]
    fn n_at_minimum_succeeds() {
        let v = split_default(3);
        assert!(matches!(v, SplitVerdict::Ok(_)));
    }

    #[test]
    fn ratios_summing_with_floating_point_tolerance() {
        // 0.7 + 0.15 + 0.15 may not == 1.0 exactly; tolerance handles it.
        let v = split(1000, 0.7, 0.15, 0.15);
        assert!(matches!(v, SplitVerdict::Ok(_)));
    }
}
