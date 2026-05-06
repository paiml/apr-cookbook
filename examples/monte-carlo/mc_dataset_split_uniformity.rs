//! # Monte-Carlo Dataset-Split Uniformity
//!
//! Verify a randomized 80/10/10 train/val/test split achieves the
//! requested ratios within tolerance. Returns observed split + chi-
//! square-style deviation metric.
//!
//! Demonstrates the **MC.20** recipe for PMAT-164 (catalog crosses 1100).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Stratified k-fold splits + chi-squared deviation.
//!
//! Run with: cargo run --example mc_dataset_split_uniformity
//!
//! Added by PMAT-164 (catalog 1099→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SplitVerdict {
    Ok {
        train: u32,
        val: u32,
        test: u32,
        max_deviation_pct: f64,
    },
    InvalidConfig,
}

pub fn simulate(n_total: u32, train_pct: f64, val_pct: f64, seed: u64) -> SplitVerdict {
    if n_total == 0
        || !train_pct.is_finite()
        || !val_pct.is_finite()
        || train_pct < 0.0
        || val_pct < 0.0
        || train_pct + val_pct > 1.0
    {
        return SplitVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut train = 0u32;
    let mut val = 0u32;
    let mut test = 0u32;
    for _ in 0..n_total {
        let u = unit(&mut rng_state);
        if u < train_pct {
            train += 1;
        } else if u < train_pct + val_pct {
            val += 1;
        } else {
            test += 1;
        }
    }
    let n = f64::from(n_total);
    let dev_train = ((f64::from(train) / n) - train_pct).abs() * 100.0;
    let dev_val = ((f64::from(val) / n) - val_pct).abs() * 100.0;
    let test_pct = 1.0 - train_pct - val_pct;
    let dev_test = ((f64::from(test) / n) - test_pct).abs() * 100.0;
    let max_deviation_pct = dev_train.max(dev_val).max(dev_test);
    SplitVerdict::Ok {
        train,
        val,
        test,
        max_deviation_pct,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_dataset_split_uniformity")?;

    println!("80/10/10 large: {:?}", simulate(10_000, 0.80, 0.10, 42));
    println!("70/20/10 small: {:?}", simulate(100, 0.70, 0.20, 42));
    println!("invalid: {:?}", simulate(0, 0.80, 0.10, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn large_split_low_deviation() {
        let v = simulate(10_000, 0.80, 0.10, 42);
        if let SplitVerdict::Ok {
            max_deviation_pct, ..
        } = v
        {
            assert!(max_deviation_pct < 2.0);
        }
    }

    #[test]
    fn small_split_higher_deviation() {
        let v_small = simulate(50, 0.80, 0.10, 42);
        let v_big = simulate(10_000, 0.80, 0.10, 42);
        if let (
            SplitVerdict::Ok {
                max_deviation_pct: s,
                ..
            },
            SplitVerdict::Ok {
                max_deviation_pct: b,
                ..
            },
        ) = (v_small, v_big)
        {
            assert!(s >= b);
        }
    }

    #[test]
    fn counts_sum_to_total() {
        let v = simulate(1000, 0.80, 0.10, 42);
        if let SplitVerdict::Ok {
            train, val, test, ..
        } = v
        {
            assert_eq!(train + val + test, 1000);
        }
    }

    #[test]
    fn invalid_zero_total() {
        assert_eq!(simulate(0, 0.80, 0.10, 42), SplitVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_train_plus_val_over_one() {
        assert_eq!(simulate(1000, 0.80, 0.30, 42), SplitVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_negative_pct() {
        assert_eq!(simulate(1000, -0.10, 0.30, 42), SplitVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(1000, f64::NAN, 0.30, 42),
            SplitVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(1000, 0.80, 0.10, 42);
        let b = simulate(1000, 0.80, 0.10, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn one_hundred_pct_train() {
        let v = simulate(100, 1.0, 0.0, 42);
        if let SplitVerdict::Ok {
            train, val, test, ..
        } = v
        {
            assert_eq!(train, 100);
            assert_eq!(val, 0);
            assert_eq!(test, 0);
        }
    }

    #[test]
    fn equal_thirds_works() {
        let v = simulate(10_000, 0.333, 0.333, 42);
        if let SplitVerdict::Ok {
            max_deviation_pct, ..
        } = v
        {
            assert!(max_deviation_pct < 3.0);
        }
    }
}
