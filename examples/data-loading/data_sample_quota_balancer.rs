//! # Data Sample Quota Balancer
//!
//! Imbalanced datasets (e.g., 100 cat / 5 dog) bias the model toward
//! the majority class. Quota balancer: compute per-class weights so
//! effective sample count is uniform. weight[c] = max_count / count[c].
//! Cap at MAX_WEIGHT to avoid overfit on tiny minorities. This recipe
//! builds the balancer.
//!
//! Demonstrates the **DATA.21** recipe for PMAT-135 (data-loading coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: scikit-learn class_weight="balanced" formula.
//!
//! Run with: cargo run --example data_sample_quota_balancer
//!
//! Added by PMAT-135 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

const MAX_WEIGHT: f64 = 100.0;

#[derive(Debug, PartialEq)]
pub enum BalanceVerdict {
    Ok {
        weights: BTreeMap<String, f64>,
        max_weight_clipped: bool,
    },
    EmptyClasses,
    AllZeroCounts,
}

pub fn balance(class_counts: &BTreeMap<String, u64>) -> BalanceVerdict {
    if class_counts.is_empty() {
        return BalanceVerdict::EmptyClasses;
    }
    let max_count = class_counts.values().max().copied().unwrap_or(0);
    if max_count == 0 {
        return BalanceVerdict::AllZeroCounts;
    }
    let mut weights = BTreeMap::new();
    let mut clipped = false;
    for (class, count) in class_counts {
        let raw_weight = if *count == 0 {
            MAX_WEIGHT
        } else {
            max_count as f64 / *count as f64
        };
        let final_weight = if raw_weight > MAX_WEIGHT {
            clipped = true;
            MAX_WEIGHT
        } else {
            raw_weight
        };
        weights.insert(class.clone(), final_weight);
    }
    BalanceVerdict::Ok {
        weights,
        max_weight_clipped: clipped,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("data_sample_quota_balancer")?;

    let mut counts = BTreeMap::new();
    counts.insert("cat".to_string(), 100u64);
    counts.insert("dog".to_string(), 5);
    println!("cat=100 dog=5: {:?}", balance(&counts));

    let mut tiny = BTreeMap::new();
    tiny.insert("rare".to_string(), 1u64);
    tiny.insert("common".to_string(), 10_000);
    println!("rare=1 common=10000: {:?}", balance(&tiny));

    println!("empty: {:?}", balance(&BTreeMap::new()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts_of(pairs: &[(&str, u64)]) -> BTreeMap<String, u64> {
        let mut m = BTreeMap::new();
        for (k, v) in pairs {
            m.insert((*k).to_string(), *v);
        }
        m
    }

    #[test]
    fn balancer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn balanced_dataset_weights_one() {
        let c = counts_of(&[("a", 100), ("b", 100)]);
        if let BalanceVerdict::Ok { weights, .. } = balance(&c) {
            assert!((weights["a"] - 1.0).abs() < 1e-9);
            assert!((weights["b"] - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn imbalanced_minority_upweighted() {
        let c = counts_of(&[("cat", 100), ("dog", 5)]);
        if let BalanceVerdict::Ok { weights, .. } = balance(&c) {
            assert!((weights["cat"] - 1.0).abs() < 1e-9);
            assert!((weights["dog"] - 20.0).abs() < 1e-9);
        }
    }

    #[test]
    fn extreme_imbalance_clipped() {
        let c = counts_of(&[("rare", 1), ("common", 10_000)]);
        if let BalanceVerdict::Ok {
            weights,
            max_weight_clipped,
        } = balance(&c)
        {
            assert!(max_weight_clipped);
            assert_eq!(weights["rare"], MAX_WEIGHT);
        }
    }

    #[test]
    fn empty_classes_rejected() {
        let c = BTreeMap::new();
        assert_eq!(balance(&c), BalanceVerdict::EmptyClasses);
    }

    #[test]
    fn all_zero_counts_rejected() {
        let c = counts_of(&[("a", 0), ("b", 0)]);
        assert_eq!(balance(&c), BalanceVerdict::AllZeroCounts);
    }

    #[test]
    fn zero_count_class_gets_max_weight() {
        // One class is zero, other is non-zero.
        let c = counts_of(&[("present", 100), ("missing", 0)]);
        if let BalanceVerdict::Ok { weights, .. } = balance(&c) {
            assert_eq!(weights["missing"], MAX_WEIGHT);
        }
    }

    #[test]
    fn three_class_typical() {
        let c = counts_of(&[("a", 100), ("b", 50), ("c", 25)]);
        if let BalanceVerdict::Ok { weights, .. } = balance(&c) {
            assert!((weights["a"] - 1.0).abs() < 1e-9);
            assert!((weights["b"] - 2.0).abs() < 1e-9);
            assert!((weights["c"] - 4.0).abs() < 1e-9);
        }
    }

    #[test]
    fn no_clip_for_moderate_imbalance() {
        let c = counts_of(&[("a", 100), ("b", 10)]);
        if let BalanceVerdict::Ok {
            max_weight_clipped, ..
        } = balance(&c)
        {
            assert!(!max_weight_clipped);
        }
    }

    #[test]
    fn weights_sum_proportional_to_class_count() {
        // Each class effective count = count × weight; should be uniform.
        let c = counts_of(&[("a", 80), ("b", 20)]);
        if let BalanceVerdict::Ok { weights, .. } = balance(&c) {
            let eff_a = 80.0 * weights["a"];
            let eff_b = 20.0 * weights["b"];
            assert!((eff_a - eff_b).abs() < 1e-9);
        }
    }

    #[test]
    fn single_class_weight_one() {
        let c = counts_of(&[("only", 50)]);
        if let BalanceVerdict::Ok { weights, .. } = balance(&c) {
            assert!((weights["only"] - 1.0).abs() < 1e-9);
        }
    }
}
