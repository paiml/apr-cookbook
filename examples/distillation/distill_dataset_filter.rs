//! # Distillation Dataset Confidence Filter
//!
//! KD samples where the teacher is unsure (max softmax probability low,
//! or top-2 margin small) introduce label noise. Filter them out.
//! Defaults: min_top_prob = 0.5, min_top2_margin = 0.10.
//! Emits Keep, DropLowConfidence, or DropAmbiguous. This recipe builds
//! the filter.
//!
//! Demonstrates the **DIST.11** recipe for PMAT-137 (distillation coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Mukherjee et al. (2020). Self-Training with Weak Supervision.
//!
//! Run with: cargo run --example distill_dataset_filter
//!
//! Added by PMAT-137 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FilterVerdict {
    Keep { top_prob: f64, top2_margin: f64 },
    DropLowConfidence { top_prob: f64 },
    DropAmbiguous { top2_margin: f64 },
    InvalidProbabilities,
    InsufficientClasses,
}

const MIN_TOP_PROB: f64 = 0.5;
const MIN_TOP2_MARGIN: f64 = 0.10;

pub fn check(probs: &[f64]) -> FilterVerdict {
    if probs.len() < 2 {
        return FilterVerdict::InsufficientClasses;
    }
    if probs.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0) {
        return FilterVerdict::InvalidProbabilities;
    }
    let sum: f64 = probs.iter().sum();
    if (sum - 1.0).abs() > 1e-3 {
        return FilterVerdict::InvalidProbabilities;
    }
    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top = sorted[0];
    let second = sorted[1];
    let margin = top - second;
    if top < MIN_TOP_PROB {
        return FilterVerdict::DropLowConfidence { top_prob: top };
    }
    if margin < MIN_TOP2_MARGIN {
        return FilterVerdict::DropAmbiguous {
            top2_margin: margin,
        };
    }
    FilterVerdict::Keep {
        top_prob: top,
        top2_margin: margin,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_dataset_filter")?;

    let confident = [0.8, 0.1, 0.1];
    println!("confident: {:?}", check(&confident));

    let low_conf = [0.4, 0.3, 0.3];
    println!("low conf: {:?}", check(&low_conf));

    let ambiguous = [0.51, 0.49, 0.0];
    println!("ambiguous: {:?}", check(&ambiguous));

    let bad = [0.5, 0.6];
    println!("bad probs (sum>1): {:?}", check(&bad));

    let single = [1.0];
    println!("single class: {:?}", check(&single));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn confident_prediction_kept() {
        let v = check(&[0.8, 0.1, 0.1]);
        assert!(matches!(v, FilterVerdict::Keep { .. }));
    }

    #[test]
    fn low_confidence_dropped() {
        let v = check(&[0.4, 0.3, 0.3]);
        assert!(matches!(v, FilterVerdict::DropLowConfidence { .. }));
    }

    #[test]
    fn ambiguous_top2_dropped() {
        // Top 0.51 (passes threshold) but margin 0.02 < 0.10.
        let v = check(&[0.51, 0.49, 0.0]);
        assert!(matches!(v, FilterVerdict::DropAmbiguous { .. }));
    }

    #[test]
    fn invalid_probs_sum_rejected() {
        let v = check(&[0.5, 0.6]);
        assert_eq!(v, FilterVerdict::InvalidProbabilities);
    }

    #[test]
    fn negative_prob_rejected() {
        let v = check(&[-0.1, 1.1]);
        assert_eq!(v, FilterVerdict::InvalidProbabilities);
    }

    #[test]
    fn nan_prob_rejected() {
        let v = check(&[f64::NAN, 1.0]);
        assert_eq!(v, FilterVerdict::InvalidProbabilities);
    }

    #[test]
    fn insufficient_classes_rejected() {
        assert_eq!(check(&[1.0]), FilterVerdict::InsufficientClasses);
        assert_eq!(check(&[]), FilterVerdict::InsufficientClasses);
    }

    #[test]
    fn at_threshold_top_prob_kept() {
        // top_prob = exactly 0.5 + margin > 0.10.
        let v = check(&[0.5, 0.3, 0.2]);
        assert!(matches!(v, FilterVerdict::Keep { .. }));
    }

    #[test]
    fn just_below_top_prob_dropped() {
        // 0.49 < 0.5 → drop.
        let v = check(&[0.49, 0.30, 0.21]);
        assert!(matches!(v, FilterVerdict::DropLowConfidence { .. }));
    }

    #[test]
    fn margin_reported_in_keep() {
        if let FilterVerdict::Keep { top2_margin, .. } = check(&[0.6, 0.3, 0.1]) {
            assert!((top2_margin - 0.30).abs() < 1e-9);
        }
    }

    #[test]
    fn many_classes_only_top2_matter() {
        let probs = vec![0.6f64, 0.1, 0.1, 0.1, 0.1];
        let v = check(&probs);
        assert!(matches!(v, FilterVerdict::Keep { .. }));
    }
}
