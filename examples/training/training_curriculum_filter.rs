//! # Training Curriculum Difficulty Filter
//!
//! Curriculum learning: feed easy examples first, hard ones later.
//! Filter samples by difficulty score percentile thresholds matching
//! current training progress (e.g., progress=0.0 → easiest 25%;
//! progress=1.0 → all). This recipe builds the per-step filter.
//!
//! Demonstrates the **TRAIN.15** recipe for PMAT-132 (training coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Bengio et al. (2009). Curriculum Learning. ICML.
//!
//! Run with: cargo run --example training_curriculum_filter
//!
//! Added by PMAT-132 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FilterVerdict {
    Ok {
        kept_indices: Vec<usize>,
        threshold: f64,
    },
    EmptyDataset,
    InvalidProgress,
    InvalidScore {
        at_index: usize,
    },
}

const MIN_FRACTION: f64 = 0.25;

pub fn filter(scores: &[f64], progress: f64) -> FilterVerdict {
    if scores.is_empty() {
        return FilterVerdict::EmptyDataset;
    }
    if !progress.is_finite() || !(0.0..=1.0).contains(&progress) {
        return FilterVerdict::InvalidProgress;
    }
    for (i, &s) in scores.iter().enumerate() {
        if !s.is_finite() {
            return FilterVerdict::InvalidScore { at_index: i };
        }
    }
    let fraction = MIN_FRACTION + (1.0 - MIN_FRACTION) * progress;
    let mut sorted: Vec<f64> = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff_idx = ((sorted.len() as f64) * fraction).round() as usize;
    let cutoff_idx = cutoff_idx.max(1).min(sorted.len());
    let threshold = sorted[cutoff_idx - 1];
    let kept_indices: Vec<usize> = scores
        .iter()
        .enumerate()
        .filter(|(_, s)| **s <= threshold)
        .map(|(i, _)| i)
        .collect();
    FilterVerdict::Ok {
        kept_indices,
        threshold,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("training_curriculum_filter")?;

    let scores = [0.1, 0.5, 0.3, 0.9, 0.7, 0.2, 0.6, 0.8];
    for prog in [0.0, 0.5, 1.0] {
        println!("progress={prog}  →  {:?}", filter(&scores, prog));
    }
    println!("invalid: {:?}", filter(&scores, 1.5));
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
    fn early_progress_keeps_easiest_quarter() {
        let scores: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        // 8 scores, MIN_FRACTION=0.25 → 2 easiest.
        if let FilterVerdict::Ok { kept_indices, .. } = filter(&scores, 0.0) {
            assert_eq!(kept_indices.len(), 2);
        }
    }

    #[test]
    fn full_progress_keeps_all() {
        let scores: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        if let FilterVerdict::Ok { kept_indices, .. } = filter(&scores, 1.0) {
            assert_eq!(kept_indices.len(), 8);
        }
    }

    #[test]
    fn mid_progress_keeps_proportional_fraction() {
        let scores: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        // progress=0.5 → frac = 0.25 + 0.75 × 0.5 = 0.625 → keep 6.
        if let FilterVerdict::Ok { kept_indices, .. } = filter(&scores, 0.5) {
            assert_eq!(kept_indices.len(), 6);
        }
    }

    #[test]
    fn empty_dataset_rejected() {
        assert_eq!(filter(&[], 0.5), FilterVerdict::EmptyDataset);
    }

    #[test]
    fn out_of_range_progress_rejected() {
        let scores = [0.1, 0.5];
        assert_eq!(filter(&scores, 1.5), FilterVerdict::InvalidProgress);
        assert_eq!(filter(&scores, -0.1), FilterVerdict::InvalidProgress);
    }

    #[test]
    fn nan_progress_rejected() {
        let scores = [0.1];
        assert_eq!(filter(&scores, f64::NAN), FilterVerdict::InvalidProgress);
    }

    #[test]
    fn nan_score_rejected() {
        let scores = [0.1, f64::NAN, 0.5];
        let v = filter(&scores, 0.5);
        assert!(matches!(v, FilterVerdict::InvalidScore { at_index: 1 }));
    }

    #[test]
    fn unsorted_input_filters_correctly() {
        // Same dataset, just shuffled — kept count should match.
        let a = [0.1, 0.5, 0.3, 0.9, 0.7, 0.2, 0.6, 0.8];
        let b = [0.9, 0.1, 0.5, 0.7, 0.2, 0.3, 0.8, 0.6];
        let va = filter(&a, 0.0);
        let vb = filter(&b, 0.0);
        if let (
            FilterVerdict::Ok {
                kept_indices: ka, ..
            },
            FilterVerdict::Ok {
                kept_indices: kb, ..
            },
        ) = (va, vb)
        {
            assert_eq!(ka.len(), kb.len());
        }
    }

    #[test]
    fn progress_zero_keeps_at_least_one() {
        let scores = [42.0];
        if let FilterVerdict::Ok { kept_indices, .. } = filter(&scores, 0.0) {
            assert!(!kept_indices.is_empty());
        }
    }

    #[test]
    fn threshold_monotonically_grows_with_progress() {
        let scores = [0.1, 0.5, 0.3, 0.9, 0.7, 0.2, 0.6, 0.8];
        let early = filter(&scores, 0.0);
        let late = filter(&scores, 0.9);
        if let (FilterVerdict::Ok { threshold: e, .. }, FilterVerdict::Ok { threshold: l, .. }) =
            (early, late)
        {
            assert!(l >= e);
        }
    }
}
