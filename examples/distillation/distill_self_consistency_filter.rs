//! # Distillation Self-Consistency Filter
//!
//! Sample teacher N times for each input. Keep only inputs where
//! majority answer occurs ≥ K times (high agreement = high quality).
//!
//! Strategy: take majority vote per input; reject if vote count < K.
//!
//! Demonstrates the **DIST.21** recipe for PMAT-149 (distillation round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Wang et al. (2022). Self-Consistency Improves CoT Reasoning.
//!
//! Run with: cargo run --example distill_self_consistency_filter
//!
//! Added by PMAT-149 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
pub enum FilterVerdict {
    Keep {
        majority_answer: String,
        vote_count: u32,
        agreement_pct: u32,
    },
    Reject {
        max_votes: u32,
        threshold: u32,
    },
    EmptySamples,
    InvalidThreshold,
}

pub fn filter(samples: &[&str], min_votes: u32) -> FilterVerdict {
    if samples.is_empty() {
        return FilterVerdict::EmptySamples;
    }
    if min_votes == 0 {
        return FilterVerdict::InvalidThreshold;
    }
    let mut counts: HashMap<&str, u32> = HashMap::new();
    for &s in samples {
        *counts.entry(s).or_insert(0) += 1;
    }
    let (top_answer, top_votes) = counts.iter().max_by_key(|(_, &v)| v).unwrap();
    let total = samples.len() as u32;
    if *top_votes < min_votes {
        return FilterVerdict::Reject {
            max_votes: *top_votes,
            threshold: min_votes,
        };
    }
    FilterVerdict::Keep {
        majority_answer: (*top_answer).to_string(),
        vote_count: *top_votes,
        agreement_pct: (top_votes * 100) / total,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_self_consistency_filter")?;

    println!(
        "high agreement: {:?}",
        filter(&["yes", "yes", "yes", "yes", "no"], 3)
    );
    println!(
        "split vote: {:?}",
        filter(&["yes", "no", "maybe", "yes", "no"], 3)
    );
    println!("empty: {:?}", filter(&[], 3));
    println!("invalid threshold: {:?}", filter(&["a"], 0));
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
    fn high_agreement_kept() {
        let v = filter(&["yes", "yes", "yes", "yes", "no"], 3);
        assert!(matches!(v, FilterVerdict::Keep { .. }));
    }

    #[test]
    fn low_agreement_rejected() {
        let v = filter(&["yes", "no", "maybe", "huh", "different"], 3);
        assert!(matches!(v, FilterVerdict::Reject { .. }));
    }

    #[test]
    fn unanimous_100_pct() {
        let v = filter(&["x", "x", "x"], 2);
        if let FilterVerdict::Keep { agreement_pct, .. } = v {
            assert_eq!(agreement_pct, 100);
        }
    }

    #[test]
    fn majority_answer_returned() {
        let v = filter(&["a", "b", "a", "c", "a"], 2);
        if let FilterVerdict::Keep {
            majority_answer, ..
        } = v
        {
            assert_eq!(majority_answer, "a");
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(filter(&[], 3), FilterVerdict::EmptySamples);
    }

    #[test]
    fn invalid_threshold_rejected() {
        assert_eq!(filter(&["a"], 0), FilterVerdict::InvalidThreshold);
    }

    #[test]
    fn vote_count_correct() {
        let v = filter(&["yes", "yes", "yes", "no"], 2);
        if let FilterVerdict::Keep { vote_count, .. } = v {
            assert_eq!(vote_count, 3);
        }
    }

    #[test]
    fn at_threshold_kept() {
        // Exactly min_votes → keep.
        let v = filter(&["a", "a", "b", "c"], 2);
        if let FilterVerdict::Keep { vote_count, .. } = v {
            assert_eq!(vote_count, 2);
        }
    }

    #[test]
    fn just_below_threshold_rejected() {
        let v = filter(&["a", "b", "c"], 2);
        assert!(matches!(v, FilterVerdict::Reject { .. }));
    }

    #[test]
    fn single_sample_one_vote() {
        let v = filter(&["only"], 1);
        if let FilterVerdict::Keep { vote_count, .. } = v {
            assert_eq!(vote_count, 1);
        }
    }
}
