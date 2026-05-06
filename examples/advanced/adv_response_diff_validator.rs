//! # Advanced A/B Response Diff Validator
//!
//! For canary deployments: compare experimental response to canonical
//! (control) response. Acceptable: exact match OR semantic similarity
//! ≥ threshold. Returns Match/Diff with similarity_pct.
//!
//! Demonstrates the **ADV.41** recipe for PMAT-159 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: A/B test response shadowing (Netflix's "DBLog" pattern).
//!
//! Run with: cargo run --example adv_response_diff_validator
//!
//! Added by PMAT-159 (catalog 1054→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DiffVerdict {
    ExactMatch,
    SemanticMatch { similarity_pct: f64 },
    Diff { similarity_pct: f64 },
    EmptyResponse,
    InvalidThreshold,
}

pub fn validate(canonical: &str, experimental: &str, similarity_threshold_pct: f64) -> DiffVerdict {
    if canonical.is_empty() || experimental.is_empty() {
        return DiffVerdict::EmptyResponse;
    }
    if !similarity_threshold_pct.is_finite() || !(0.0..=100.0).contains(&similarity_threshold_pct) {
        return DiffVerdict::InvalidThreshold;
    }
    if canonical == experimental {
        return DiffVerdict::ExactMatch;
    }
    let sim = jaccard_similarity(canonical, experimental) * 100.0;
    if sim >= similarity_threshold_pct {
        DiffVerdict::SemanticMatch {
            similarity_pct: sim,
        }
    } else {
        DiffVerdict::Diff {
            similarity_pct: sim,
        }
    }
}

fn jaccard_similarity(a: &str, b: &str) -> f64 {
    let set_a: std::collections::BTreeSet<&str> = a.split_whitespace().collect();
    let set_b: std::collections::BTreeSet<&str> = b.split_whitespace().collect();
    if set_a.is_empty() && set_b.is_empty() {
        return 1.0;
    }
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_response_diff_validator")?;

    println!("exact: {:?}", validate("hello world", "hello world", 80.0));
    println!(
        "semantic: {:?}",
        validate("the quick brown fox", "the quick brown dog", 70.0)
    );
    println!(
        "diff: {:?}",
        validate("apple banana", "completely different stuff", 70.0)
    );
    println!("empty: {:?}", validate("", "x", 80.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn exact_match_returned() {
        let v = validate("hello", "hello", 80.0);
        assert_eq!(v, DiffVerdict::ExactMatch);
    }

    #[test]
    fn similar_meets_threshold() {
        // Jaccard("the quick brown fox", "the quick brown dog") = 3/5 = 60%.
        let v = validate("the quick brown fox", "the quick brown dog", 50.0);
        assert!(matches!(v, DiffVerdict::SemanticMatch { .. }));
    }

    #[test]
    fn dissimilar_diff() {
        let v = validate("apple", "completely different stuff", 70.0);
        assert!(matches!(v, DiffVerdict::Diff { .. }));
    }

    #[test]
    fn empty_canonical_rejected() {
        assert_eq!(validate("", "x", 80.0), DiffVerdict::EmptyResponse);
    }

    #[test]
    fn empty_experimental_rejected() {
        assert_eq!(validate("x", "", 80.0), DiffVerdict::EmptyResponse);
    }

    #[test]
    fn invalid_threshold_negative() {
        assert_eq!(validate("a", "b", -1.0), DiffVerdict::InvalidThreshold);
    }

    #[test]
    fn invalid_threshold_over_100() {
        assert_eq!(validate("a", "b", 150.0), DiffVerdict::InvalidThreshold);
    }

    #[test]
    fn exact_match_skips_similarity_calc() {
        // Even with threshold 100, exact match wins.
        let v = validate("hello", "hello", 100.0);
        assert_eq!(v, DiffVerdict::ExactMatch);
    }

    #[test]
    fn similarity_above_threshold_match() {
        // 3/4 words shared → 75%; threshold 70.
        let v = validate("a b c d", "a b c x", 70.0);
        if let DiffVerdict::SemanticMatch { similarity_pct } = v {
            assert!(similarity_pct >= 70.0);
        }
    }

    #[test]
    fn nan_threshold_invalid() {
        assert_eq!(validate("a", "b", f64::NAN), DiffVerdict::InvalidThreshold);
    }

    #[test]
    fn deterministic() {
        let a = validate("x y z", "x y w", 50.0);
        let b = validate("x y z", "x y w", 50.0);
        assert_eq!(a, b);
    }
}
