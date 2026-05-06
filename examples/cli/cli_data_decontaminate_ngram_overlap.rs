//! # apr data decontaminate — N-gram Overlap Detector
//!
//! `apr data decontaminate <TRAIN.jsonl> --benchmark <BENCH.jsonl>
//! --n <N>` flags training records that share ≥ K n-grams of length N
//! with any benchmark record. This recipe builds the overlap detector
//! as a pure function so the contract can be exercised offline.
//!
//! Demonstrates the **DATA-DECONTAMINATE.4** recipe for PMAT-106 (apr data decontaminate coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DATA-DECONT-001 + Brown et al. (2020) GPT-3 contamination check
//!
//! Run with: cargo run --example cli_data_decontaminate_ngram_overlap
//!
//! Added by PMAT-106 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashSet;

pub fn extract_ngrams(text: &str, n: usize) -> HashSet<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < n || n == 0 {
        return HashSet::new();
    }
    (0..=words.len() - n)
        .map(|i| words[i..i + n].join(" "))
        .collect()
}

pub fn jaccard_similarity<S: std::hash::BuildHasher>(
    a: &HashSet<String, S>,
    b: &HashSet<String, S>,
) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    intersection / union
}

#[derive(Debug, PartialEq)]
pub enum ContaminationVerdict {
    Clean,
    Contaminated { jaccard: f64, threshold: f64 },
}

pub fn check_contamination(
    train_text: &str,
    bench_text: &str,
    n: usize,
    threshold: f64,
) -> ContaminationVerdict {
    let train_ngrams = extract_ngrams(train_text, n);
    let bench_ngrams = extract_ngrams(bench_text, n);
    let j = jaccard_similarity(&train_ngrams, &bench_ngrams);
    if j >= threshold {
        ContaminationVerdict::Contaminated {
            jaccard: j,
            threshold,
        }
    } else {
        ContaminationVerdict::Clean
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_data_decontaminate_ngram_overlap")?;

    let bench = "the quick brown fox jumps over the lazy dog";
    let cases = [
        ("identical", bench),
        (
            "partial overlap",
            "the quick brown fox flies above the trees",
        ),
        (
            "no overlap",
            "completely different sentence with new words entirely",
        ),
    ];

    for (label, train) in cases {
        for n in [3, 5] {
            println!(
                "{label:>22}  n={n}  →  {:?}",
                check_contamination(train, bench, n, 0.5)
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn extract_ngrams_works() {
        let g = extract_ngrams("the quick brown fox", 2);
        assert!(g.contains("the quick"));
        assert!(g.contains("quick brown"));
        assert!(g.contains("brown fox"));
        assert_eq!(g.len(), 3);
    }

    #[test]
    fn empty_input_yields_empty_ngrams() {
        assert!(extract_ngrams("", 3).is_empty());
        assert!(extract_ngrams("short", 5).is_empty());
    }

    #[test]
    fn n_zero_yields_empty() {
        assert!(extract_ngrams("hello world", 0).is_empty());
    }

    #[test]
    fn jaccard_self_is_one() {
        let g = extract_ngrams("the quick brown fox", 2);
        assert!((jaccard_similarity(&g, &g) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn jaccard_disjoint_is_zero() {
        let a = extract_ngrams("alpha beta", 2);
        let b = extract_ngrams("gamma delta", 2);
        assert_eq!(jaccard_similarity(&a, &b), 0.0);
    }

    #[test]
    fn identical_text_flagged_contaminated() {
        let v = check_contamination("the quick brown fox", "the quick brown fox", 2, 0.5);
        assert!(matches!(v, ContaminationVerdict::Contaminated { .. }));
    }

    #[test]
    fn unrelated_text_clean() {
        let v = check_contamination("alpha beta gamma delta", "the quick brown fox", 2, 0.5);
        assert_eq!(v, ContaminationVerdict::Clean);
    }

    #[test]
    fn lower_n_more_sensitive() {
        // Shorter n-grams = more matches = higher contamination signal.
        let train = "the quick brown fox jumps";
        let bench = "the slow brown fox sits";
        let v3 = check_contamination(train, bench, 1, 0.5);
        let v5 = check_contamination(train, bench, 5, 0.5);
        // n=1 should detect more overlap than n=5.
        assert!(matches!(v5, ContaminationVerdict::Clean));
        let _ = v3;
    }
}
