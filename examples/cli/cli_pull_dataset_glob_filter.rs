//! # apr pull — Dataset Mode `--include <GLOB>` Filter
//!
//! `apr pull dataset <REPO> --include <PATTERN>` selects a subset of
//! shards from a HuggingFace dataset using fnmatch-compatible globs.
//! Multiple `--include` flags are unioned. Per the spec, no-match is
//! fail-fast (operator probably misnamed the shard family). This recipe
//! builds the include-filter and asserts the contract.
//!
//! Demonstrates the **PULL.5** recipe for PMAT-101 (apr pull coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender apr-cli-pull-dataset-v1.yaml + fnmatch (PEP 380)
//!
//! Run with: cargo run --example cli_pull_dataset_glob_filter
//!
//! Added by PMAT-101 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FilterVerdict<'a> {
    Selected(Vec<&'a str>),
    NoMatch { patterns: Vec<String> },
}

pub fn fnmatch(pattern: &str, name: &str) -> bool {
    // Simple two-state matcher supporting '*' and '?'. Sufficient for the
    // cookbook documentation contract.
    let p: Vec<char> = pattern.chars().collect();
    let n: Vec<char> = name.chars().collect();
    let mut dp = vec![vec![false; n.len() + 1]; p.len() + 1];
    dp[0][0] = true;
    for i in 1..=p.len() {
        if p[i - 1] == '*' {
            dp[i][0] = dp[i - 1][0];
        }
    }
    for i in 1..=p.len() {
        for j in 1..=n.len() {
            dp[i][j] = match p[i - 1] {
                '*' => dp[i - 1][j] || dp[i][j - 1],
                '?' => dp[i - 1][j - 1],
                c if c == n[j - 1] => dp[i - 1][j - 1],
                _ => false,
            };
        }
    }
    dp[p.len()][n.len()]
}

pub fn select_shards<'a>(shards: &[&'a str], includes: &[&str]) -> FilterVerdict<'a> {
    let selected: Vec<&'a str> = shards
        .iter()
        .copied()
        .filter(|s| includes.iter().any(|p| fnmatch(p, s)))
        .collect();
    if selected.is_empty() {
        FilterVerdict::NoMatch {
            patterns: includes.iter().map(|s| (*s).to_string()).collect(),
        }
    } else {
        FilterVerdict::Selected(selected)
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_pull_dataset_glob_filter")?;

    let shards = [
        "train-00000-of-00010.parquet",
        "train-00001-of-00010.parquet",
        "train-00009-of-00010.parquet",
        "test-00000-of-00001.parquet",
        "validation-00000-of-00001.parquet",
        "README.md",
    ];

    for includes in [
        vec!["train-*.parquet"],
        vec!["train-?????-of-*.parquet"],
        vec!["train-*.parquet", "test-*.parquet"],
        vec!["nonexistent-*.parquet"],
    ] {
        println!("--include {includes:?}");
        println!("  → {:?}", select_shards(&shards, &includes));
    }
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
    fn star_wildcard_matches() {
        assert!(fnmatch("train-*.parquet", "train-00000-of-00010.parquet"));
        assert!(fnmatch("*", "anything"));
    }

    #[test]
    fn question_mark_matches_single_char() {
        assert!(fnmatch("a?c", "abc"));
        assert!(!fnmatch("a?c", "abbc"));
    }

    #[test]
    fn literal_chars_must_match_exactly() {
        assert!(!fnmatch("train-*.parquet", "test-00000-of-00010.parquet"));
    }

    #[test]
    fn no_match_yields_fail_fast_verdict() {
        let shards = ["train-00000.parquet"];
        let v = select_shards(&shards, &["test-*.parquet"]);
        assert!(matches!(v, FilterVerdict::NoMatch { .. }));
    }

    #[test]
    fn multiple_includes_are_unioned() {
        let shards = [
            "train-001.parquet",
            "test-001.parquet",
            "validation-001.parquet",
        ];
        let v = select_shards(&shards, &["train-*.parquet", "test-*.parquet"]);
        if let FilterVerdict::Selected(s) = v {
            assert_eq!(s.len(), 2);
        } else {
            panic!("expected Selected");
        }
    }

    #[test]
    fn five_question_marks_match_5_digit_index() {
        let shards = [
            "train-00000-of-00010.parquet",
            "train-00010-of-00100.parquet",
        ];
        let v = select_shards(&shards, &["train-?????-of-?????.parquet"]);
        if let FilterVerdict::Selected(s) = v {
            assert_eq!(s.len(), 2);
        } else {
            panic!("expected Selected");
        }
    }

    #[test]
    fn empty_includes_yields_no_match() {
        let shards = ["train-001.parquet"];
        let v = select_shards(&shards, &[]);
        assert!(matches!(v, FilterVerdict::NoMatch { .. }));
    }
}
