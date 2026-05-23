//! # Monitoring Query-Pattern Analyzer
//!
//! Find common log query patterns by replacing variable parts with
//! placeholders. e.g.:
//!   "GET /users/123" → "GET /users/{n}"
//!   "GET /users/abc" → "GET /users/{s}"
//!
//! Top-K patterns by frequency identify dominant traffic shapes.
//!
//! Demonstrates the **MON.36** recipe for PMAT-154 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: log clustering / Drain log-mining algorithm.
//!
//! Run with: cargo run --example monitor_query_pattern
//!
//! Added by PMAT-154 (catalog 1009→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum PatternVerdict {
    Ok {
        top_patterns: Vec<(String, u32)>,
        total_unique: u32,
    },
    EmptyLogs,
}

pub fn analyze(log_lines: &[&str], top_k: u32) -> PatternVerdict {
    if log_lines.is_empty() {
        return PatternVerdict::EmptyLogs;
    }
    let mut counts: BTreeMap<String, u32> = BTreeMap::new();
    for line in log_lines {
        let pattern = templatize(line);
        *counts.entry(pattern).or_insert(0) += 1;
    }
    let total_unique = counts.len() as u32;
    let mut sorted: Vec<(String, u32)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let top_patterns: Vec<(String, u32)> = sorted.into_iter().take(top_k as usize).collect();
    PatternVerdict::Ok {
        top_patterns,
        total_unique,
    }
}

fn templatize(line: &str) -> String {
    line.split_whitespace()
        .map(|tok| {
            if tok.chars().all(|c| c.is_ascii_digit()) && !tok.is_empty() {
                "{n}".to_string()
            } else if tok.chars().any(|c| c.is_ascii_digit())
                && tok.chars().any(|c| c.is_ascii_alphabetic())
            {
                "{s}".to_string()
            } else {
                tok.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_query_pattern")?;

    let logs = [
        "GET /users/123 200",
        "GET /users/456 200",
        "GET /users/789 404",
        "POST /orders 201",
        "POST /orders 201",
    ];
    println!("top 3: {:?}", analyze(&logs, 3));
    println!("empty: {:?}", analyze(&[], 3));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyzer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn similar_lines_clustered() {
        let logs = ["GET /users/123", "GET /users/456", "GET /users/789"];
        let v = analyze(&logs, 5);
        if let PatternVerdict::Ok { top_patterns, .. } = v {
            // Should be one pattern: "GET /users/{n}".
            assert_eq!(top_patterns.len(), 1);
            assert_eq!(top_patterns[0].1, 3);
        }
    }

    #[test]
    fn distinct_patterns_separate() {
        let logs = ["GET /users", "POST /orders"];
        let v = analyze(&logs, 5);
        if let PatternVerdict::Ok { total_unique, .. } = v {
            assert_eq!(total_unique, 2);
        }
    }

    #[test]
    fn empty_logs_rejected() {
        assert_eq!(analyze(&[], 3), PatternVerdict::EmptyLogs);
    }

    #[test]
    fn top_k_limits_output() {
        let logs = ["a", "b", "c", "d", "e"];
        let v = analyze(&logs, 2);
        if let PatternVerdict::Ok { top_patterns, .. } = v {
            assert_eq!(top_patterns.len(), 2);
        }
    }

    #[test]
    fn most_frequent_first() {
        let logs = ["A", "A", "A", "B", "B", "C"];
        let v = analyze(&logs, 3);
        if let PatternVerdict::Ok { top_patterns, .. } = v {
            assert_eq!(top_patterns[0].0, "A");
            assert_eq!(top_patterns[0].1, 3);
        }
    }

    #[test]
    fn alphanumeric_token_replaced_with_s() {
        let logs = ["GET /users/abc123"];
        let v = analyze(&logs, 1);
        if let PatternVerdict::Ok { top_patterns, .. } = v {
            assert!(top_patterns[0].0.contains("{s}"));
        }
    }

    #[test]
    fn pure_digit_replaced_with_n() {
        let logs = ["count 12345"];
        let v = analyze(&logs, 1);
        if let PatternVerdict::Ok { top_patterns, .. } = v {
            assert!(top_patterns[0].0.contains("{n}"));
        }
    }

    #[test]
    fn pure_alpha_unchanged() {
        let logs = ["hello world"];
        let v = analyze(&logs, 1);
        if let PatternVerdict::Ok { top_patterns, .. } = v {
            assert!(top_patterns[0].0.contains("hello"));
            assert!(top_patterns[0].0.contains("world"));
        }
    }

    #[test]
    fn total_unique_count_correct() {
        let logs = ["A", "B", "A", "B", "C"];
        let v = analyze(&logs, 10);
        if let PatternVerdict::Ok { total_unique, .. } = v {
            assert_eq!(total_unique, 3);
        }
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let logs = ["A", "B"];
        let v = analyze(&logs, 0);
        if let PatternVerdict::Ok { top_patterns, .. } = v {
            assert!(top_patterns.is_empty());
        }
    }
}
