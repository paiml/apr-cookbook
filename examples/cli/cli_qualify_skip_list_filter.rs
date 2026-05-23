//! # apr qualify — `--skip` Comma-Separated Gate Filter
//!
//! `apr qualify <FILE> --skip gate1,gate2` removes specific gates from
//! the tier-determined list. Skip names must match exactly (typos silently
//! skip nothing — the CLI surfaces unknown skip names as a warning so
//! the operator catches misspellings). This recipe builds the parser
//! and asserts the contract.
//!
//! Demonstrates the **QUALIFY.4** recipe for PMAT-094 (apr qualify coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender QUALIFY-002
//!
//! Run with: cargo run --example cli_qualify_skip_list_filter
//!
//! Added by PMAT-094 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashSet;

#[derive(Debug, PartialEq, Eq)]
pub struct SkipApplied {
    pub kept: Vec<&'static str>,
    pub unknown_skips: Vec<String>,
}

pub fn parse_skip_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

pub fn apply_skips(gates: &[&'static str], skips: &[String]) -> SkipApplied {
    let known: HashSet<&str> = gates.iter().copied().collect();
    let unknown: Vec<String> = skips
        .iter()
        .filter(|s| !known.contains(s.as_str()))
        .cloned()
        .collect();
    let drop: HashSet<&str> = skips.iter().map(String::as_str).collect();
    let kept: Vec<&'static str> = gates
        .iter()
        .copied()
        .filter(|g| !drop.contains(g))
        .collect();
    SkipApplied {
        kept,
        unknown_skips: unknown,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_qualify_skip_list_filter")?;

    let gates = [
        "info", "tensors", "tree", "validate", "check", "bench", "qa",
    ];
    println!("=== Recipe: cli_qualify_skip_list_filter ===");

    for raw in [
        "",
        "bench",
        "bench,qa",
        "bench, qa,    info",
        "bench,banana",
    ] {
        let skips = parse_skip_list(raw);
        let applied = apply_skips(&gates, &skips);
        println!(
            "--skip {raw:>22}  →  kept={:?}  unknown={:?}",
            applied.kept, applied.unknown_skips
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_gates() -> Vec<&'static str> {
        vec![
            "info", "tensors", "tree", "validate", "check", "bench", "qa",
        ]
    }

    #[test]
    fn skip_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_skip_list_keeps_all_gates() {
        let g = sample_gates();
        let applied = apply_skips(&g, &[]);
        assert_eq!(applied.kept, g);
        assert!(applied.unknown_skips.is_empty());
    }

    #[test]
    fn parser_handles_whitespace_and_empties() {
        let v = parse_skip_list("a, b ,, c");
        assert_eq!(v, vec!["a".to_string(), "b".into(), "c".into()]);
    }

    #[test]
    fn known_skip_removes_matching_gate() {
        let g = sample_gates();
        let skips = parse_skip_list("bench,qa");
        let applied = apply_skips(&g, &skips);
        assert!(!applied.kept.contains(&"bench"));
        assert!(!applied.kept.contains(&"qa"));
        assert_eq!(applied.kept.len(), 5);
    }

    #[test]
    fn unknown_skip_surfaced_in_warning() {
        // Critical: typos like "benchmrk" would silently skip nothing.
        // The CLI must surface this so the operator catches the typo.
        let g = sample_gates();
        let skips = parse_skip_list("benchmrk,qa");
        let applied = apply_skips(&g, &skips);
        assert_eq!(applied.unknown_skips, vec!["benchmrk".to_string()]);
        // qa was a real skip and was applied
        assert!(!applied.kept.contains(&"qa"));
    }

    #[test]
    fn skip_list_is_case_sensitive() {
        // Gate names are canonical lowercase; "Bench" must NOT skip "bench".
        let g = sample_gates();
        let skips = parse_skip_list("Bench");
        let applied = apply_skips(&g, &skips);
        assert!(applied.kept.contains(&"bench"));
        assert_eq!(applied.unknown_skips, vec!["Bench".to_string()]);
    }

    #[test]
    fn duplicate_skip_idempotent() {
        // "bench,bench,bench" should still keep the same gates as "bench".
        let g = sample_gates();
        let skips = parse_skip_list("bench,bench,bench");
        let applied = apply_skips(&g, &skips);
        assert!(!applied.kept.contains(&"bench"));
        assert_eq!(applied.kept.len(), 6);
    }
}
