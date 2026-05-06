//! # apr debug --layer — Glob Matcher
//!
//! `apr debug --layer "model.layers.*.attn.*"` selects layers via glob.
//! Supports `*` (any single segment, no dots) and `**` (any subpath
//! including dots). Anchored at start; suffix-flexible. This recipe
//! builds the matcher.
//!
//! Demonstrates the **DBG.4** recipe for PMAT-117 (apr debug coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DBG-001 + glob(7) conventions
//!
//! Run with: cargo run --example cli_debug_layer_glob_matcher
//!
//! Added by PMAT-117 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MatchVerdict {
    Matched,
    NotMatched,
    InvalidPattern,
}

pub fn matches_glob(pattern: &str, name: &str) -> MatchVerdict {
    if pattern.is_empty() {
        return MatchVerdict::InvalidPattern;
    }
    if pattern.contains("***") {
        return MatchVerdict::InvalidPattern;
    }
    if matches_inner(pattern, name) {
        MatchVerdict::Matched
    } else {
        MatchVerdict::NotMatched
    }
}

fn matches_inner(pattern: &str, name: &str) -> bool {
    let p_chars: Vec<char> = pattern.chars().collect();
    let n_chars: Vec<char> = name.chars().collect();
    matches_helper(&p_chars, 0, &n_chars, 0)
}

fn matches_helper(p: &[char], pi: usize, n: &[char], ni: usize) -> bool {
    if pi == p.len() {
        return ni == n.len();
    }
    if pi + 1 < p.len() && p[pi] == '*' && p[pi + 1] == '*' {
        // Zero-or-more chars including dots.
        let next_pi = pi + 2;
        for end in ni..=n.len() {
            if matches_helper(p, next_pi, n, end) {
                return true;
            }
        }
        return false;
    }
    if p[pi] == '*' {
        // Zero-or-more non-dot chars.
        let next_pi = pi + 1;
        for end in ni..=n.len() {
            if n[ni..end].contains(&'.') {
                return false;
            }
            if matches_helper(p, next_pi, n, end) {
                return true;
            }
        }
        return false;
    }
    if ni < n.len() && p[pi] == n[ni] {
        return matches_helper(p, pi + 1, n, ni + 1);
    }
    false
}

pub fn select_matching<'a>(pattern: &str, names: &[&'a str]) -> Vec<&'a str> {
    names
        .iter()
        .copied()
        .filter(|n| matches!(matches_glob(pattern, n), MatchVerdict::Matched))
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_debug_layer_glob_matcher")?;

    let names = [
        "model.layers.0.attn.q_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.5.attn.v_proj",
        "model.embed_tokens.weight",
    ];
    println!(
        "attn-only: {:?}",
        select_matching("model.layers.*.attn.*", &names)
    );
    println!(
        "all layers: {:?}",
        select_matching("model.layers.**", &names)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn exact_match() {
        assert_eq!(matches_glob("foo.bar", "foo.bar"), MatchVerdict::Matched);
    }

    #[test]
    fn star_matches_single_segment() {
        assert_eq!(
            matches_glob("foo.*.bar", "foo.123.bar"),
            MatchVerdict::Matched
        );
    }

    #[test]
    fn star_does_not_match_dot() {
        // `*` is non-dot — should not match across segment boundary.
        assert_eq!(
            matches_glob("foo.*", "foo.bar.baz"),
            MatchVerdict::NotMatched
        );
    }

    #[test]
    fn double_star_matches_subpath() {
        // `**` matches across dots.
        assert_eq!(
            matches_glob("foo.**", "foo.bar.baz.qux"),
            MatchVerdict::Matched
        );
    }

    #[test]
    fn empty_pattern_invalid() {
        assert_eq!(matches_glob("", "anything"), MatchVerdict::InvalidPattern);
    }

    #[test]
    fn triple_star_invalid() {
        assert_eq!(
            matches_glob("foo.***.bar", "anything"),
            MatchVerdict::InvalidPattern
        );
    }

    #[test]
    fn no_match_returns_not_matched() {
        assert_eq!(matches_glob("foo.bar", "qux.bar"), MatchVerdict::NotMatched);
    }

    #[test]
    fn select_filters_to_matching_only() {
        let names = ["model.layers.0.attn", "model.layers.0.mlp", "embed.weight"];
        let m = select_matching("model.layers.*.attn", &names);
        assert_eq!(m, vec!["model.layers.0.attn"]);
    }

    #[test]
    fn glob_prefix_with_double_star_catches_all_in_subtree() {
        let names = ["model.layers.0.x", "model.layers.5.y.z", "embed.weight"];
        let m = select_matching("model.layers.**", &names);
        assert_eq!(m.len(), 2);
    }
}
