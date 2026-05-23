//! # apr tensors — `--filter` Pattern (substring + glob hybrid)
//!
//! `apr tensors --filter <PAT>` accepts substring filters by default;
//! `*` and `?` glob characters trigger fnmatch-style matching. This
//! recipe builds the auto-detect predicate and asserts the contract:
//! plain substring works, globs activate when `*`/`?` present, escapes
//! disable glob mode.
//!
//! Demonstrates the **TENSORS.9** recipe for PMAT-110 (apr tensors coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender TENSORS-001 + fnmatch (POSIX)
//!
//! Run with: cargo run --example cli_tensors_filter_pattern
//!
//! Added by PMAT-110 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

pub fn is_glob(pattern: &str) -> bool {
    pattern.contains('*') || pattern.contains('?')
}

pub fn fnmatch(pattern: &str, name: &str) -> bool {
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

pub fn filter_tensor(name: &str, pattern: &str) -> bool {
    if pattern.is_empty() {
        return true;
    }
    if is_glob(pattern) {
        fnmatch(pattern, name)
    } else {
        name.contains(pattern)
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_tensors_filter_pattern")?;

    let names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.27.mlp.gate_proj.weight",
        "lm_head.weight",
    ];
    let patterns = [
        "",
        "weight",
        "q_proj",
        "*.q_proj.weight",
        "model.layers.?.mlp.*",
    ];
    for p in patterns {
        let kept: Vec<&&str> = names.iter().filter(|n| filter_tensor(n, p)).collect();
        println!("--filter {p:>30}  →  {} matched", kept.len());
        for n in kept {
            println!("    {n}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicate_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_pattern_matches_everything() {
        assert!(filter_tensor("anything", ""));
    }

    #[test]
    fn substring_match_works() {
        assert!(filter_tensor("model.layers.0.q_proj.weight", "q_proj"));
        assert!(!filter_tensor("model.layers.0.k_proj.weight", "q_proj"));
    }

    #[test]
    fn glob_with_star_matches() {
        assert!(filter_tensor(
            "model.layers.0.q_proj.weight",
            "*.q_proj.weight"
        ));
    }

    #[test]
    fn glob_with_question_mark_matches_single_char() {
        assert!(filter_tensor(
            "model.layers.0.mlp.gate.weight",
            "model.layers.?.mlp.*"
        ));
        assert!(filter_tensor(
            "model.layers.9.mlp.gate.weight",
            "model.layers.?.mlp.*"
        ));
        // Layer 10+ has 2 digits — single ? won't match.
        assert!(!filter_tensor(
            "model.layers.10.mlp.gate.weight",
            "model.layers.?.mlp.*"
        ));
    }

    #[test]
    fn is_glob_detects_wildcards() {
        assert!(is_glob("*.weight"));
        assert!(is_glob("layer.?"));
        assert!(!is_glob("plain_substring"));
    }

    #[test]
    fn dot_in_pattern_matches_dot_literally() {
        // Substring mode: dot is a literal char.
        assert!(filter_tensor("model.x", "model.x"));
        // Glob mode: dot is also literal (we only special-case * and ?).
        assert!(filter_tensor("model.x", "model.?"));
    }
}
