//! # apr compare-hf — Tensor Filter (Glob Selectors)
//!
//! `apr compare-hf` accepts `--include` and `--exclude` glob filters so that
//! large LLM checkpoints can be diffed one block (or one head) at a time
//! without holding the entire weight matrix in memory. This recipe builds
//! the glob-match decision in pure Rust so the contract surface is exercised
//! offline.
//!
//! Demonstrates the **CMPHF.1** recipe for PMAT-088 (apr compare-hf coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CMPHF-001 + glob 0.3 (BSD/MIT)
//!
//! Run with: cargo run --example cli_compare_hf_tensor_filter
//!
//! Added by PMAT-088 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn glob_match(pattern: &str, name: &str) -> bool {
    // Two-state matcher: '*' matches any (possibly empty) substring,
    // every other char matches itself. Sufficient for tensor-name globs
    // like "model.layers.*.self_attn.q_proj.weight".
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
            if p[i - 1] == '*' {
                dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
            } else if p[i - 1] == n[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }
    dp[p.len()][n.len()]
}

fn select_tensors<'a>(names: &'a [&'a str], include: &[&str], exclude: &[&str]) -> Vec<&'a str> {
    names
        .iter()
        .filter(|n| include.is_empty() || include.iter().any(|p| glob_match(p, n)))
        .filter(|n| !exclude.iter().any(|p| glob_match(p, n)))
        .copied()
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_compare_hf_tensor_filter")?;

    let tensors = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "lm_head.weight",
    ];

    let attn_q = select_tensors(&tensors, &["model.layers.*.self_attn.q_proj.weight"], &[]);
    let attn_no_layer1 = select_tensors(
        &tensors,
        &["model.layers.*.self_attn.*"],
        &["model.layers.1.*"],
    );

    println!("attention Q projections:    {attn_q:#?}");
    println!("attention layer-0 only:     {attn_no_layer1:#?}");
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
    fn glob_star_matches_any_substring() {
        assert!(glob_match("model.*.weight", "model.embed_tokens.weight"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("a*z", "az"));
    }

    #[test]
    fn glob_literal_must_match() {
        assert!(!glob_match("a*z", "ab"));
        assert!(!glob_match("model.weight", "model.bias"));
    }

    #[test]
    fn include_filter_narrows() {
        let names = ["a.weight", "b.weight", "a.bias"];
        let kept = select_tensors(&names, &["a.*"], &[]);
        assert_eq!(kept, vec!["a.weight", "a.bias"]);
    }

    #[test]
    fn exclude_filter_subtracts() {
        let names = ["a.weight", "b.weight", "a.bias"];
        let kept = select_tensors(&names, &[], &["*.bias"]);
        assert_eq!(kept, vec!["a.weight", "b.weight"]);
    }

    #[test]
    fn include_and_exclude_compose() {
        let names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
        ];
        let kept = select_tensors(
            &names,
            &["model.layers.*.self_attn.*"],
            &["model.layers.1.*"],
        );
        assert_eq!(
            kept,
            vec![
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
            ]
        );
    }
}
