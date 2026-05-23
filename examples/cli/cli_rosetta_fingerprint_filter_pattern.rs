//! # apr rosetta fingerprint — `--filter` Pattern
//!
//! `apr rosetta fingerprint <FILE> --filter <PATTERN>` restricts
//! fingerprinting to tensors matching the substring pattern. The match
//! is documented as case-sensitive substring (matching the probar
//! contract). This recipe asserts the same semantics so tooling shared
//! across rosetta and probar agrees.
//!
//! Demonstrates the **ROSETTA-FINGERPRINT.2** recipe for PMAT-097 (fingerprint coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-201 + JAX-STAT-001
//!
//! Run with: cargo run --example cli_rosetta_fingerprint_filter_pattern
//!
//! Added by PMAT-097 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

pub fn select_for_fingerprint<'a>(tensors: &'a [&'a str], filter: Option<&str>) -> Vec<&'a str> {
    match filter {
        None => tensors.to_vec(),
        Some(pat) => tensors
            .iter()
            .copied()
            .filter(|t| t.contains(pat))
            .collect(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_fingerprint_filter_pattern")?;

    let tensors = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "lm_head.weight",
    ];

    println!(
        "no filter:       {} tensors",
        select_for_fingerprint(&tensors, None).len()
    );
    println!(
        "--filter q_proj: {:?}",
        select_for_fingerprint(&tensors, Some("q_proj"))
    );
    println!(
        "--filter mlp:    {:?}",
        select_for_fingerprint(&tensors, Some("mlp"))
    );
    println!(
        "--filter lm:     {:?}",
        select_for_fingerprint(&tensors, Some("lm"))
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<&'static str> {
        vec![
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "lm_head.weight",
        ]
    }

    #[test]
    fn filter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_filter_keeps_all() {
        let t = sample();
        assert_eq!(select_for_fingerprint(&t, None).len(), t.len());
    }

    #[test]
    fn substring_filter_matches() {
        let t = sample();
        let kept = select_for_fingerprint(&t, Some("q_proj"));
        assert_eq!(kept, vec!["model.layers.0.self_attn.q_proj.weight"]);
    }

    #[test]
    fn no_match_returns_empty() {
        // Empty selection is allowed; downstream emits a warning.
        let t = sample();
        assert!(select_for_fingerprint(&t, Some("nonexistent")).is_empty());
    }

    #[test]
    fn case_sensitive_match() {
        // Documented contract: case-sensitive (matches grep + probar).
        let t = ["model.SelfAttn.weight"];
        assert!(select_for_fingerprint(&t, Some("selfattn")).is_empty());
        assert_eq!(select_for_fingerprint(&t, Some("SelfAttn")).len(), 1);
    }

    #[test]
    fn empty_pattern_keeps_all() {
        // "".contains("") is true for all strings.
        let t = sample();
        assert_eq!(select_for_fingerprint(&t, Some("")).len(), t.len());
    }
}
