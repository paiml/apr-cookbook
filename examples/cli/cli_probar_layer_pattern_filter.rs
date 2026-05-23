//! # apr probar — Layer Pattern Filter
//!
//! `apr probar <FILE> --layer <PATTERN>` restricts visual regression
//! coverage to a subset of tensors. The pattern matching uses substring
//! match (not regex/glob) to keep the surface predictable for non-Rust
//! consumers. This recipe documents and tests that contract.
//!
//! Demonstrates the **PROBAR.5** recipe for PMAT-093 (apr probar coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-481
//!
//! Run with: cargo run --example cli_probar_layer_pattern_filter
//!
//! Added by PMAT-093 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

pub fn filter_layers<'a>(layers: &'a [&'a str], pattern: Option<&str>) -> Vec<&'a str> {
    let Some(pat) = pattern else {
        return layers.to_vec();
    };
    layers.iter().copied().filter(|l| l.contains(pat)).collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_probar_layer_pattern_filter")?;

    let layers = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "lm_head.weight",
    ];

    println!(
        "no filter:    {} layers",
        filter_layers(&layers, None).len()
    );
    println!(
        "self_attn:    {:?}",
        filter_layers(&layers, Some("self_attn"))
    );
    println!(
        "layers.0:     {:?}",
        filter_layers(&layers, Some("layers.0"))
    );
    println!("q_proj:       {:?}", filter_layers(&layers, Some("q_proj")));
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
            "model.layers.1.self_attn.q_proj.weight",
            "lm_head.weight",
        ]
    }

    #[test]
    fn filter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_pattern_returns_all_layers() {
        let layers = sample();
        let filtered = filter_layers(&layers, None);
        assert_eq!(filtered.len(), layers.len());
    }

    #[test]
    fn substring_match_is_inclusive() {
        let layers = sample();
        // "self_attn" matches three q/k_proj layers (excludes embed/lm_head).
        let filtered = filter_layers(&layers, Some("self_attn"));
        assert_eq!(filtered.len(), 3);
    }

    #[test]
    fn no_matches_returns_empty_not_panic() {
        // Empty result is allowed — caller checks length and emits a warning.
        let layers = sample();
        let filtered = filter_layers(&layers, Some("nonexistent_layer"));
        assert!(filtered.is_empty());
    }

    #[test]
    fn pattern_is_case_sensitive() {
        // Documented contract: substring match is case-sensitive (matches grep
        // default). Avoids surprising matches when the operator types lowercase
        // but the tensor names are mixed-case.
        let layers = ["model.SelfAttn.q_proj.weight"];
        assert!(filter_layers(&layers, Some("selfattn")).is_empty());
        assert_eq!(filter_layers(&layers, Some("SelfAttn")).len(), 1);
    }

    #[test]
    fn pattern_matches_trailing_dots() {
        // Substring match (not glob) means dots match dots literally.
        let layers = ["a.b.c", "abc"];
        let f = filter_layers(&layers, Some("a.b"));
        assert_eq!(f, vec!["a.b.c"]); // "abc" must NOT match
    }

    #[test]
    fn empty_pattern_matches_everything() {
        // "".contains("") is true for all strings — so empty filter == no filter.
        let layers = sample();
        let filtered = filter_layers(&layers, Some(""));
        assert_eq!(filtered.len(), layers.len());
    }
}
