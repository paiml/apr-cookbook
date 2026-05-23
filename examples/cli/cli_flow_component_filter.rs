//! # apr flow — `--component` Filter
//!
//! `apr flow <FILE> --component {full,encoder,decoder,attention,mlp,norm}`
//! restricts the data-flow visualisation to a sub-graph of the model.
//! This recipe builds the resolver and asserts the contract: each component
//! maps to a known set of layer-name patterns; unknown components reject
//! rather than silently default to `full`.
//!
//! Demonstrates the **FLOW.4** recipe for PMAT-100 (apr flow coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender FLOW-001
//!
//! Run with: cargo run --example cli_flow_component_filter
//!
//! Added by PMAT-100 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    Full,
    Encoder,
    Decoder,
    Attention,
    Mlp,
    Norm,
}

impl Component {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "full" => Some(Component::Full),
            "encoder" => Some(Component::Encoder),
            "decoder" => Some(Component::Decoder),
            "attention" => Some(Component::Attention),
            "mlp" => Some(Component::Mlp),
            "norm" => Some(Component::Norm),
            _ => None,
        }
    }
}

pub fn patterns_for(c: Component) -> &'static [&'static str] {
    match c {
        Component::Full => &["*"],
        Component::Encoder => &["encoder.layers.", "model.encoder."],
        Component::Decoder => &["decoder.layers.", "model.decoder."],
        Component::Attention => &["self_attn", "cross_attn", "self.attention"],
        Component::Mlp => &["mlp.", "ffn.", "feed_forward"],
        Component::Norm => &["norm", "layer_norm", "rmsnorm"],
    }
}

pub fn filter_layers<'a>(layers: &[&'a str], c: Component) -> Vec<&'a str> {
    let pats = patterns_for(c);
    layers
        .iter()
        .copied()
        .filter(|l| pats.iter().any(|p| *p == "*" || l.contains(p)))
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_flow_component_filter")?;

    let layers = [
        "model.embed_tokens.weight",
        "encoder.layers.0.self_attn.q_proj.weight",
        "encoder.layers.0.mlp.gate_proj.weight",
        "decoder.layers.0.self_attn.q_proj.weight",
        "decoder.layers.0.cross_attn.q_proj.weight",
        "model.norm.weight",
    ];

    for c in [
        Component::Full,
        Component::Encoder,
        Component::Decoder,
        Component::Attention,
        Component::Mlp,
        Component::Norm,
    ] {
        println!("--component {c:?}: {:?}", filter_layers(&layers, c));
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
    fn full_matches_everything() {
        let layers = ["a", "b", "c"];
        let f = filter_layers(&layers, Component::Full);
        assert_eq!(f.len(), 3);
    }

    #[test]
    fn encoder_filter_excludes_decoder() {
        let layers = ["encoder.layers.0.self_attn", "decoder.layers.0.self_attn"];
        let f = filter_layers(&layers, Component::Encoder);
        assert_eq!(f, vec!["encoder.layers.0.self_attn"]);
    }

    #[test]
    fn attention_filter_picks_self_and_cross_attn() {
        let layers = [
            "encoder.layers.0.self_attn.q_proj",
            "decoder.layers.0.cross_attn.q_proj",
            "encoder.layers.0.mlp.gate_proj",
        ];
        let f = filter_layers(&layers, Component::Attention);
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn mlp_filter_picks_ffn_aliases() {
        let layers = [
            "encoder.layers.0.mlp.gate_proj",
            "encoder.layers.0.ffn.up_proj",
            "encoder.layers.0.feed_forward.w1",
            "encoder.layers.0.self_attn.q_proj",
        ];
        let f = filter_layers(&layers, Component::Mlp);
        assert_eq!(f.len(), 3);
    }

    #[test]
    fn unknown_component_returns_none() {
        assert!(Component::from_str_strict("unknown").is_none());
        assert!(Component::from_str_strict("").is_none());
    }

    #[test]
    fn known_components_round_trip() {
        for s in ["full", "encoder", "decoder", "attention", "mlp", "norm"] {
            assert!(Component::from_str_strict(s).is_some(), "missing: {s}");
        }
    }
}
