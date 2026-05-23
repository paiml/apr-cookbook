//! # apr flow — Per-Layer Aggregation
//!
//! `apr flow <FILE>` aggregates tensors with the same layer prefix
//! (`model.layers.0.*` → one node "layer 0") so the output graph stays
//! readable for 80-layer models. This recipe builds the aggregator and
//! asserts the contract: the prefix regex `model\.layers\.(\d+)\.` is
//! the canonical aggregation key, non-layer tensors form their own
//! group ("global").
//!
//! Demonstrates the **FLOW.6** recipe for PMAT-100 (apr flow coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender FLOW-003 + tensor-name canonical convention
//!
//! Run with: cargo run --example cli_flow_layer_aggregation
//!
//! Added by PMAT-100 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerGroup {
    pub layer_index: Option<u32>, // None for "global" group
    pub tensors: Vec<String>,
}

pub fn aggregate_by_layer<'a>(tensors: &[&'a str]) -> BTreeMap<i64, Vec<&'a str>> {
    // Use i64 as key with -1 for "global" so BTreeMap orders them naturally.
    let mut groups: BTreeMap<i64, Vec<&'a str>> = BTreeMap::new();
    for &t in tensors {
        if let Some(idx) = parse_layer_index(t) {
            groups.entry(idx as i64).or_default().push(t);
        } else {
            groups.entry(-1).or_default().push(t);
        }
    }
    groups
}

pub fn parse_layer_index(name: &str) -> Option<u32> {
    let prefix = "model.layers.";
    let rest = name.strip_prefix(prefix)?;
    let dot = rest.find('.')?;
    rest[..dot].parse::<u32>().ok()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_flow_layer_aggregation")?;

    let tensors = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.27.self_attn.q_proj.weight",
        "lm_head.weight",
    ];

    for (key, ts) in aggregate_by_layer(&tensors) {
        let label = if key < 0 {
            "global".to_string()
        } else {
            format!("layer {key}")
        };
        println!("{label:>10}  ({} tensors)", ts.len());
        for t in ts {
            println!("    {t}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregation_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn parse_layer_index_works() {
        assert_eq!(parse_layer_index("model.layers.0.foo"), Some(0));
        assert_eq!(parse_layer_index("model.layers.27.foo"), Some(27));
    }

    #[test]
    fn parse_returns_none_for_non_layer() {
        assert!(parse_layer_index("model.embed_tokens.weight").is_none());
        assert!(parse_layer_index("lm_head.weight").is_none());
    }

    #[test]
    fn parse_returns_none_for_malformed() {
        // Missing trailing dot, non-numeric index, etc.
        assert!(parse_layer_index("model.layers.0").is_none());
        assert!(parse_layer_index("model.layers.abc.foo").is_none());
        assert!(parse_layer_index("").is_none());
    }

    #[test]
    fn aggregator_groups_layer_tensors() {
        let tensors = [
            "model.layers.0.q",
            "model.layers.0.k",
            "model.layers.0.v",
            "model.layers.1.q",
        ];
        let g = aggregate_by_layer(&tensors);
        assert_eq!(g[&0].len(), 3);
        assert_eq!(g[&1].len(), 1);
    }

    #[test]
    fn aggregator_groups_globals_separately() {
        let tensors = [
            "model.embed_tokens.weight",
            "model.layers.0.q",
            "lm_head.weight",
        ];
        let g = aggregate_by_layer(&tensors);
        assert_eq!(g[&-1].len(), 2); // embed + lm_head
        assert_eq!(g[&0].len(), 1);
    }

    #[test]
    fn empty_input_yields_empty_map() {
        let g = aggregate_by_layer(&[]);
        assert!(g.is_empty());
    }

    #[test]
    fn keys_in_btreemap_are_sorted_naturally() {
        // -1 (globals) sorts before 0, 1, 2, …
        let tensors = [
            "model.layers.5.q",
            "model.layers.0.q",
            "lm_head.weight",
            "model.layers.10.q",
        ];
        let g = aggregate_by_layer(&tensors);
        let keys: Vec<i64> = g.keys().copied().collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
        assert_eq!(keys.first(), Some(&-1));
    }
}
