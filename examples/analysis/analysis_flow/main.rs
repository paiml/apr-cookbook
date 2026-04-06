#![allow(unused_imports)]
//! # Tensor Transformation Flow Visualization
//! **CLI Equivalent**: `apr flow`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Renders a model's tensor transformation flow as an ASCII pipeline diagram,
//! showing data path through architecture components with parameter counts.
//!
//! ## CLI equivalent
//! ```bash
//! apr flow model.apr
//! ```
//!
//! ## What this demonstrates
//! - Parsing tensor names into architecture components
//! - Building a flow graph from flat tensor metadata
//! - Rendering ASCII flow diagrams for architecture visualization
//! - Computing per-component parameter breakdowns
//!
//!
//! ## Format Variants
//! ```bash
//! apr flow model.apr          # APR native format
//! apr flow model.gguf         # GGUF (llama.cpp compatible)
//! apr flow model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use std::collections::BTreeMap;
use std::fmt;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_flow")?;

    println!("=== Tensor Transformation Flow ===\n");

    // -- Section 1: Define a synthetic transformer model --
    println!("--- Section 1: Model Tensor Definitions ---");
    let tensors: Vec<(String, Vec<usize>)> = vec![
        // Embedding
        ("model.embed_tokens.weight".into(), vec![32000, 4096]),
        // Layer 0
        (
            "model.layers.0.self_attn.q_proj.weight".into(),
            vec![4096, 4096],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight".into(),
            vec![4096, 4096],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight".into(),
            vec![4096, 4096],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight".into(),
            vec![4096, 4096],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".into(),
            vec![11008, 4096],
        ),
        (
            "model.layers.0.mlp.up_proj.weight".into(),
            vec![11008, 4096],
        ),
        (
            "model.layers.0.mlp.down_proj.weight".into(),
            vec![4096, 11008],
        ),
        ("model.layers.0.input_layernorm.weight".into(), vec![4096]),
        (
            "model.layers.0.post_attention_layernorm.weight".into(),
            vec![4096],
        ),
        // Layer 1
        (
            "model.layers.1.self_attn.q_proj.weight".into(),
            vec![4096, 4096],
        ),
        (
            "model.layers.1.self_attn.k_proj.weight".into(),
            vec![4096, 4096],
        ),
        (
            "model.layers.1.self_attn.v_proj.weight".into(),
            vec![4096, 4096],
        ),
        (
            "model.layers.1.self_attn.o_proj.weight".into(),
            vec![4096, 4096],
        ),
        (
            "model.layers.1.mlp.gate_proj.weight".into(),
            vec![11008, 4096],
        ),
        (
            "model.layers.1.mlp.up_proj.weight".into(),
            vec![11008, 4096],
        ),
        (
            "model.layers.1.mlp.down_proj.weight".into(),
            vec![4096, 11008],
        ),
        ("model.layers.1.input_layernorm.weight".into(), vec![4096]),
        (
            "model.layers.1.post_attention_layernorm.weight".into(),
            vec![4096],
        ),
        // Final norm + output head
        ("model.norm.weight".into(), vec![4096]),
        ("lm_head.weight".into(), vec![32000, 4096]),
    ];

    for (name, shape) in &tensors {
        let params: usize = shape.iter().product();
        let shape_str: Vec<String> = shape.iter().map(ToString::to_string).collect();
        println!(
            "  {:<55} [{:<16}] {:>10}",
            name,
            shape_str.join(", "),
            format_params(params)
        );
    }

    // -- Section 2: Classify tensors into components --
    println!("\n--- Section 2: Component Classification ---");
    let entries = parse_tensors(&tensors);

    for entry in &entries {
        println!("  {:<55} -> {}", entry.name, entry.component);
    }

    // -- Section 3: Build flow graph --
    println!("\n--- Section 3: Flow Graph ---");
    let graph = build_flow_graph(&entries);

    for (component, param_count) in &graph.components {
        println!("  {} : {} params", component, format_params(*param_count));
    }

    // -- Section 4: ASCII flow diagram --
    println!("\n--- Section 4: Flow Diagram ---");
    let diagram = render_flow_diagram(&entries);
    println!("  {diagram}");

    // -- Section 5: Component breakdown table --
    println!("\n--- Section 5: Component Breakdown ---");
    let table = render_breakdown_table(&graph);
    println!("{table}");

    // -- Section 6: Per-layer summary --
    println!("--- Section 6: Per-Layer Summary ---");
    let (_, layer_groups) = group_by_layer(&entries);
    for group in &layer_groups {
        let layer_total: usize = group.components.iter().map(|(_, p)| *p).sum();
        let inner: Vec<String> = group
            .components
            .iter()
            .map(|(c, p)| format!("{}={}", c, format_params(*p)))
            .collect();
        println!(
            "  Layer {}: {} (total: {})",
            group.layer_idx,
            inner.join(", "),
            format_params(layer_total)
        );
    }

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tensors() -> Vec<(String, Vec<usize>)> {
        vec![
            ("model.embed_tokens.weight".into(), vec![32000, 4096]),
            (
                "model.layers.0.self_attn.q_proj.weight".into(),
                vec![4096, 4096],
            ),
            (
                "model.layers.0.self_attn.k_proj.weight".into(),
                vec![4096, 4096],
            ),
            (
                "model.layers.0.mlp.gate_proj.weight".into(),
                vec![11008, 4096],
            ),
            ("model.layers.0.input_layernorm.weight".into(), vec![4096]),
            ("model.norm.weight".into(), vec![4096]),
            ("lm_head.weight".into(), vec![32000, 4096]),
        ]
    }

    #[test]
    fn test_classify_embedding() {
        assert_eq!(
            classify_tensor("model.embed_tokens.weight"),
            FlowComponent::Embedding
        );
        assert_eq!(classify_tensor("wte.weight"), FlowComponent::Embedding);
        assert_eq!(classify_tensor("wpe.weight"), FlowComponent::Embedding);
    }

    #[test]
    fn test_classify_self_attention() {
        assert_eq!(
            classify_tensor("model.layers.0.self_attn.q_proj.weight"),
            FlowComponent::SelfAttention
        );
        assert_eq!(
            classify_tensor("model.layers.0.self_attn.k_proj.weight"),
            FlowComponent::SelfAttention
        );
        assert_eq!(
            classify_tensor("model.layers.0.self_attn.v_proj.weight"),
            FlowComponent::SelfAttention
        );
        assert_eq!(
            classify_tensor("model.layers.0.self_attn.o_proj.weight"),
            FlowComponent::SelfAttention
        );
    }

    #[test]
    fn test_classify_cross_attention() {
        assert_eq!(
            classify_tensor("model.layers.0.cross_attn.q_proj.weight"),
            FlowComponent::CrossAttention
        );
        assert_eq!(
            classify_tensor("decoder.layers.0.encoder_attn.k_proj.weight"),
            FlowComponent::CrossAttention
        );
    }

    #[test]
    fn test_classify_ffn() {
        assert_eq!(
            classify_tensor("model.layers.0.mlp.gate_proj.weight"),
            FlowComponent::Ffn
        );
        assert_eq!(
            classify_tensor("model.layers.0.mlp.up_proj.weight"),
            FlowComponent::Ffn
        );
        assert_eq!(
            classify_tensor("model.layers.0.mlp.down_proj.weight"),
            FlowComponent::Ffn
        );
    }

    #[test]
    fn test_classify_layernorm() {
        assert_eq!(
            classify_tensor("model.layers.0.input_layernorm.weight"),
            FlowComponent::LayerNorm
        );
        assert_eq!(
            classify_tensor("model.layers.0.post_attention_layernorm.weight"),
            FlowComponent::LayerNorm
        );
        assert_eq!(
            classify_tensor("model.norm.weight"),
            FlowComponent::LayerNorm
        );
    }

    #[test]
    fn test_classify_output() {
        assert_eq!(classify_tensor("lm_head.weight"), FlowComponent::Output);
    }

    #[test]
    fn test_flow_graph_aggregation() {
        let entries = parse_tensors(&sample_tensors());
        let graph = build_flow_graph(&entries);

        // Should contain exactly 5 distinct component types from sample data
        let component_types: Vec<FlowComponent> =
            graph.components.iter().map(|(c, _)| *c).collect();
        assert!(component_types.contains(&FlowComponent::Embedding));
        assert!(component_types.contains(&FlowComponent::SelfAttention));
        assert!(component_types.contains(&FlowComponent::Ffn));
        assert!(component_types.contains(&FlowComponent::LayerNorm));
        assert!(component_types.contains(&FlowComponent::Output));
    }

    #[test]
    fn test_param_count_embedding() {
        let entries = parse_tensors(&sample_tensors());
        let embed_params: usize = entries
            .iter()
            .filter(|e| e.component == FlowComponent::Embedding)
            .map(TensorEntry::param_count)
            .sum();
        assert_eq!(embed_params, 32000 * 4096);
    }

    #[test]
    fn test_flow_diagram_contains_components() {
        let entries = parse_tensors(&sample_tensors());
        let diagram = render_flow_diagram(&entries);

        assert!(diagram.contains("Input"), "Diagram must start with Input");
        assert!(
            diagram.contains("Embedding"),
            "Diagram must include Embedding"
        );
        assert!(diagram.contains("Layer 0"), "Diagram must include Layer 0");
        assert!(diagram.contains("->"), "Diagram must use arrow connectors");
    }

    #[test]
    fn test_breakdown_table_percentages_sum() {
        let entries = parse_tensors(&sample_tensors());
        let graph = build_flow_graph(&entries);
        let total: usize = graph.components.iter().map(|(_, p)| *p).sum();

        // Verify all components sum to total
        let sum_parts: usize = graph.components.iter().map(|(_, p)| *p).sum();
        assert_eq!(sum_parts, total);

        // Verify table renders without panic
        let table = render_breakdown_table(&graph);
        assert!(table.contains("TOTAL"));
        assert!(table.contains('%'));
    }
}
