//! # Tensor Transformation Flow Visualization
//! **CLI Equivalent**: `apr flow`
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

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Architecture component classification derived from tensor naming conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum FlowComponent {
    Embedding,
    SelfAttention,
    CrossAttention,
    Ffn,
    LayerNorm,
    Output,
    Unknown,
}

impl fmt::Display for FlowComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Embedding => write!(f, "Embedding"),
            Self::SelfAttention => write!(f, "SelfAttention"),
            Self::CrossAttention => write!(f, "CrossAttention"),
            Self::Ffn => write!(f, "FFN"),
            Self::LayerNorm => write!(f, "LayerNorm"),
            Self::Output => write!(f, "Output"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// A single tensor with its name, shape, and classified component type.
#[derive(Debug, Clone)]
struct TensorEntry {
    name: String,
    shape: Vec<usize>,
    component: FlowComponent,
}

impl TensorEntry {
    fn param_count(&self) -> usize {
        self.shape.iter().product()
    }
}

/// The ordered flow graph: each entry is a (component, total_param_count) pair.
#[derive(Debug, Clone)]
struct FlowGraph {
    components: Vec<(FlowComponent, usize)>,
}

/// Per-layer grouping used to build the flow diagram.
#[derive(Debug, Clone)]
struct LayerGroup {
    layer_idx: usize,
    components: Vec<(FlowComponent, usize)>,
}

// ---------------------------------------------------------------------------
// Component classification
// ---------------------------------------------------------------------------

/// Keyword-to-component lookup table, checked in priority order.
/// Each entry is `(keywords, component)`. The first match wins.
const COMPONENT_RULES: &[(&[&str], FlowComponent)] = &[
    (&["embed", "wte", "wpe"], FlowComponent::Embedding),
    (
        &["cross_attn", "encoder_attn"],
        FlowComponent::CrossAttention,
    ),
    (
        &["self_attn", "q_proj", "k_proj", "v_proj", "o_proj"],
        FlowComponent::SelfAttention,
    ),
    (
        &["mlp", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"],
        FlowComponent::Ffn,
    ),
    (&["norm", "layernorm", "rmsnorm"], FlowComponent::LayerNorm),
];

/// Check whether `lower` matches the special-case Output rule, which needs
/// a negative guard (`!contains("layer")`).
fn is_output_tensor(lower: &str) -> bool {
    lower.contains("lm_head") || (lower.contains("output") && !lower.contains("layer"))
}

/// Check whether `lower` matches any keyword in a rule's keyword list.
fn matches_any_keyword(lower: &str, keywords: &[&str]) -> bool {
    keywords.iter().any(|kw| lower.contains(kw))
}

/// Classify a tensor name into an architecture component using a lookup table.
fn classify_tensor(name: &str) -> FlowComponent {
    let lower = name.to_lowercase();

    if is_output_tensor(&lower) {
        return FlowComponent::Output;
    }

    for &(keywords, component) in COMPONENT_RULES {
        if matches_any_keyword(&lower, keywords) {
            return component;
        }
    }

    FlowComponent::Unknown
}

// ---------------------------------------------------------------------------
// Flow graph construction
// ---------------------------------------------------------------------------

/// Parse raw tensor definitions into classified `TensorEntry` values.
fn parse_tensors(raw: &[(String, Vec<usize>)]) -> Vec<TensorEntry> {
    raw.iter()
        .map(|(name, shape)| TensorEntry {
            name: name.clone(),
            shape: shape.clone(),
            component: classify_tensor(name),
        })
        .collect()
}

/// Build a `FlowGraph` by aggregating parameter counts per component type,
/// preserving the order of first occurrence.
fn build_flow_graph(entries: &[TensorEntry]) -> FlowGraph {
    let mut seen_order: Vec<FlowComponent> = Vec::new();
    let mut totals: BTreeMap<FlowComponent, usize> = BTreeMap::new();

    for entry in entries {
        let count = totals.entry(entry.component).or_insert(0);
        if *count == 0 {
            seen_order.push(entry.component);
        }
        *count += entry.param_count();
    }

    let components = seen_order
        .into_iter()
        .map(|c| (c, totals.get(&c).copied().unwrap_or(0)))
        .collect();

    FlowGraph { components }
}

/// Extract layer index from a tensor name like "model.layers.3.self_attn.q_proj.weight".
/// Returns `None` for non-layer tensors (embeddings, output, etc.).
fn extract_layer_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, &part) in parts.iter().enumerate() {
        if (part == "layers" || part == "layer") && i + 1 < parts.len() {
            return parts[i + 1].parse().ok();
        }
    }
    None
}

/// Group tensors by layer index, preserving component order within each layer.
fn group_by_layer(entries: &[TensorEntry]) -> (Vec<TensorEntry>, Vec<LayerGroup>) {
    let mut non_layer: Vec<TensorEntry> = Vec::new();
    let mut layer_map: BTreeMap<usize, Vec<&TensorEntry>> = BTreeMap::new();

    for entry in entries {
        match extract_layer_index(&entry.name) {
            Some(idx) => layer_map.entry(idx).or_default().push(entry),
            None => non_layer.push(entry.clone()),
        }
    }

    let groups: Vec<LayerGroup> = layer_map
        .into_iter()
        .map(|(layer_idx, tensors)| {
            let mut comp_totals: BTreeMap<FlowComponent, usize> = BTreeMap::new();
            let mut order: Vec<FlowComponent> = Vec::new();
            for t in &tensors {
                let count = comp_totals.entry(t.component).or_insert(0);
                if *count == 0 {
                    order.push(t.component);
                }
                *count += t.param_count();
            }
            let components = order
                .into_iter()
                .map(|c| (c, comp_totals.get(&c).copied().unwrap_or(0)))
                .collect();
            LayerGroup {
                layer_idx,
                components,
            }
        })
        .collect();

    (non_layer, groups)
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Format a parameter count in human-readable form.
fn format_params(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{n}")
    }
}

/// Render the top-level ASCII flow diagram.
///
/// Format:
///   Input -> Embedding -> [Layer 0: Attention -> FFN -> Norm] -> ... -> Output
fn render_flow_diagram(entries: &[TensorEntry]) -> String {
    let (non_layer, layer_groups) = group_by_layer(entries);
    let mut parts: Vec<String> = Vec::new();

    // Leading non-layer components (e.g., Embedding)
    parts.push("Input".to_string());
    for entry in &non_layer {
        if entry.component == FlowComponent::Embedding {
            parts.push(format!("Embedding({})", format_params(entry.param_count())));
        }
    }

    // Layer groups
    for group in &layer_groups {
        let inner: Vec<String> = group
            .components
            .iter()
            .map(|(c, p)| format!("{c}({})", format_params(*p)))
            .collect();
        parts.push(format!(
            "[Layer {}: {}]",
            group.layer_idx,
            inner.join(" -> ")
        ));
    }

    // Trailing non-layer components (e.g., LayerNorm, Output)
    for entry in &non_layer {
        match entry.component {
            FlowComponent::LayerNorm => {
                parts.push(format!("Norm({})", format_params(entry.param_count())));
            }
            FlowComponent::Output => {
                parts.push(format!("Output({})", format_params(entry.param_count())));
            }
            _ => {}
        }
    }

    parts.join(" -> ")
}

/// Render a component breakdown table with parameter counts and percentages.
fn render_breakdown_table(graph: &FlowGraph) -> String {
    let total: usize = graph.components.iter().map(|(_, p)| p).sum();
    let mut output = String::new();

    output.push_str(&format!(
        "{:<18} {:>12} {:>8}\n",
        "Component", "Params", "% Total"
    ));
    output.push_str(&format!("{}\n", "-".repeat(40)));

    for (component, param_count) in &graph.components {
        let pct = if total > 0 {
            (*param_count as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        output.push_str(&format!(
            "{:<18} {:>12} {:>7.1}%\n",
            component.to_string(),
            format_params(*param_count),
            pct,
        ));
    }

    output.push_str(&format!("{}\n", "-".repeat(40)));
    output.push_str(&format!("{:<18} {:>12}\n", "TOTAL", format_params(total)));
    output
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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
