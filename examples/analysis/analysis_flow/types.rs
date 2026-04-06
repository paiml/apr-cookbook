#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use std::collections::BTreeMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Architecture component classification derived from tensor naming conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FlowComponent {
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
pub struct TensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub component: FlowComponent,
}

impl TensorEntry {
    pub fn param_count(&self) -> usize {
        self.shape.iter().product()
    }
}

/// The ordered flow graph: each entry is a (component, total_param_count) pair.
#[derive(Debug, Clone)]
pub struct FlowGraph {
    pub components: Vec<(FlowComponent, usize)>,
}

/// Per-layer grouping used to build the flow diagram.
#[derive(Debug, Clone)]
pub struct LayerGroup {
    pub layer_idx: usize,
    pub components: Vec<(FlowComponent, usize)>,
}

// ---------------------------------------------------------------------------
// Component classification
// ---------------------------------------------------------------------------

// Keyword-to-component lookup table, checked in priority order.
/// Each entry is `(keywords, component)`. The first match wins.
pub const COMPONENT_RULES: &[(&[&str], FlowComponent)] = &[
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

// Check whether `lower` matches the special-case Output rule, which needs
/// a negative guard (`!contains("layer")`).
pub fn is_output_tensor(lower: &str) -> bool {
    lower.contains("lm_head") || (lower.contains("output") && !lower.contains("layer"))
}

/// Check whether `lower` matches any keyword in a rule's keyword list.
pub fn matches_any_keyword(lower: &str, keywords: &[&str]) -> bool {
    keywords.iter().any(|kw| lower.contains(kw))
}

/// Classify a tensor name into an architecture component using a lookup table.
pub fn classify_tensor(name: &str) -> FlowComponent {
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
pub fn parse_tensors(raw: &[(String, Vec<usize>)]) -> Vec<TensorEntry> {
    raw.iter()
        .map(|(name, shape)| TensorEntry {
            name: name.clone(),
            shape: shape.clone(),
            component: classify_tensor(name),
        })
        .collect()
}

// Build a `FlowGraph` by aggregating parameter counts per component type,
/// preserving the order of first occurrence.
pub fn build_flow_graph(entries: &[TensorEntry]) -> FlowGraph {
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

// Extract layer index from a tensor name like "model.layers.3.self_attn.q_proj.weight".
/// Returns `None` for non-layer tensors (embeddings, output, etc.).
pub fn extract_layer_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, &part) in parts.iter().enumerate() {
        if (part == "layers" || part == "layer") && i + 1 < parts.len() {
            return parts[i + 1].parse().ok();
        }
    }
    None
}

/// Group tensors by layer index, preserving component order within each layer.
pub fn group_by_layer(entries: &[TensorEntry]) -> (Vec<TensorEntry>, Vec<LayerGroup>) {
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
pub fn format_params(n: usize) -> String {
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

// Render the top-level ASCII flow diagram.
//
// Format:
///   Input -> Embedding -> [Layer 0: Attention -> FFN -> Norm] -> ... -> Output
pub fn render_flow_diagram(entries: &[TensorEntry]) -> String {
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
pub fn render_breakdown_table(graph: &FlowGraph) -> String {
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
