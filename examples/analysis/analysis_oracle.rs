//! # Model Family Oracle
//!
//! Identifies model architecture family from weight tensor names and shapes.
//!
//! ## CLI equivalent
//! ```bash
//! apr oracle model.apr
//! ```
//!
//! ## What this demonstrates
//! - Heuristic classification of model architectures
//! - Pattern matching on tensor naming conventions
//! - Confidence scoring with evidence accumulation

use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
enum ModelFamily {
    Transformer,
    CNN,
    RNN,
    MLP,
    Unknown,
}

impl std::fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transformer => write!(f, "Transformer"),
            Self::CNN => write!(f, "CNN"),
            Self::RNN => write!(f, "RNN"),
            Self::MLP => write!(f, "MLP"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

#[derive(Debug, Clone)]
struct OracleResult {
    family: ModelFamily,
    confidence: f64,
    evidence: Vec<String>,
}

// ---------------------------------------------------------------------------
// Heuristic patterns
// ---------------------------------------------------------------------------

const TRANSFORMER_PATTERNS: &[&str] = &[
    "attn",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "self_attn",
    "attention",
    "query",
    "key",
    "value",
    "multihead",
    "mha",
    "embed_tokens",
    "lm_head",
];

const CNN_PATTERNS: &[&str] = &[
    "conv",
    "kernel",
    "pool",
    "batch_norm",
    "bn",
    "downsample",
    "upsample",
    "stride",
    "padding",
];

const RNN_PATTERNS: &[&str] = &[
    "hidden",
    "cell",
    "lstm",
    "gru",
    "rnn",
    "h_0",
    "c_0",
    "weight_ih",
    "weight_hh",
];

const MLP_PATTERNS: &[&str] = &[
    "fc",
    "linear",
    "dense",
    "classifier",
    "feedforward",
    "mlp_head",
];

/// Returns the first matching pattern for a tensor name, or `None`.
fn first_pattern_match<'a>(name_lower: &str, patterns: &[&'a str]) -> Option<&'a str> {
    patterns
        .iter()
        .copied()
        .find(|&pat| name_lower.contains(pat))
}

/// Counts name-based pattern hits and collects evidence strings.
fn collect_name_hits(tensor_names: &[String], patterns: &[&str]) -> (usize, Vec<String>) {
    let mut hits = 0usize;
    let mut evidence = Vec::new();

    for name in tensor_names {
        if let Some(pat) = first_pattern_match(&name.to_lowercase(), patterns) {
            hits += 1;
            evidence.push(format!("tensor '{}' matches pattern '{}'", name, pat));
        }
    }

    (hits, evidence)
}

/// Adds evidence for square projection matrices whose names match a pattern.
fn collect_shape_evidence(
    shapes: &[(String, Vec<usize>)],
    patterns: &[&str],
    evidence: &mut Vec<String>,
) {
    for (name, shape) in shapes {
        let is_square_projection = shape.len() == 2 && shape[0] == shape[1] && shape[0] >= 64;
        if !is_square_projection {
            continue;
        }
        if first_pattern_match(&name.to_lowercase(), patterns).is_some() {
            evidence.push(format!(
                "tensor '{}' has square shape {}x{} (projection matrix)",
                name, shape[0], shape[1]
            ));
        }
    }
}

fn score_family(
    tensor_names: &[String],
    shapes: &[(String, Vec<usize>)],
    patterns: &[&str],
) -> (f64, Vec<String>) {
    let (hits, mut evidence) = collect_name_hits(tensor_names, patterns);
    collect_shape_evidence(shapes, patterns, &mut evidence);

    let total = tensor_names.len().max(1) as f64;
    let confidence = (hits as f64 / total).clamp(0.0, 1.0);

    (confidence, evidence)
}

fn identify_family(tensor_names: &[String], shapes: &[(String, Vec<usize>)]) -> OracleResult {
    let candidates: Vec<_> = [
        (ModelFamily::Transformer, TRANSFORMER_PATTERNS),
        (ModelFamily::CNN, CNN_PATTERNS),
        (ModelFamily::RNN, RNN_PATTERNS),
        (ModelFamily::MLP, MLP_PATTERNS),
    ]
    .iter()
    .map(|(family, patterns)| {
        let (conf, ev) = score_family(tensor_names, shapes, patterns);
        (family.clone(), conf, ev)
    })
    .collect();

    let (best_family, best_confidence, best_evidence) = candidates
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or_else(|| (ModelFamily::Unknown, 0.0, Vec::new()));

    // Require minimum confidence threshold
    if best_confidence < 0.1 {
        return OracleResult {
            family: ModelFamily::Unknown,
            confidence: best_confidence,
            evidence: vec!["no strong pattern matches found".to_string()],
        };
    }

    OracleResult {
        family: best_family,
        confidence: best_confidence,
        evidence: best_evidence,
    }
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_oracle")?;

    // ── Section 1: Build a synthetic transformer model ──────────────────
    println!("=== Model Family Oracle ===\n");

    let tensor_names: Vec<String> = vec![
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let shapes: Vec<(String, Vec<usize>)> = vec![
        ("model.embed_tokens.weight".into(), vec![32000, 768]),
        (
            "model.layers.0.self_attn.q_proj.weight".into(),
            vec![768, 768],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight".into(),
            vec![768, 768],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight".into(),
            vec![768, 768],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight".into(),
            vec![768, 768],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".into(),
            vec![768, 3072],
        ),
        ("model.layers.0.mlp.up_proj.weight".into(), vec![768, 3072]),
        (
            "model.layers.0.mlp.down_proj.weight".into(),
            vec![3072, 768],
        ),
        ("model.norm.weight".into(), vec![768]),
        ("lm_head.weight".into(), vec![32000, 768]),
    ];

    // ── Section 2: Tensor name analysis ─────────────────────────────────
    println!("--- Tensor Name Analysis ---");
    println!("Model contains {} tensors:", tensor_names.len());
    for name in &tensor_names {
        println!("  {}", name);
    }
    println!();

    // ── Section 3: Shape pattern matching ───────────────────────────────
    println!("--- Shape Pattern Matching ---");
    for (name, shape) in &shapes {
        let shape_str: Vec<String> = shape.iter().map(ToString::to_string).collect();
        println!("  {} : [{}]", name, shape_str.join(", "));
    }
    println!();

    // ── Section 4: Confidence scoring ───────────────────────────────────
    let result = identify_family(&tensor_names, &shapes);

    println!("--- Confidence Scoring ---");
    println!("Confidence: {:.1}%", result.confidence * 100.0);
    println!("Evidence ({} signals):", result.evidence.len());
    for ev in &result.evidence {
        println!("  - {}", ev);
    }
    println!();

    // ── Section 5: Family identification ────────────────────────────────
    println!("--- Family Identification ---");
    println!("Detected family: {}", result.family);
    println!("Confidence: {:.1}%", result.confidence * 100.0);

    // ── Section 6: Demonstrate with a CNN model ─────────────────────────
    println!("\n--- CNN Model Test ---");
    let cnn_names: Vec<String> = vec![
        "backbone.conv1.weight",
        "backbone.conv1.bias",
        "backbone.bn1.weight",
        "backbone.conv2.weight",
        "backbone.pool.weight",
        "head.fc.weight",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let cnn_shapes: Vec<(String, Vec<usize>)> = vec![
        ("backbone.conv1.weight".into(), vec![64, 3, 7, 7]),
        ("backbone.conv1.bias".into(), vec![64]),
        ("backbone.bn1.weight".into(), vec![64]),
        ("backbone.conv2.weight".into(), vec![128, 64, 3, 3]),
        ("backbone.pool.weight".into(), vec![128]),
        ("head.fc.weight".into(), vec![1000, 128]),
    ];

    let cnn_result = identify_family(&cnn_names, &cnn_shapes);
    println!(
        "Detected family: {} ({:.1}%)",
        cnn_result.family,
        cnn_result.confidence * 100.0
    );

    // Use hash to demonstrate determinism
    let mut hasher = DefaultHasher::new();
    result.family.to_string().hash(&mut hasher);
    result.evidence.len().hash(&mut hasher);
    println!("\nOracle fingerprint: {:016x}", hasher.finish());

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(raw: &[&str]) -> Vec<String> {
        raw.iter().map(|s| s.to_string()).collect()
    }

    fn shapes_from(raw: &[(&str, Vec<usize>)]) -> Vec<(String, Vec<usize>)> {
        raw.iter()
            .map(|(n, s)| (n.to_string(), s.clone()))
            .collect()
    }

    #[test]
    fn test_transformer_detected() {
        let n = names(&[
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
            "layers.0.mlp.gate.weight",
            "embed_tokens.weight",
        ]);
        let s = shapes_from(&[
            ("layers.0.self_attn.q_proj.weight", vec![768, 768]),
            ("layers.0.self_attn.k_proj.weight", vec![768, 768]),
            ("layers.0.self_attn.v_proj.weight", vec![768, 768]),
            ("layers.0.mlp.gate.weight", vec![768, 3072]),
            ("embed_tokens.weight", vec![32000, 768]),
        ]);
        let result = identify_family(&n, &s);
        assert_eq!(result.family, ModelFamily::Transformer);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_cnn_detected() {
        let n = names(&[
            "backbone.conv1.weight",
            "backbone.conv2.weight",
            "backbone.bn1.weight",
            "backbone.pool.weight",
        ]);
        let s = shapes_from(&[
            ("backbone.conv1.weight", vec![64, 3, 7, 7]),
            ("backbone.conv2.weight", vec![128, 64, 3, 3]),
            ("backbone.bn1.weight", vec![64]),
            ("backbone.pool.weight", vec![128]),
        ]);
        let result = identify_family(&n, &s);
        assert_eq!(result.family, ModelFamily::CNN);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_rnn_detected() {
        let n = names(&[
            "encoder.lstm.weight_ih",
            "encoder.lstm.weight_hh",
            "encoder.lstm.cell.weight",
            "decoder.rnn.hidden.weight",
        ]);
        let s = shapes_from(&[
            ("encoder.lstm.weight_ih", vec![512, 128]),
            ("encoder.lstm.weight_hh", vec![512, 512]),
            ("encoder.lstm.cell.weight", vec![512]),
            ("decoder.rnn.hidden.weight", vec![256, 512]),
        ]);
        let result = identify_family(&n, &s);
        assert_eq!(result.family, ModelFamily::RNN);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_mlp_detected() {
        let n = names(&[
            "classifier.fc1.weight",
            "classifier.fc1.bias",
            "classifier.fc2.weight",
            "classifier.fc2.bias",
            "classifier.fc3.weight",
            "classifier.linear.weight",
        ]);
        let s = shapes_from(&[
            ("classifier.fc1.weight", vec![256, 784]),
            ("classifier.fc1.bias", vec![256]),
            ("classifier.fc2.weight", vec![128, 256]),
            ("classifier.fc2.bias", vec![128]),
            ("classifier.fc3.weight", vec![10, 128]),
            ("classifier.linear.weight", vec![10, 128]),
        ]);
        let result = identify_family(&n, &s);
        assert_eq!(result.family, ModelFamily::MLP);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_unknown_for_random_names() {
        let n = names(&["xyz_123", "foo_bar", "baz_qux"]);
        let s = shapes_from(&[
            ("xyz_123", vec![10]),
            ("foo_bar", vec![20]),
            ("baz_qux", vec![30]),
        ]);
        let result = identify_family(&n, &s);
        assert_eq!(result.family, ModelFamily::Unknown);
    }

    #[test]
    fn test_confidence_bounded_zero_to_one() {
        let n = names(&[
            "layers.0.self_attn.q_proj",
            "layers.0.self_attn.k_proj",
            "layers.0.self_attn.v_proj",
        ]);
        let s = shapes_from(&[
            ("layers.0.self_attn.q_proj", vec![768, 768]),
            ("layers.0.self_attn.k_proj", vec![768, 768]),
            ("layers.0.self_attn.v_proj", vec![768, 768]),
        ]);
        let result = identify_family(&n, &s);
        assert!(result.confidence >= 0.0);
        assert!(result.confidence <= 1.0);
    }

    #[test]
    fn test_evidence_populated_for_matches() {
        let n = names(&["layer.attn.q_proj.weight", "layer.attn.k_proj.weight"]);
        let s = shapes_from(&[
            ("layer.attn.q_proj.weight", vec![768, 768]),
            ("layer.attn.k_proj.weight", vec![768, 768]),
        ]);
        let result = identify_family(&n, &s);
        assert!(!result.evidence.is_empty());
    }

    #[test]
    fn test_empty_tensors_returns_unknown() {
        let n: Vec<String> = vec![];
        let s: Vec<(String, Vec<usize>)> = vec![];
        let result = identify_family(&n, &s);
        assert_eq!(result.family, ModelFamily::Unknown);
    }

    #[test]
    fn test_display_impl_all_families() {
        assert_eq!(ModelFamily::Transformer.to_string(), "Transformer");
        assert_eq!(ModelFamily::CNN.to_string(), "CNN");
        assert_eq!(ModelFamily::RNN.to_string(), "RNN");
        assert_eq!(ModelFamily::MLP.to_string(), "MLP");
        assert_eq!(ModelFamily::Unknown.to_string(), "Unknown");
    }

    #[test]
    fn test_mixed_signals_picks_strongest() {
        // Mostly transformer with one CNN tensor
        let n = names(&["attn.q_proj", "attn.k_proj", "attn.v_proj", "conv1.weight"]);
        let s = shapes_from(&[
            ("attn.q_proj", vec![768, 768]),
            ("attn.k_proj", vec![768, 768]),
            ("attn.v_proj", vec![768, 768]),
            ("conv1.weight", vec![64, 3, 7, 7]),
        ]);
        let result = identify_family(&n, &s);
        assert_eq!(result.family, ModelFamily::Transformer);
    }
}
