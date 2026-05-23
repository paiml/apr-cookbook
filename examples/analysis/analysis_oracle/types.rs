//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub enum ModelFamily {
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
pub struct OracleResult {
    pub family: ModelFamily,
    pub confidence: f64,
    pub evidence: Vec<String>,
}

// ---------------------------------------------------------------------------
// Heuristic patterns
// ---------------------------------------------------------------------------

pub const TRANSFORMER_PATTERNS: &[&str] = &[
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

pub const CNN_PATTERNS: &[&str] = &[
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

pub const RNN_PATTERNS: &[&str] = &[
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

pub const MLP_PATTERNS: &[&str] = &[
    "fc",
    "linear",
    "dense",
    "classifier",
    "feedforward",
    "mlp_head",
];

/// Returns the first matching pattern for a tensor name, or `None`.
pub fn first_pattern_match<'a>(name_lower: &str, patterns: &[&'a str]) -> Option<&'a str> {
    patterns
        .iter()
        .copied()
        .find(|&pat| name_lower.contains(pat))
}

/// Counts name-based pattern hits and collects evidence strings.
pub fn collect_name_hits(tensor_names: &[String], patterns: &[&str]) -> (usize, Vec<String>) {
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
pub fn collect_shape_evidence(
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

pub fn score_family(
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

pub fn identify_family(tensor_names: &[String], shapes: &[(String, Vec<usize>)]) -> OracleResult {
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
