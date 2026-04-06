#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
//! # Model Migration Pipeline
//!
//! **CLI equivalent:** `apr convert model.safetensors --to apr2 --lint --verify`
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/apr-format-roundtrip-v1.yaml
//!
//! Demonstrates a complete model migration pipeline composing four stages:
//! import, lint, convert, and export. This is the workflow used when
//! migrating a HuggingFace SafeTensors model into the APR v2 format
//! with quality checks and round-trip verification.
//!
//! ## Sections
//! 1. Import — simulate importing a HuggingFace SafeTensors model
//! 2. Lint — run quality checks on the imported model
//! 3. Convert — transform from source format to APR v2
//! 4. Verify — round-trip verification with cosine similarity
//! 5. Export — write final APR bundle with checksum and manifest
//!
//!
//! ## Format Variants
//! ```bash
//! apr convert model.apr          # APR native format
//! apr convert model.gguf         # GGUF (llama.cpp compatible)
//! apr convert model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP. DOI: 10.18653/v1/2020.emnlp-demos.6

use apr_cookbook::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Migration types
// ---------------------------------------------------------------------------

/// Status of a single pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationStatus {
    Pass,
    Fail,
    Warn,
}

impl fmt::Display for MigrationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Fail => write!(f, "FAIL"),
            Self::Warn => write!(f, "WARN"),
        }
    }
}

/// Result of a single pipeline stage.
#[derive(Debug, Clone)]
pub struct MigrationStage {
    pub name: String,
    pub status: MigrationStatus,
    pub duration_ms: f64,
    pub bytes_processed: usize,
    pub detail: String,
}

/// Cumulative migration log across all stages.
#[derive(Debug)]
pub struct MigrationLog {
    pub stages: Vec<MigrationStage>,
    pub source_format: String,
    pub target_format: String,
    pub source_size: usize,
    pub target_size: usize,
}

impl MigrationLog {
    pub fn new(source_format: &str, target_format: &str) -> Self {
        Self {
            stages: Vec::new(),
            source_format: source_format.to_string(),
            target_format: target_format.to_string(),
            source_size: 0,
            target_size: 0,
        }
    }

    pub fn push(&mut self, stage: MigrationStage) {
        self.stages.push(stage);
    }

    pub fn all_passed(&self) -> bool {
        self.stages
            .iter()
            .all(|s| s.status != MigrationStatus::Fail)
    }

    pub fn total_bytes_processed(&self) -> usize {
        self.stages.iter().map(|s| s.bytes_processed).sum()
    }

    pub fn compression_ratio(&self) -> f64 {
        if self.source_size == 0 {
            return 0.0;
        }
        self.target_size as f64 / self.source_size as f64
    }
}

/// Mapping between source and target tensor names.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TensorMapping {
    pub source_name: String,
    pub target_name: String,
    pub shape: Vec<usize>,
    pub source_dtype: String,
    pub target_dtype: String,
}

// ---------------------------------------------------------------------------
// Stage 1: Import
// ---------------------------------------------------------------------------

/// HuggingFace-style tensor names for a 2-layer transformer.
pub const HF_TENSOR_NAMES: &[&str] = &[
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.layers.1.self_attn.k_proj.weight",
    "model.layers.1.self_attn.v_proj.weight",
    "model.layers.1.self_attn.o_proj.weight",
    "model.layers.1.mlp.gate_proj.weight",
    "model.layers.1.mlp.up_proj.weight",
];

/// Imported model with raw tensor data.
pub struct ImportedModel {
    pub tensors: HashMap<String, Vec<u8>>,
    pub shape: Vec<usize>,
    pub metadata: HashMap<String, String>,
}

/// Simulate importing a HuggingFace SafeTensors model.
///
/// Generates deterministic synthetic tensor data using the recipe RNG.
pub fn import_hf_model(rng: &mut impl Rng, dim: usize) -> (ImportedModel, MigrationStage) {
    let shape = vec![dim, dim];
    let elements = dim * dim;
    let mut tensors = HashMap::new();
    let mut total_bytes = 0usize;

    for &name in HF_TENSOR_NAMES {
        let data: Vec<f32> = (0..elements)
            .map(|_| rng.gen_range(-1.0f32..1.0f32))
            .collect();
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        total_bytes += bytes.len();
        tensors.insert(name.to_string(), bytes);
    }

    let mut metadata = HashMap::new();
    metadata.insert("model_type".to_string(), "llama".to_string());
    metadata.insert("hidden_size".to_string(), dim.to_string());
    metadata.insert("num_layers".to_string(), "2".to_string());
    metadata.insert("source_format".to_string(), "safetensors".to_string());

    let model = ImportedModel {
        tensors,
        shape,
        metadata,
    };

    let stage = MigrationStage {
        name: "import".to_string(),
        status: MigrationStatus::Pass,
        duration_ms: 12.3,
        bytes_processed: total_bytes,
        detail: format!(
            "Imported {} tensors ({} bytes) from HuggingFace SafeTensors",
            HF_TENSOR_NAMES.len(),
            total_bytes
        ),
    };

    (model, stage)
}

// ---------------------------------------------------------------------------
// Stage 2: Lint
// ---------------------------------------------------------------------------

/// A single lint finding.
#[derive(Debug)]
pub struct LintFinding {
    pub severity: MigrationStatus,
    pub message: String,
}

/// Run quality checks on the imported model.
///
/// Checks:
/// - Naming convention compliance (dotted HF path)
/// - Dtype consistency (all tensors same size => same dtype)
/// - Missing metadata fields
/// - Tensor shape validation (all shapes match declared shape)
pub fn lint_model(model: &ImportedModel) -> (Vec<LintFinding>, MigrationStage) {
    let mut findings = Vec::new();
    let expected_byte_len = model.shape.iter().product::<usize>() * 4; // FP32

    // Check naming convention
    for name in model.tensors.keys() {
        if !name.contains('.') {
            findings.push(LintFinding {
                severity: MigrationStatus::Warn,
                message: format!("Tensor '{name}' does not use dotted naming convention"),
            });
        }
    }

    // Check dtype consistency (all tensors should have the same byte length)
    let mut inconsistent_count = 0usize;
    for (name, data) in &model.tensors {
        if data.len() != expected_byte_len {
            findings.push(LintFinding {
                severity: MigrationStatus::Fail,
                message: format!(
                    "Tensor '{name}' has {} bytes, expected {expected_byte_len}",
                    data.len()
                ),
            });
            inconsistent_count += 1;
        }
    }

    // Check required metadata
    let required_keys = ["model_type", "hidden_size", "num_layers"];
    for key in &required_keys {
        if !model.metadata.contains_key(*key) {
            findings.push(LintFinding {
                severity: MigrationStatus::Warn,
                message: format!("Missing metadata field: '{key}'"),
            });
        }
    }

    // Check tensor shape validity
    for &dim in &model.shape {
        if dim == 0 {
            findings.push(LintFinding {
                severity: MigrationStatus::Fail,
                message: "Shape contains zero dimension".to_string(),
            });
        }
    }

    let status = if inconsistent_count > 0 {
        MigrationStatus::Fail
    } else if findings.iter().any(|f| f.severity == MigrationStatus::Warn) {
        MigrationStatus::Warn
    } else {
        MigrationStatus::Pass
    };

    let total_bytes: usize = model.tensors.values().map(Vec::len).sum();

    let stage = MigrationStage {
        name: "lint".to_string(),
        status,
        duration_ms: 1.8,
        bytes_processed: total_bytes,
        detail: format!(
            "{} findings ({} errors, {} warnings)",
            findings.len(),
            findings
                .iter()
                .filter(|f| f.severity == MigrationStatus::Fail)
                .count(),
            findings
                .iter()
                .filter(|f| f.severity == MigrationStatus::Warn)
                .count(),
        ),
    };

    (findings, stage)
}
