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
use std::collections::HashMap;

/// Manifest for publishing a model to HuggingFace.
#[derive(Debug)]
#[allow(dead_code)]
pub struct PublishManifest {
    pub repo: String,
    pub model_card: String,
    pub tensor_files: Vec<FileEntry>,
    pub config: HashMap<String, String>,
    pub tokenizer_config: Option<HashMap<String, String>>,
    pub total_size_bytes: usize,
}

/// A file entry in the publish manifest.
#[derive(Debug, Clone)]
pub struct FileEntry {
    pub filename: String,
    pub size_bytes: usize,
    pub checksum: String,
}

/// Model metadata for card generation.
#[derive(Debug)]
pub struct ModelMetadata {
    pub name: String,
    pub architecture: String,
    pub parameters: usize,
    pub quantization: String,
    pub license: String,
    pub language: String,
    pub tags: Vec<String>,
    pub datasets: Vec<String>,
    pub metrics: HashMap<String, f64>,
}

/// Validation issue found during pre-publish checks.
#[derive(Debug)]
pub struct ValidationIssue {
    pub severity: Severity,
    pub field: String,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Severity {
    Error,
    Warning,
}

/// Generate a model card (README.md) from model metadata.
pub fn generate_model_card(metadata: &ModelMetadata) -> String {
    let mut card = String::new();

    // YAML front matter
    card.push_str("---\n");
    card.push_str(&format!("license: {}\n", metadata.license));
    card.push_str(&format!("language:\n- {}\n", metadata.language));
    card.push_str("tags:\n");
    for tag in &metadata.tags {
        card.push_str(&format!("- {tag}\n"));
    }
    if !metadata.datasets.is_empty() {
        card.push_str("datasets:\n");
        for ds in &metadata.datasets {
            card.push_str(&format!("- {ds}\n"));
        }
    }
    card.push_str("---\n\n");

    // Title
    card.push_str(&format!("# {}\n\n", metadata.name));

    // Description
    card.push_str("## Model Description\n\n");
    card.push_str(&format!(
        "This model uses the **{}** architecture with **{}** parameters, ",
        metadata.architecture,
        format_number(metadata.parameters),
    ));
    card.push_str(&format!(
        "quantized to **{}** precision.\n\n",
        metadata.quantization
    ));

    // Usage
    card.push_str("## Usage\n\n");
    card.push_str("```bash\n");
    card.push_str(&format!("apr pull hf://{}\n", metadata.name));
    card.push_str(&format!("apr run {}\n", metadata.name));
    card.push_str("```\n\n");

    // Metrics
    if !metadata.metrics.is_empty() {
        card.push_str("## Evaluation\n\n");
        card.push_str("| Metric | Value |\n");
        card.push_str("|--------|-------|\n");
        let mut metric_keys: Vec<_> = metadata.metrics.keys().collect();
        metric_keys.sort();
        for key in metric_keys {
            card.push_str(&format!("| {key} | {:.4} |\n", metadata.metrics[key]));
        }
        card.push('\n');
    }

    // Training details
    card.push_str("## Training Details\n\n");
    card.push_str("- **Format:** APR v2\n");
    card.push_str(&format!("- **Quantization:** {}\n", metadata.quantization));
    card.push_str(&format!("- **License:** {}\n", metadata.license));
    card.push('\n');

    card
}

/// Format a number with K/M/B suffixes.
pub fn format_number(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{n}")
    }
}

/// Compute a simple checksum for a byte slice (simulated SHA-256).
pub fn compute_checksum(data: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    format!("{hash:016x}")
}

/// Prepare a publish manifest from model data.
pub fn prepare_publish(
    model_name: &str,
    repo: &str,
    apr_data: &[u8],
    metadata: &ModelMetadata,
) -> PublishManifest {
    let model_card = generate_model_card(metadata);

    let model_file = FileEntry {
        filename: format!("{model_name}.apr"),
        size_bytes: apr_data.len(),
        checksum: compute_checksum(apr_data),
    };

    let card_bytes = model_card.as_bytes();
    let readme_file = FileEntry {
        filename: "README.md".to_string(),
        size_bytes: card_bytes.len(),
        checksum: compute_checksum(card_bytes),
    };

    let mut config = HashMap::new();
    config.insert("model_type".to_string(), metadata.architecture.clone());
    config.insert(
        "num_parameters".to_string(),
        metadata.parameters.to_string(),
    );
    config.insert("quantization".to_string(), metadata.quantization.clone());

    let total_size = model_file.size_bytes + readme_file.size_bytes;

    PublishManifest {
        repo: repo.to_string(),
        model_card,
        tensor_files: vec![model_file, readme_file],
        config,
        tokenizer_config: None,
        total_size_bytes: total_size,
    }
}

/// Validate a publish manifest for completeness and correctness.
pub fn validate_manifest(manifest: &PublishManifest) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    // Check repo format
    if !manifest.repo.contains('/') {
        issues.push(ValidationIssue {
            severity: Severity::Error,
            field: "repo".to_string(),
            message: "Repository must be in format 'org/name'".to_string(),
        });
    }

    // Check model card
    if manifest.model_card.is_empty() {
        issues.push(ValidationIssue {
            severity: Severity::Error,
            field: "model_card".to_string(),
            message: "Model card (README.md) is empty".to_string(),
        });
    }

    if !manifest.model_card.contains("---\n") {
        issues.push(ValidationIssue {
            severity: Severity::Warning,
            field: "model_card".to_string(),
            message: "Model card missing YAML front matter".to_string(),
        });
    }

    if !manifest.model_card.contains("## Usage") {
        issues.push(ValidationIssue {
            severity: Severity::Warning,
            field: "model_card".to_string(),
            message: "Model card missing Usage section".to_string(),
        });
    }

    // Check for .apr file
    let has_apr = manifest.tensor_files.iter().any(|f| {
        std::path::Path::new(&f.filename)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("apr"))
    });
    if !has_apr {
        issues.push(ValidationIssue {
            severity: Severity::Error,
            field: "tensor_files".to_string(),
            message: "No .apr model file found".to_string(),
        });
    }

    // Check for README
    let has_readme = manifest
        .tensor_files
        .iter()
        .any(|f| f.filename == "README.md");
    if !has_readme {
        issues.push(ValidationIssue {
            severity: Severity::Warning,
            field: "tensor_files".to_string(),
            message: "No README.md file found".to_string(),
        });
    }

    // Check config
    if !manifest.config.contains_key("model_type") {
        issues.push(ValidationIssue {
            severity: Severity::Error,
            field: "config".to_string(),
            message: "Missing model_type in config".to_string(),
        });
    }

    // Check file sizes
    for file in &manifest.tensor_files {
        if file.size_bytes == 0 {
            issues.push(ValidationIssue {
                severity: Severity::Error,
                field: file.filename.clone(),
                message: "File has zero size".to_string(),
            });
        }
    }

    issues
}
