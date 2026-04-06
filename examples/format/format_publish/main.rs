#![allow(unused_imports)]
//! # Publish Model to HuggingFace
//!
//! **CLI equivalent:** `apr publish model.apr --repo org/model`
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/apr-format-roundtrip-v1.yaml
//!
//! Demonstrates the publishing workflow: preparing a model for upload
//! to HuggingFace Hub. Generates a model card (README.md), validates
//! the manifest, and prepares all files needed for the repository.
//!
//! ## Sections
//! 1. Manifest preparation — list all files needed for the repo
//! 2. Model card generation — Markdown README with model metadata
//! 3. File listing — final set of files to upload
//! 4. Pre-publish validation — check for missing or invalid files
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
use std::collections::HashMap;

mod types;
use types::ModelMetadata;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*; // explicit: shadow prelude's ModelMetadata with local one

fn main() -> Result<()> {
    let ctx = RecipeContext::new("format_publish")?;

    // Build model
    let model_payload = generate_model_payload(42, 128 * 128);
    let apr_bundle = ModelBundleV2::new()
        .with_name("phi-mini-apr")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::Int8)
        .add_tensor("model.weight", vec![128, 128], model_payload)
        .build();

    let mut metrics = HashMap::new();
    metrics.insert("accuracy".to_string(), 0.8934);
    metrics.insert("perplexity".to_string(), 12.45);
    metrics.insert("f1_score".to_string(), 0.8821);

    let metadata = ModelMetadata {
        name: "paiml/phi-mini-apr".to_string(),
        architecture: "transformer".to_string(),
        parameters: 2_700_000,
        quantization: "INT8".to_string(),
        license: "apache-2.0".to_string(),
        language: "en".to_string(),
        tags: vec![
            "text-generation".to_string(),
            "apr".to_string(),
            "quantized".to_string(),
        ],
        datasets: vec!["openwebtext".to_string(), "c4".to_string()],
        metrics,
    };

    // Section 1: Manifest preparation
    let manifest = prepare_publish("phi-mini-apr", "paiml/phi-mini-apr", &apr_bundle, &metadata);
    println!("=== Publish Manifest ===");
    println!("Repository:   {}", manifest.repo);
    println!("Total size:   {} bytes", manifest.total_size_bytes);
    println!("Config keys:  {}", manifest.config.len());
    println!();

    // Section 2: Model card
    println!("=== Generated Model Card ===");
    for line in manifest.model_card.lines().take(25) {
        println!("  {line}");
    }
    if manifest.model_card.lines().count() > 25 {
        println!(
            "  ... ({} more lines)",
            manifest.model_card.lines().count() - 25
        );
    }
    println!();

    // Section 3: File listing
    println!("=== Files to Upload ===");
    println!("{:<25} {:<12} {:<20}", "Filename", "Size (B)", "Checksum");
    println!("{}", "-".repeat(57));
    for file in &manifest.tensor_files {
        println!(
            "{:<25} {:<12} {:<20}",
            file.filename, file.size_bytes, file.checksum
        );
    }
    println!();

    // Section 4: Pre-publish validation
    let issues = validate_manifest(&manifest);
    println!("=== Pre-Publish Validation ===");
    if issues.is_empty() {
        println!("All checks passed!");
    } else {
        for issue in &issues {
            let severity = match issue.severity {
                Severity::Error => "ERROR",
                Severity::Warning => "WARN",
            };
            println!("  [{severity}] {}: {}", issue.field, issue.message);
        }
    }
    let errors = issues
        .iter()
        .filter(|i| i.severity == Severity::Error)
        .count();
    println!("Errors:   {errors}");
    println!("Warnings: {}", issues.len() - errors);

    assert_eq!(errors, 0, "Must have zero errors to publish");

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> ModelMetadata {
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);
        ModelMetadata {
            name: "test/model".to_string(),
            architecture: "transformer".to_string(),
            parameters: 1_000_000,
            quantization: "FP32".to_string(),
            license: "mit".to_string(),
            language: "en".to_string(),
            tags: vec!["test".to_string()],
            datasets: vec!["dataset-1".to_string()],
            metrics,
        }
    }

    #[test]
    fn test_manifest_complete() {
        let data = generate_model_payload(42, 256);
        let apr = ModelBundleV2::new()
            .with_name("test")
            .with_compression(Compression::None)
            .with_quantization(Quantization::FP32)
            .add_tensor("w", vec![16, 16], data.clone())
            .build();
        let manifest = prepare_publish("test", "org/test", &apr, &sample_metadata());
        assert_eq!(manifest.repo, "org/test");
        assert!(!manifest.model_card.is_empty());
        assert!(manifest.tensor_files.len() >= 2);
        assert!(manifest.config.contains_key("model_type"));
    }

    #[test]
    fn test_model_card_frontmatter_and_usage() {
        let card = generate_model_card(&sample_metadata());
        assert!(card.starts_with("---\n"), "missing YAML front matter");
        assert!(card.contains("license: mit"));
        assert!(card.contains("## Usage"));
    }

    #[test]
    fn test_model_card_evaluation_and_training() {
        let card = generate_model_card(&sample_metadata());
        assert!(card.contains("## Evaluation"));
        assert!(card.contains("accuracy"));
        assert!(card.contains("## Training Details"));
    }

    #[test]
    fn test_validation_passes_good_manifest() {
        let data = generate_model_payload(42, 256);
        let apr = ModelBundleV2::new()
            .with_name("test")
            .with_compression(Compression::None)
            .with_quantization(Quantization::FP32)
            .add_tensor("w", vec![16, 16], data)
            .build();
        let manifest = prepare_publish("test", "org/test", &apr, &sample_metadata());
        let issues = validate_manifest(&manifest);
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty(), "Errors: {errors:?}");
    }

    fn stub_manifest(repo: &str, card: &str) -> PublishManifest {
        PublishManifest {
            repo: repo.to_string(),
            model_card: card.to_string(),
            tensor_files: vec![FileEntry {
                filename: "model.apr".to_string(),
                size_bytes: 100,
                checksum: "abc".to_string(),
            }],
            config: HashMap::from([("model_type".to_string(), "t".to_string())]),
            tokenizer_config: None,
            total_size_bytes: 100,
        }
    }

    #[test]
    fn test_validation_catches_bad_repo() {
        let issues = validate_manifest(&stub_manifest("noslash", "---\n---\n## Usage\n"));
        assert!(issues.iter().any(|i| i.field == "repo"));
    }

    #[test]
    fn test_validation_catches_empty_card() {
        let issues = validate_manifest(&stub_manifest("org/name", ""));
        assert!(issues
            .iter()
            .any(|i| i.field == "model_card" && i.severity == Severity::Error));
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(500), "500");
        assert_eq!(format_number(1_500), "1.5K");
        assert_eq!(format_number(2_500_000), "2.5M");
        assert_eq!(format_number(7_000_000_000), "7.0B");
    }

    #[test]
    fn test_checksum_deterministic_and_unique() {
        let d1 = generate_model_payload(42, 128);
        let d2 = generate_model_payload(43, 256);
        assert_eq!(compute_checksum(&d1), compute_checksum(&d1));
        assert_ne!(compute_checksum(&d1), compute_checksum(&d2));
    }
}
