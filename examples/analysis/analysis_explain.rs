//! # APR Error Code Explanation System
//! **CLI Equivalent**: `apr explain`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Provides detailed explanations, causes, and solutions for APR error codes,
//! similar to `rustc --explain`.
//!
//! ## CLI equivalent
//! ```bash
//! apr explain E001
//! ```
//!
//! ## What this demonstrates
//! - Error catalog design pattern
//! - Structured error documentation
//! - Lookup-based CLI diagnostics
//!
//!
//! ## Format Variants
//! ```bash
//! apr explain model.apr          # APR native format
//! apr explain model.gguf         # GGUF (llama.cpp compatible)
//! apr explain model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ErrorCode {
    code: String,
    title: String,
    description: String,
    causes: Vec<String>,
    solutions: Vec<String>,
    related: Vec<String>,
}

// ---------------------------------------------------------------------------
// Error catalog
// ---------------------------------------------------------------------------

fn build_catalog() -> Vec<ErrorCode> {
    vec![
        ErrorCode {
            code: "E001".to_string(),
            title: "Invalid magic bytes".to_string(),
            description: "The file does not begin with the expected APR magic bytes (APR2). \
                This indicates the file is not a valid APR v2 model or has been corrupted."
                .to_string(),
            causes: vec![
                "File is not an APR model".to_string(),
                "File was truncated during download".to_string(),
                "File is an APR v1 model (magic: APR1)".to_string(),
                "Binary corruption of the first 4 bytes".to_string(),
            ],
            solutions: vec![
                "Verify the file was downloaded completely".to_string(),
                "Check file extension matches format (.apr)".to_string(),
                "Use `apr hex model.apr` to inspect the magic bytes".to_string(),
                "If APR v1, convert with `apr convert --from aprv1 --to aprv2`".to_string(),
            ],
            related: vec!["E002".to_string(), "E003".to_string()],
        },
        ErrorCode {
            code: "E002".to_string(),
            title: "Version mismatch".to_string(),
            description: "The APR format version in the file header does not match the version \
                supported by this library. This typically occurs when using an older library \
                with a newer model format."
                .to_string(),
            causes: vec![
                "Model was saved with a newer APR version".to_string(),
                "Library is outdated and does not support this version".to_string(),
                "Header corruption in the version field".to_string(),
            ],
            solutions: vec![
                "Update aprender to the latest version".to_string(),
                "Check version with `apr hex model.apr` (bytes 4-7)".to_string(),
                "Downgrade the model format with `apr convert --version 2`".to_string(),
            ],
            related: vec!["E001".to_string()],
        },
        ErrorCode {
            code: "E003".to_string(),
            title: "Tensor data corruption".to_string(),
            description: "One or more tensor data regions failed integrity checks. The stored \
                checksum does not match the computed checksum of the tensor data."
                .to_string(),
            causes: vec![
                "File was corrupted during transfer".to_string(),
                "Disk I/O error during write".to_string(),
                "Incomplete model save operation".to_string(),
                "Memory corruption during serialization".to_string(),
            ],
            solutions: vec![
                "Re-download the model from the original source".to_string(),
                "Verify file checksum against the published hash".to_string(),
                "Use `apr canary check model.apr` to identify corrupted tensors".to_string(),
                "Re-export the model from the training checkpoint".to_string(),
            ],
            related: vec!["E001".to_string(), "E004".to_string()],
        },
        ErrorCode {
            code: "E004".to_string(),
            title: "Tensor size mismatch".to_string(),
            description: "The declared tensor dimensions do not match the actual data size. \
                The product of shape dimensions times dtype size does not equal the stored \
                byte count."
                .to_string(),
            causes: vec![
                "Metadata specifies wrong shape".to_string(),
                "Quantization changed dtype without updating metadata".to_string(),
                "Tensor data was truncated".to_string(),
                "Manual editing of model metadata".to_string(),
            ],
            solutions: vec![
                "Inspect tensor metadata with `apr tree model.apr`".to_string(),
                "Verify shapes match the model architecture documentation".to_string(),
                "Re-export with correct quantization settings".to_string(),
                "Use `apr hex model.apr` to check raw byte counts".to_string(),
            ],
            related: vec!["E003".to_string(), "E005".to_string()],
        },
        ErrorCode {
            code: "E005".to_string(),
            title: "Unsupported quantization".to_string(),
            description: "The model uses a quantization scheme that is not supported by this \
                version of the library. Common unsupported schemes include experimental \
                quantization formats."
                .to_string(),
            causes: vec![
                "Model uses a newer quantization format".to_string(),
                "Custom quantization not registered with the library".to_string(),
                "Quantization metadata is corrupted".to_string(),
            ],
            solutions: vec![
                "Update aprender to the latest version".to_string(),
                "Dequantize and re-quantize with a supported scheme".to_string(),
                "Supported schemes: FP32, FP16, INT8, INT4, Q4_0, Q4_1, Q8_0".to_string(),
                "Use `apr convert --quantize fp32` to dequantize first".to_string(),
            ],
            related: vec!["E004".to_string(), "E006".to_string()],
        },
        ErrorCode {
            code: "E006".to_string(),
            title: "Decompression failed".to_string(),
            description: "The compressed data region could not be decompressed. This typically \
                indicates corruption in the compressed payload or a mismatch between the \
                declared compression algorithm and the actual data."
                .to_string(),
            causes: vec![
                "Compressed data is corrupted".to_string(),
                "Wrong compression algorithm specified in metadata".to_string(),
                "Library does not support the compression algorithm".to_string(),
                "File was partially overwritten".to_string(),
            ],
            solutions: vec![
                "Re-download the model file".to_string(),
                "Check compression metadata with `apr hex model.apr`".to_string(),
                "Re-export with a different compression: `apr convert --compress lz4`".to_string(),
                "Supported algorithms: none, lz4, zstd, snappy".to_string(),
            ],
            related: vec!["E003".to_string(), "E005".to_string()],
        },
    ]
}

/// Look up an error code in the catalog.
fn explain_error(code: &str) -> Option<ErrorCode> {
    let catalog = build_catalog();
    let normalized = code.to_uppercase();
    catalog.into_iter().find(|e| e.code == normalized)
}

/// Get all error codes in the catalog.
fn all_error_codes() -> Vec<String> {
    build_catalog().iter().map(|e| e.code.clone()).collect()
}

/// Format an error explanation for display.
fn format_explanation(error: &ErrorCode) -> String {
    let mut output = String::new();

    output.push_str(&format!("{}: {}\n", error.code, error.title));
    output.push_str(&"=".repeat(error.code.len() + error.title.len() + 2));
    output.push('\n');
    output.push('\n');
    output.push_str(&error.description);
    output.push_str("\n\n");

    output.push_str("Possible causes:\n");
    for cause in &error.causes {
        output.push_str(&format!("  - {}\n", cause));
    }
    output.push('\n');

    output.push_str("Solutions:\n");
    for (i, solution) in error.solutions.iter().enumerate() {
        output.push_str(&format!("  {}. {}\n", i + 1, solution));
    }

    if !error.related.is_empty() {
        output.push('\n');
        output.push_str(&format!("Related errors: {}\n", error.related.join(", ")));
    }

    output
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_explain")?;

    // ── Section 1: Error lookup ─────────────────────────────────────────
    println!("=== APR Error Code Explainer ===\n");

    let target_code = "E001";
    println!("--- Error Lookup: {} ---", target_code);
    match explain_error(target_code) {
        Some(error) => {
            println!("{}", format_explanation(&error));
        }
        None => {
            println!("Unknown error code: {}", target_code);
        }
    }

    // ── Section 2: Detailed explanation ──────────────────────────────────
    println!("--- Detailed Explanation: E003 ---");
    if let Some(error) = explain_error("E003") {
        println!("{}", format_explanation(&error));
    }

    // ── Section 3: Suggested fixes ──────────────────────────────────────
    println!("--- Suggested Fixes for E005 ---");
    if let Some(error) = explain_error("E005") {
        println!("Solutions for \"{} - {}\":", error.code, error.title);
        for (i, sol) in error.solutions.iter().enumerate() {
            println!("  {}. {}", i + 1, sol);
        }
        println!();
    }

    // ── Section 4: Related errors ───────────────────────────────────────
    println!("--- Related Error Navigation ---");
    if let Some(error) = explain_error("E004") {
        println!("{} relates to: {}", error.code, error.related.join(", "));
        for rel_code in &error.related {
            if let Some(rel) = explain_error(rel_code) {
                println!(
                    "  {} - {}: {}",
                    rel.code,
                    rel.title,
                    &rel.description[..60.min(rel.description.len())]
                );
            }
        }
        println!();
    }

    // ── Section 5: Full catalog listing ─────────────────────────────────
    println!("--- Full Error Catalog ---");
    let codes = all_error_codes();
    println!("{} error codes defined:", codes.len());
    for code in &codes {
        if let Some(error) = explain_error(code) {
            println!("  {} - {}", error.code, error.title);
        }
    }

    // ── Section 6: Unknown code handling ────────────────────────────────
    println!("\n--- Unknown Code Handling ---");
    let unknown = "E999";
    match explain_error(unknown) {
        Some(_) => println!("{} found", unknown),
        None => println!(
            "{}: Unknown error code. Use `apr explain --list` to see all codes.",
            unknown
        ),
    }

    // Fingerprint
    let mut hasher = DefaultHasher::new();
    codes.len().hash(&mut hasher);
    target_code.hash(&mut hasher);
    println!("\nExplainer fingerprint: {:016x}", hasher.finish());

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e001_returns_some() {
        assert!(explain_error("E001").is_some());
    }

    #[test]
    fn test_e002_returns_some() {
        assert!(explain_error("E002").is_some());
    }

    #[test]
    fn test_e003_returns_some() {
        assert!(explain_error("E003").is_some());
    }

    #[test]
    fn test_e004_returns_some() {
        assert!(explain_error("E004").is_some());
    }

    #[test]
    fn test_e005_returns_some() {
        assert!(explain_error("E005").is_some());
    }

    #[test]
    fn test_e006_returns_some() {
        assert!(explain_error("E006").is_some());
    }

    #[test]
    fn test_unknown_returns_none() {
        assert!(explain_error("E999").is_none());
        assert!(explain_error("X001").is_none());
        assert!(explain_error("").is_none());
    }

    #[test]
    fn test_case_insensitive_lookup() {
        assert!(explain_error("e001").is_some());
        assert!(explain_error("e003").is_some());
    }

    #[test]
    fn test_all_codes_have_solutions() {
        let catalog = build_catalog();
        for error in &catalog {
            assert!(
                !error.solutions.is_empty(),
                "Error {} has no solutions",
                error.code,
            );
        }
    }

    #[test]
    fn test_all_codes_have_causes() {
        let catalog = build_catalog();
        for error in &catalog {
            assert!(
                !error.causes.is_empty(),
                "Error {} has no causes",
                error.code,
            );
        }
    }

    #[test]
    fn test_descriptions_non_empty() {
        let catalog = build_catalog();
        for error in &catalog {
            assert!(
                !error.description.is_empty(),
                "Error {} has empty description",
                error.code,
            );
        }
    }

    #[test]
    fn test_titles_non_empty() {
        let catalog = build_catalog();
        for error in &catalog {
            assert!(
                !error.title.is_empty(),
                "Error {} has empty title",
                error.code,
            );
        }
    }

    #[test]
    fn test_all_error_codes_returns_six() {
        let codes = all_error_codes();
        assert_eq!(codes.len(), 6);
        assert!(codes.contains(&"E001".to_string()));
        assert!(codes.contains(&"E006".to_string()));
    }

    #[test]
    fn test_format_explanation_contains_sections() {
        let error = explain_error("E001").unwrap();
        let formatted = format_explanation(&error);
        assert!(formatted.contains("E001"));
        assert!(formatted.contains("Invalid magic bytes"));
        assert!(formatted.contains("Possible causes:"));
        assert!(formatted.contains("Solutions:"));
        assert!(formatted.contains("Related errors:"));
    }

    #[test]
    fn test_related_codes_exist_in_catalog() {
        let catalog = build_catalog();
        let all_codes: Vec<String> = catalog.iter().map(|e| e.code.clone()).collect();
        for error in &catalog {
            for related in &error.related {
                assert!(
                    all_codes.contains(related),
                    "Error {} references unknown related code {}",
                    error.code,
                    related,
                );
            }
        }
    }
}
