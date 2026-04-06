//! # APR Model Lint
//!
//! CLI equivalent: `apr lint model.apr`
//!
//! Runs static quality checks on model metadata for best practices.
//! Each lint rule checks a specific aspect of the model (compression,
//! quantization, naming conventions, dtype consistency, etc.) and reports
//! findings with severity, message, and actionable suggestion.
//!
//!
//! ## Format Variants
//! ```bash
//! apr lint model.apr          # APR native format
//! apr lint model.gguf         # GGUF (llama.cpp compatible)
//! apr lint model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use std::fmt;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LintLevel {
    Info,
    Warn,
    Error,
}

impl fmt::Display for LintLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LintLevel::Info => write!(f, "INFO"),
            LintLevel::Warn => write!(f, "WARN"),
            LintLevel::Error => write!(f, "ERROR"),
        }
    }
}

#[derive(Debug, Clone)]
struct LintRule {
    id: &'static str,
    level: LintLevel,
    category: &'static str,
    description: &'static str,
}

#[derive(Debug, Clone)]
struct LintIssue {
    rule_id: String,
    level: LintLevel,
    message: String,
    suggestion: String,
}

#[derive(Debug, Clone)]
struct LintReport {
    model_name: String,
    issues: Vec<LintIssue>,
}

impl LintReport {
    fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            issues: Vec::new(),
        }
    }

    fn add(&mut self, issue: LintIssue) {
        self.issues.push(issue);
    }

    fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.level == LintLevel::Error)
            .count()
    }

    fn warn_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.level == LintLevel::Warn)
            .count()
    }

    fn info_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.level == LintLevel::Info)
            .count()
    }

    fn passed(&self) -> bool {
        self.error_count() == 0
    }

    fn verdict(&self) -> &str {
        if self.passed() {
            "PASS"
        } else {
            "FAIL"
        }
    }
}

// ---------------------------------------------------------------------------
// Synthetic model metadata (the structure we lint against)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TensorMeta {
    name: String,
    dtype: String,
    param_count: usize,
}

#[derive(Debug, Clone, Default)]
struct ModelMeta {
    name: String,
    model_type: Option<String>,
    version: Option<String>,
    compression: Option<String>,
    quantization: Option<String>,
    tokenizer_ref: Option<String>,
    total_params: usize,
    tensors: Vec<TensorMeta>,
    embedding_tied: bool,
}

// ---------------------------------------------------------------------------
// Rule table
// ---------------------------------------------------------------------------

fn lint_rules() -> Vec<LintRule> {
    vec![
        LintRule {
            id: "L001",
            level: LintLevel::Warn,
            category: "compression",
            description: "Missing compression metadata",
        },
        LintRule {
            id: "L002",
            level: LintLevel::Info,
            category: "quantization",
            description: "No quantization applied to large model",
        },
        LintRule {
            id: "L003",
            level: LintLevel::Error,
            category: "metadata",
            description: "Missing model_type field",
        },
        LintRule {
            id: "L004",
            level: LintLevel::Warn,
            category: "naming",
            description: "Non-standard tensor naming",
        },
        LintRule {
            id: "L005",
            level: LintLevel::Warn,
            category: "dtype",
            description: "Tensor dtype inconsistency (mixed f32/f16)",
        },
        LintRule {
            id: "L006",
            level: LintLevel::Info,
            category: "tokenizer",
            description: "Missing tokenizer reference",
        },
        LintRule {
            id: "L007",
            level: LintLevel::Info,
            category: "embedding",
            description: "Large embedding table without tying",
        },
        LintRule {
            id: "L008",
            level: LintLevel::Error,
            category: "metadata",
            description: "Version field missing",
        },
    ]
}

// ---------------------------------------------------------------------------
// Individual lint checks
// ---------------------------------------------------------------------------

const LARGE_MODEL_THRESHOLD: usize = 1_000_000;
const LARGE_EMBEDDING_THRESHOLD: usize = 100_000;

fn check_compression(meta: &ModelMeta, report: &mut LintReport) {
    if meta.compression.is_none() {
        report.add(LintIssue {
            rule_id: "L001".to_string(),
            level: LintLevel::Warn,
            message: "Model has no compression metadata set".to_string(),
            suggestion: "Add compression (LZ4 or Zstd) to reduce model size".to_string(),
        });
    }
}

fn check_quantization(meta: &ModelMeta, report: &mut LintReport) {
    let is_large = meta.total_params >= LARGE_MODEL_THRESHOLD;
    let has_quant = meta.quantization.is_some();
    if is_large && !has_quant {
        report.add(LintIssue {
            rule_id: "L002".to_string(),
            level: LintLevel::Info,
            message: format!(
                "Model has {} params but no quantization applied",
                meta.total_params
            ),
            suggestion: "Consider INT8 or INT4 quantization for deployment".to_string(),
        });
    }
}

fn check_model_type(meta: &ModelMeta, report: &mut LintReport) {
    if meta.model_type.is_none() {
        report.add(LintIssue {
            rule_id: "L003".to_string(),
            level: LintLevel::Error,
            message: "Required field 'model_type' is missing".to_string(),
            suggestion: "Set model_type (e.g., 'transformer', 'cnn', 'linear')".to_string(),
        });
    }
}

fn is_standard_name(name: &str) -> bool {
    // Standard patterns: dotted hierarchy like layer0.attn.weight
    let valid_chars = name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_');
    let no_spaces = !name.contains(' ');
    let no_upper_start = name
        .split('.')
        .all(|seg| seg.starts_with(|c: char| c.is_ascii_lowercase() || c.is_ascii_digit()));
    valid_chars && no_spaces && no_upper_start
}

fn check_tensor_naming(meta: &ModelMeta, report: &mut LintReport) {
    for tensor in &meta.tensors {
        if !is_standard_name(&tensor.name) {
            report.add(LintIssue {
                rule_id: "L004".to_string(),
                level: LintLevel::Warn,
                message: format!("Tensor '{}' uses non-standard naming", tensor.name),
                suggestion: "Use dotted lowercase names (e.g., 'layer0.attn.weight')".to_string(),
            });
        }
    }
}

fn check_dtype_consistency(meta: &ModelMeta, report: &mut LintReport) {
    if meta.tensors.len() < 2 {
        return;
    }
    let first_dtype = &meta.tensors[0].dtype;
    let mixed = meta.tensors.iter().any(|t| t.dtype != *first_dtype);
    if mixed {
        let dtypes: Vec<&str> = meta.tensors.iter().map(|t| t.dtype.as_str()).collect();
        report.add(LintIssue {
            rule_id: "L005".to_string(),
            level: LintLevel::Warn,
            message: format!("Mixed dtypes detected: {:?}", dtypes),
            suggestion: "Use uniform dtype or document mixed-precision intent".to_string(),
        });
    }
}

fn check_tokenizer_ref(meta: &ModelMeta, report: &mut LintReport) {
    if meta.tokenizer_ref.is_none() {
        report.add(LintIssue {
            rule_id: "L006".to_string(),
            level: LintLevel::Info,
            message: "No tokenizer reference found in model metadata".to_string(),
            suggestion: "Add tokenizer_ref for reproducible text preprocessing".to_string(),
        });
    }
}

fn check_embedding_tying(meta: &ModelMeta, report: &mut LintReport) {
    let has_large_embed = meta
        .tensors
        .iter()
        .any(|t| t.name.contains("embed") && t.param_count >= LARGE_EMBEDDING_THRESHOLD);
    if has_large_embed && !meta.embedding_tied {
        report.add(LintIssue {
            rule_id: "L007".to_string(),
            level: LintLevel::Info,
            message: "Large embedding table detected without weight tying".to_string(),
            suggestion: "Tie input/output embeddings to reduce model size".to_string(),
        });
    }
}

fn check_version_field(meta: &ModelMeta, report: &mut LintReport) {
    if meta.version.is_none() {
        report.add(LintIssue {
            rule_id: "L008".to_string(),
            level: LintLevel::Error,
            message: "Required field 'version' is missing".to_string(),
            suggestion: "Add a semver version string (e.g., '1.0.0')".to_string(),
        });
    }
}

// ---------------------------------------------------------------------------
// Lint orchestrator
// ---------------------------------------------------------------------------

fn lint_model(meta: &ModelMeta) -> LintReport {
    let mut report = LintReport::new(&meta.name);
    check_compression(meta, &mut report);
    check_quantization(meta, &mut report);
    check_model_type(meta, &mut report);
    check_tensor_naming(meta, &mut report);
    check_dtype_consistency(meta, &mut report);
    check_tokenizer_ref(meta, &mut report);
    check_embedding_tying(meta, &mut report);
    check_version_field(meta, &mut report);
    report
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

fn print_rule_table(rules: &[LintRule]) {
    println!(
        "{:<6} {:<8} {:<14} Description",
        "Rule", "Level", "Category"
    );
    println!("{}", "-".repeat(72));
    for rule in rules {
        println!(
            "{:<6} {:<8} {:<14} {}",
            rule.id, rule.level, rule.category, rule.description,
        );
    }
}

fn print_lint_report(report: &LintReport) {
    println!(
        "\n{:<8} {:<8} {:<40} Suggestion",
        "Level", "Rule", "Message"
    );
    println!("{}", "-".repeat(96));
    for issue in &report.issues {
        println!(
            "{:<8} {:<8} {:<40} {}",
            issue.level, issue.rule_id, issue.message, issue.suggestion,
        );
    }
}

fn print_summary(report: &LintReport) {
    println!("\n--- Summary for '{}' ---", report.model_name);
    println!("  Errors:   {}", report.error_count());
    println!("  Warnings: {}", report.warn_count());
    println!("  Info:     {}", report.info_count());
    println!("  Verdict:  {}", report.verdict());
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_lint")?;

    // --- Section 1: Rule table ---
    println!("=== APR Model Lint ===\n");
    println!("--- Lint Rules ---\n");
    let rules = lint_rules();
    print_rule_table(&rules);

    // --- Section 2: Build a synthetic model bundle for context ---
    let seed = hash_name_to_seed("lint-model");
    let weight_bytes = generate_model_payload(seed, 64 * 64);
    let bias_bytes = generate_model_payload(seed + 1, 64);

    let bundle = ModelBundleV2::new()
        .with_name("lint-target")
        .with_description("Model for lint demo")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("weight", vec![64, 64], weight_bytes)
        .add_tensor("bias", vec![64], bias_bytes)
        .build();

    let model_path = ctx.path("lint-target.apr");
    std::fs::write(&model_path, &bundle)?;
    println!(
        "\nCreated test model: {} ({} bytes)",
        model_path.display(),
        bundle.len()
    );

    // --- Section 3: Lint a well-formed model ---
    println!("\n--- Linting Well-Formed Model ---");
    let good_meta = ModelMeta {
        name: "good-model".to_string(),
        model_type: Some("transformer".to_string()),
        version: Some("1.0.0".to_string()),
        compression: Some("lz4".to_string()),
        quantization: Some("int8".to_string()),
        tokenizer_ref: Some("bpe-50k".to_string()),
        total_params: 500_000,
        tensors: vec![
            TensorMeta {
                name: "layer0.attn.weight".to_string(),
                dtype: "f32".to_string(),
                param_count: 250_000,
            },
            TensorMeta {
                name: "layer0.ffn.weight".to_string(),
                dtype: "f32".to_string(),
                param_count: 250_000,
            },
        ],
        embedding_tied: true,
    };
    let good_report = lint_model(&good_meta);
    if good_report.issues.is_empty() {
        println!("  No issues found.");
    } else {
        print_lint_report(&good_report);
    }
    print_summary(&good_report);

    // --- Section 4: Lint a model with issues ---
    println!("\n--- Linting Model with Issues ---");
    let bad_meta = ModelMeta {
        name: "bad-model".to_string(),
        model_type: None,    // L003: Error
        version: None,       // L008: Error
        compression: None,   // L001: Warn
        quantization: None,  // L002: Info (large)
        tokenizer_ref: None, // L006: Info
        total_params: 2_000_000,
        tensors: vec![
            TensorMeta {
                name: "Embed Table".to_string(), // L004: non-standard
                dtype: "f32".to_string(),
                param_count: 500_000,
            },
            TensorMeta {
                name: "layer0.attn.weight".to_string(),
                dtype: "f16".to_string(), // L005: mixed dtype
                param_count: 1_000_000,
            },
            TensorMeta {
                name: "output.embed".to_string(),
                dtype: "f32".to_string(),
                param_count: 500_000, // L007: large embed, not tied
            },
        ],
        embedding_tied: false,
    };
    let bad_report = lint_model(&bad_meta);
    print_lint_report(&bad_report);
    print_summary(&bad_report);

    // --- Section 5: Overall results ---
    println!("\n--- Overall ---");
    println!(
        "  Well-formed model: {} ({} issues)",
        good_report.verdict(),
        good_report.issues.len()
    );
    println!(
        "  Problem model:     {} ({} issues)",
        bad_report.verdict(),
        bad_report.issues.len()
    );

    assert!(good_report.passed(), "Well-formed model must pass lint");
    assert!(!bad_report.passed(), "Problem model must fail lint");

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clean_meta() -> ModelMeta {
        ModelMeta {
            name: "test-clean".to_string(),
            model_type: Some("linear".to_string()),
            version: Some("0.1.0".to_string()),
            compression: Some("lz4".to_string()),
            quantization: Some("fp32".to_string()),
            tokenizer_ref: Some("bpe-32k".to_string()),
            total_params: 1000,
            tensors: vec![TensorMeta {
                name: "layer0.weight".to_string(),
                dtype: "f32".to_string(),
                param_count: 1000,
            }],
            embedding_tied: true,
        }
    }

    #[test]
    fn test_clean_model_no_issues() {
        let report = lint_model(&make_clean_meta());
        assert!(
            report.issues.is_empty(),
            "Clean model should have no issues"
        );
        assert!(report.passed());
        assert_eq!(report.verdict(), "PASS");
    }

    #[test]
    fn test_missing_model_type_is_error() {
        let mut meta = make_clean_meta();
        meta.model_type = None;
        let report = lint_model(&meta);
        let l003 = report.issues.iter().find(|i| i.rule_id == "L003");
        assert!(l003.is_some(), "L003 should fire when model_type is None");
        assert_eq!(l003.map(|i| i.level), Some(LintLevel::Error));
        assert!(!report.passed());
    }

    #[test]
    fn test_missing_version_is_error() {
        let mut meta = make_clean_meta();
        meta.version = None;
        let report = lint_model(&meta);
        let l008 = report.issues.iter().find(|i| i.rule_id == "L008");
        assert!(l008.is_some(), "L008 should fire when version is None");
        assert_eq!(l008.map(|i| i.level), Some(LintLevel::Error));
    }

    #[test]
    fn test_missing_compression_is_warn() {
        let mut meta = make_clean_meta();
        meta.compression = None;
        let report = lint_model(&meta);
        let l001 = report.issues.iter().find(|i| i.rule_id == "L001");
        assert!(l001.is_some(), "L001 should fire when compression is None");
        assert_eq!(l001.map(|i| i.level), Some(LintLevel::Warn));
    }

    #[test]
    fn test_large_model_no_quant_is_info() {
        let mut meta = make_clean_meta();
        meta.total_params = 2_000_000;
        meta.quantization = None;
        let report = lint_model(&meta);
        let l002 = report.issues.iter().find(|i| i.rule_id == "L002");
        assert!(
            l002.is_some(),
            "L002 should fire for large unquantized models"
        );
        assert_eq!(l002.map(|i| i.level), Some(LintLevel::Info));
    }

    #[test]
    fn test_small_model_no_quant_ok() {
        let mut meta = make_clean_meta();
        meta.total_params = 500;
        meta.quantization = None;
        let report = lint_model(&meta);
        let l002 = report.issues.iter().find(|i| i.rule_id == "L002");
        assert!(l002.is_none(), "L002 should NOT fire for small models");
    }

    #[test]
    fn test_non_standard_tensor_name_is_warn() {
        let mut meta = make_clean_meta();
        meta.tensors = vec![TensorMeta {
            name: "My Weight".to_string(),
            dtype: "f32".to_string(),
            param_count: 100,
        }];
        let report = lint_model(&meta);
        let l004 = report.issues.iter().find(|i| i.rule_id == "L004");
        assert!(l004.is_some(), "L004 should fire for non-standard names");
        assert_eq!(l004.map(|i| i.level), Some(LintLevel::Warn));
    }

    #[test]
    fn test_mixed_dtype_is_warn() {
        let mut meta = make_clean_meta();
        meta.tensors = vec![
            TensorMeta {
                name: "layer0.weight".to_string(),
                dtype: "f32".to_string(),
                param_count: 500,
            },
            TensorMeta {
                name: "layer1.weight".to_string(),
                dtype: "f16".to_string(),
                param_count: 500,
            },
        ];
        let report = lint_model(&meta);
        let l005 = report.issues.iter().find(|i| i.rule_id == "L005");
        assert!(l005.is_some(), "L005 should fire for mixed dtypes");
        assert_eq!(l005.map(|i| i.level), Some(LintLevel::Warn));
    }

    #[test]
    fn test_missing_tokenizer_ref_is_info() {
        let mut meta = make_clean_meta();
        meta.tokenizer_ref = None;
        let report = lint_model(&meta);
        let l006 = report.issues.iter().find(|i| i.rule_id == "L006");
        assert!(
            l006.is_some(),
            "L006 should fire when tokenizer_ref is None"
        );
        assert_eq!(l006.map(|i| i.level), Some(LintLevel::Info));
    }

    #[test]
    fn test_large_embed_untied_is_info() {
        let mut meta = make_clean_meta();
        meta.embedding_tied = false;
        meta.tensors = vec![TensorMeta {
            name: "embed.weight".to_string(),
            dtype: "f32".to_string(),
            param_count: 200_000,
        }];
        let report = lint_model(&meta);
        let l007 = report.issues.iter().find(|i| i.rule_id == "L007");
        assert!(
            l007.is_some(),
            "L007 should fire for large untied embeddings"
        );
        assert_eq!(l007.map(|i| i.level), Some(LintLevel::Info));
    }

    #[test]
    fn test_report_counts() {
        let meta = ModelMeta {
            name: "count-test".to_string(),
            model_type: None,  // L003 Error
            version: None,     // L008 Error
            compression: None, // L001 Warn
            quantization: None,
            tokenizer_ref: None, // L006 Info
            total_params: 100,
            tensors: vec![TensorMeta {
                name: "w".to_string(),
                dtype: "f32".to_string(),
                param_count: 100,
            }],
            embedding_tied: true,
        };
        let report = lint_model(&meta);
        assert_eq!(report.error_count(), 2, "Expected 2 errors (L003, L008)");
        assert_eq!(report.warn_count(), 1, "Expected 1 warning (L001)");
        assert_eq!(report.info_count(), 1, "Expected 1 info (L006)");
        assert!(!report.passed());
        assert_eq!(report.verdict(), "FAIL");
    }
}
