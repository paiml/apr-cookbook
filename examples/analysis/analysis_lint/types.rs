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
use std::fmt;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintLevel {
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
pub struct LintRule {
    pub id: &'static str,
    pub level: LintLevel,
    pub category: &'static str,
    pub description: &'static str,
}

#[derive(Debug, Clone)]
pub struct LintIssue {
    pub rule_id: String,
    pub level: LintLevel,
    pub message: String,
    pub suggestion: String,
}

#[derive(Debug, Clone)]
pub struct LintReport {
    pub model_name: String,
    pub issues: Vec<LintIssue>,
}

impl LintReport {
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            issues: Vec::new(),
        }
    }

    pub fn add(&mut self, issue: LintIssue) {
        self.issues.push(issue);
    }

    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.level == LintLevel::Error)
            .count()
    }

    pub fn warn_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.level == LintLevel::Warn)
            .count()
    }

    pub fn info_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.level == LintLevel::Info)
            .count()
    }

    pub fn passed(&self) -> bool {
        self.error_count() == 0
    }

    pub fn verdict(&self) -> &str {
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
pub struct TensorMeta {
    pub name: String,
    pub dtype: String,
    pub param_count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ModelMeta {
    pub name: String,
    pub model_type: Option<String>,
    pub version: Option<String>,
    pub compression: Option<String>,
    pub quantization: Option<String>,
    pub tokenizer_ref: Option<String>,
    pub total_params: usize,
    pub tensors: Vec<TensorMeta>,
    pub embedding_tied: bool,
}

// ---------------------------------------------------------------------------
// Rule table
// ---------------------------------------------------------------------------

pub fn lint_rules() -> Vec<LintRule> {
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

pub const LARGE_MODEL_THRESHOLD: usize = 1_000_000;
pub const LARGE_EMBEDDING_THRESHOLD: usize = 100_000;

pub fn check_compression(meta: &ModelMeta, report: &mut LintReport) {
    if meta.compression.is_none() {
        report.add(LintIssue {
            rule_id: "L001".to_string(),
            level: LintLevel::Warn,
            message: "Model has no compression metadata set".to_string(),
            suggestion: "Add compression (LZ4 or Zstd) to reduce model size".to_string(),
        });
    }
}

pub fn check_quantization(meta: &ModelMeta, report: &mut LintReport) {
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

pub fn check_model_type(meta: &ModelMeta, report: &mut LintReport) {
    if meta.model_type.is_none() {
        report.add(LintIssue {
            rule_id: "L003".to_string(),
            level: LintLevel::Error,
            message: "Required field 'model_type' is missing".to_string(),
            suggestion: "Set model_type (e.g., 'transformer', 'cnn', 'linear')".to_string(),
        });
    }
}

pub fn is_standard_name(name: &str) -> bool {
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

pub fn check_tensor_naming(meta: &ModelMeta, report: &mut LintReport) {
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

pub fn check_dtype_consistency(meta: &ModelMeta, report: &mut LintReport) {
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

pub fn check_tokenizer_ref(meta: &ModelMeta, report: &mut LintReport) {
    if meta.tokenizer_ref.is_none() {
        report.add(LintIssue {
            rule_id: "L006".to_string(),
            level: LintLevel::Info,
            message: "No tokenizer reference found in model metadata".to_string(),
            suggestion: "Add tokenizer_ref for reproducible text preprocessing".to_string(),
        });
    }
}

pub fn check_embedding_tying(meta: &ModelMeta, report: &mut LintReport) {
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

pub fn check_version_field(meta: &ModelMeta, report: &mut LintReport) {
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

pub fn lint_model(meta: &ModelMeta) -> LintReport {
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

pub fn print_rule_table(rules: &[LintRule]) {
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

pub fn print_lint_report(report: &LintReport) {
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

pub fn print_summary(report: &LintReport) {
    println!("\n--- Summary for '{}' ---", report.model_name);
    println!("  Errors:   {}", report.error_count());
    println!("  Warnings: {}", report.warn_count());
    println!("  Info:     {}", report.info_count());
    println!("  Verdict:  {}", report.verdict());
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
