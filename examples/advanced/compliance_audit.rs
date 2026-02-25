//! # Recipe: Compliance Audit Pipeline
//!
//! **Category**: Advanced - Governance & Compliance
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## Learning Objective
//! Demonstrates a full compliance audit pipeline for model deployment approval,
//! composing five stages: inspect, oracle, qualify, qa, and report.
//!
//! ## Pipeline Architecture
//!
//! ```text
//! Model Bytes
//!   |
//!   v
//! [1. Inspect] --> metadata, tensor count, dtype distribution
//!   |
//!   v
//! [2. Oracle]  --> architecture family detection (LLaMA, GPT, BERT)
//!   |
//!   v
//! [3. Qualify] --> 8 qualification gates (format, NaN, Inf, size, ...)
//!   |
//!   v
//! [4. QA]      --> 4 quality-of-service gates (latency, accuracy, ...)
//!   |
//!   v
//! [5. Report]  --> compliance certificate with verdict
//! ```
//!
//! ## Run Command
//! ```bash
//! cargo run --example compliance_audit
//! ```
//!
//! ## Toyota Way Principles
//! - **Jidoka** (Quality built-in): Automated gates prevent non-compliant deployments
//! - **Poka-yoke** (Error-proofing): 12 gates catch issues before production
//! - **Genchi Genbutsu** (Go and see): Direct byte-level model inspection

use apr_cookbook::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::fmt;

// ============================================================================
// Data Structures
// ============================================================================

/// Severity level for audit findings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Severity {
    /// Informational observation, no action required
    Info,
    /// Warning that may require attention
    Warning,
    /// Critical issue that blocks deployment
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Critical => write!(f, "CRIT"),
        }
    }
}

/// A single finding from an audit stage.
#[derive(Debug, Clone)]
pub struct Finding {
    /// Severity level
    pub severity: Severity,
    /// Machine-readable finding code (e.g., "QUAL-001")
    pub code: String,
    /// Human-readable description
    pub message: String,
    /// Optional remediation suggestion
    pub remediation: Option<String>,
}

/// A stage in the audit pipeline with collected findings.
#[derive(Debug, Clone)]
pub struct AuditStage {
    /// Stage name (e.g., "Inspect", "Oracle")
    pub name: String,
    /// Findings collected during this stage
    pub findings: Vec<Finding>,
}

impl AuditStage {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            findings: Vec::new(),
        }
    }

    fn add(&mut self, severity: Severity, code: &str, message: &str) {
        self.findings.push(Finding {
            severity,
            code: code.to_string(),
            message: message.to_string(),
            remediation: None,
        });
    }

    fn add_with_fix(&mut self, severity: Severity, code: &str, message: &str, fix: &str) {
        self.findings.push(Finding {
            severity,
            code: code.to_string(),
            message: message.to_string(),
            remediation: Some(fix.to_string()),
        });
    }

    fn critical_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == Severity::Critical)
            .count()
    }
}

/// Final compliance verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplianceVerdict {
    /// All gates passed, model approved for deployment
    Compliant,
    /// Minor issues found, deployment allowed with conditions
    Conditional,
    /// Critical issues found, deployment blocked
    NonCompliant,
}

impl fmt::Display for ComplianceVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compliant => write!(f, "COMPLIANT"),
            Self::Conditional => write!(f, "CONDITIONAL"),
            Self::NonCompliant => write!(f, "NON-COMPLIANT"),
        }
    }
}

/// Complete audit report with all stages and final verdict.
#[derive(Debug, Clone)]
pub struct AuditReport {
    /// Model name under audit
    pub model_name: String,
    /// Audit stages with findings
    pub stages: Vec<AuditStage>,
    /// Final compliance verdict
    pub verdict: ComplianceVerdict,
    /// Qualification score (passed / 8 total)
    pub qualification_score: (u8, u8),
    /// QA score (passed / 4 total)
    pub qa_score: (u8, u8),
}

/// Simulated model metadata for audit.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Model display name
    pub name: String,
    /// Format magic bytes
    pub magic: [u8; 4],
    /// Format version string
    pub format_version: String,
    /// Number of tensors in the model
    pub tensor_count: usize,
    /// Total parameter count
    pub total_params: u64,
    /// Distribution of data types
    pub dtype_distribution: HashMap<String, f32>,
    /// Compression type (e.g., "none", "lz4", "zstd")
    pub compression: String,
    /// Tensor names (for architecture detection)
    pub tensor_names: Vec<String>,
    /// Tensor shapes (parallel to tensor_names)
    pub tensor_shapes: Vec<Vec<usize>>,
    /// Metadata key-value pairs
    pub metadata: HashMap<String, String>,
    /// Whether weights contain NaN values
    pub has_nan: bool,
    /// Whether weights contain Inf values
    pub has_inf: bool,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Simulated checksum validity
    pub checksum_valid: bool,
    /// Simulated inference latency in milliseconds
    pub latency_ms: f64,
    /// Simulated accuracy (0.0-1.0)
    pub accuracy: f64,
    /// Simulated memory footprint in MB
    pub memory_mb: f64,
    /// Simulated throughput in tokens/sec
    pub throughput_tps: f64,
}

// ============================================================================
// Stage 1: Inspect
// ============================================================================

/// Extract model metadata and flag missing required fields.
fn stage_inspect(spec: &ModelSpec) -> AuditStage {
    let mut stage = AuditStage::new("Inspect");

    // Report basic metadata
    stage.add(
        Severity::Info,
        "INS-001",
        &format!("Format version: {}", spec.format_version),
    );
    stage.add(
        Severity::Info,
        "INS-002",
        &format!("Tensor count: {}", spec.tensor_count),
    );
    stage.add(
        Severity::Info,
        "INS-003",
        &format!("Total parameters: {}", spec.total_params),
    );

    // Dtype distribution
    let dtypes: Vec<String> = spec
        .dtype_distribution
        .iter()
        .map(|(k, v)| format!("{}={:.0}%", k, v * 100.0))
        .collect();
    stage.add(
        Severity::Info,
        "INS-004",
        &format!("Dtype distribution: {}", dtypes.join(", ")),
    );

    stage.add(
        Severity::Info,
        "INS-005",
        &format!("Compression: {}", spec.compression),
    );

    // Flag missing required metadata fields
    if !spec.metadata.contains_key("model_type") {
        stage.add_with_fix(
            Severity::Critical,
            "INS-010",
            "Missing required metadata field: model_type",
            "Add 'model_type' to model metadata before export",
        );
    }
    if !spec.metadata.contains_key("version") {
        stage.add_with_fix(
            Severity::Critical,
            "INS-011",
            "Missing required metadata field: version",
            "Add 'version' to model metadata before export",
        );
    }

    stage
}

// ============================================================================
// Stage 2: Oracle
// ============================================================================

/// Approved architecture families for deployment.
const APPROVED_ARCHITECTURES: &[&str] = &["LLaMA", "GPT", "BERT"];

/// Detect model architecture family from tensor naming patterns.
fn detect_architecture(tensor_names: &[String]) -> &'static str {
    let joined = tensor_names.join(" ");

    if joined.contains("self_attn.q_proj") || joined.contains("self_attn.k_proj") {
        return "LLaMA";
    }
    if joined.contains("attn.c_attn") || joined.contains("attn.c_proj") {
        return "GPT";
    }
    if joined.contains("attention.self.query") || joined.contains("attention.self.key") {
        return "BERT";
    }
    "Unknown"
}

/// Identify model family and verify it is on the approved list.
fn stage_oracle(spec: &ModelSpec) -> AuditStage {
    let mut stage = AuditStage::new("Oracle");

    let family = detect_architecture(&spec.tensor_names);
    stage.add(
        Severity::Info,
        "ORA-001",
        &format!("Detected architecture family: {family}"),
    );

    if APPROVED_ARCHITECTURES.contains(&family) {
        stage.add(
            Severity::Info,
            "ORA-002",
            &format!("{family} is on the approved architecture list"),
        );
    } else {
        stage.add_with_fix(
            Severity::Critical,
            "ORA-010",
            &format!("{family} is NOT on the approved architecture list"),
            &format!(
                "Convert model to an approved architecture: {}",
                APPROVED_ARCHITECTURES.join(", ")
            ),
        );
    }

    stage
}

// ============================================================================
// Stage 3: Qualify (8 gates)
// ============================================================================

/// Supported data types for deployment.
const SUPPORTED_DTYPES: &[&str] = &["FP32", "FP16", "INT8"];

/// Maximum allowed model size in bytes (10 GB).
const MAX_SIZE_BYTES: u64 = 10 * 1024 * 1024 * 1024;

/// Run 8 qualification gates and return findings.
fn stage_qualify(spec: &ModelSpec) -> (AuditStage, u8) {
    let mut stage = AuditStage::new("Qualify");
    let mut passed: u8 = 0;

    // Gate 1: Format validity (APR2 magic)
    if spec.magic == *b"APR2" {
        stage.add(
            Severity::Info,
            "QAL-001",
            "Gate 1 PASS: APR2 magic bytes valid",
        );
        passed += 1;
    } else {
        stage.add_with_fix(
            Severity::Critical,
            "QAL-001",
            &format!(
                "Gate 1 FAIL: Invalid magic bytes (expected APR2, got {:?})",
                std::str::from_utf8(&spec.magic).unwrap_or("????")
            ),
            "Re-export model using aprender with APR2 format",
        );
    }

    // Gate 2: Header parseable
    if !spec.format_version.is_empty() && spec.tensor_count > 0 {
        stage.add(Severity::Info, "QAL-002", "Gate 2 PASS: Header parseable");
        passed += 1;
    } else {
        stage.add_with_fix(
            Severity::Critical,
            "QAL-002",
            "Gate 2 FAIL: Header not parseable (missing version or zero tensors)",
            "Verify model export completed without errors",
        );
    }

    // Gate 3: No NaN in weights
    if spec.has_nan {
        stage.add_with_fix(
            Severity::Critical,
            "QAL-003",
            "Gate 3 FAIL: NaN values detected in model weights",
            "Re-train with gradient clipping or check for numerical instability",
        );
    } else {
        stage.add(
            Severity::Info,
            "QAL-003",
            "Gate 3 PASS: No NaN values in weights",
        );
        passed += 1;
    }

    // Gate 4: No Inf in weights
    if spec.has_inf {
        stage.add_with_fix(
            Severity::Critical,
            "QAL-004",
            "Gate 4 FAIL: Inf values detected in model weights",
            "Re-train with loss scaling or reduce learning rate",
        );
    } else {
        stage.add(
            Severity::Info,
            "QAL-004",
            "Gate 4 PASS: No Inf values in weights",
        );
        passed += 1;
    }

    // Gate 5: Size within budget
    if spec.size_bytes < MAX_SIZE_BYTES {
        stage.add(
            Severity::Info,
            "QAL-005",
            &format!(
                "Gate 5 PASS: Size {:.2} GB within 10 GB budget",
                spec.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            ),
        );
        passed += 1;
    } else {
        stage.add_with_fix(
            Severity::Critical,
            "QAL-005",
            &format!(
                "Gate 5 FAIL: Size {:.2} GB exceeds 10 GB budget",
                spec.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            ),
            "Apply quantization (INT8/INT4) or pruning to reduce model size",
        );
    }

    // Gate 6: Dtype supported
    let all_supported = spec
        .dtype_distribution
        .keys()
        .all(|dt| SUPPORTED_DTYPES.contains(&dt.as_str()));
    if all_supported {
        stage.add(
            Severity::Info,
            "QAL-006",
            "Gate 6 PASS: All dtypes supported",
        );
        passed += 1;
    } else {
        let unsupported: Vec<&String> = spec
            .dtype_distribution
            .keys()
            .filter(|dt| !SUPPORTED_DTYPES.contains(&dt.as_str()))
            .collect();
        stage.add_with_fix(
            Severity::Critical,
            "QAL-006",
            &format!("Gate 6 FAIL: Unsupported dtypes: {:?}", unsupported),
            &format!(
                "Convert to a supported dtype: {}",
                SUPPORTED_DTYPES.join(", ")
            ),
        );
    }

    // Gate 7: Metadata complete
    let has_model_type = spec.metadata.contains_key("model_type");
    let has_version = spec.metadata.contains_key("version");
    if has_model_type && has_version {
        stage.add(
            Severity::Info,
            "QAL-007",
            "Gate 7 PASS: Required metadata fields present (model_type, version)",
        );
        passed += 1;
    } else {
        let missing: Vec<&str> = [
            (!has_model_type).then_some("model_type"),
            (!has_version).then_some("version"),
        ]
        .into_iter()
        .flatten()
        .collect();
        stage.add_with_fix(
            Severity::Critical,
            "QAL-007",
            &format!(
                "Gate 7 FAIL: Missing metadata fields: {}",
                missing.join(", ")
            ),
            "Add missing metadata fields before model export",
        );
    }

    // Gate 8: Checksum valid
    if spec.checksum_valid {
        stage.add(Severity::Info, "QAL-008", "Gate 8 PASS: Checksum valid");
        passed += 1;
    } else {
        stage.add_with_fix(
            Severity::Critical,
            "QAL-008",
            "Gate 8 FAIL: Checksum mismatch (possible data corruption)",
            "Re-export model and verify file integrity after transfer",
        );
    }

    (stage, passed)
}

// ============================================================================
// Stage 4: QA (4 gates)
// ============================================================================

/// Maximum acceptable inference latency in milliseconds.
const LATENCY_BUDGET_MS: f64 = 100.0;

/// Minimum acceptable accuracy.
const ACCURACY_THRESHOLD: f64 = 0.85;

/// Maximum acceptable memory footprint in MB.
const MEMORY_LIMIT_MB: f64 = 4096.0;

/// Minimum acceptable throughput in tokens/sec.
const MIN_THROUGHPUT_TPS: f64 = 50.0;

/// Run 4 quality-of-service gates.
fn stage_qa(spec: &ModelSpec) -> (AuditStage, u8) {
    let mut stage = AuditStage::new("QA");
    let mut passed: u8 = 0;

    // Gate 1: Inference latency < budget
    if spec.latency_ms < LATENCY_BUDGET_MS {
        stage.add(
            Severity::Info,
            "QA-001",
            &format!(
                "Gate 1 PASS: Latency {:.1}ms < {:.1}ms budget",
                spec.latency_ms, LATENCY_BUDGET_MS
            ),
        );
        passed += 1;
    } else {
        stage.add_with_fix(
            Severity::Critical,
            "QA-001",
            &format!(
                "Gate 1 FAIL: Latency {:.1}ms exceeds {:.1}ms budget",
                spec.latency_ms, LATENCY_BUDGET_MS
            ),
            "Apply quantization, reduce model size, or enable batched inference",
        );
    }

    // Gate 2: Accuracy > threshold
    if spec.accuracy > ACCURACY_THRESHOLD {
        stage.add(
            Severity::Info,
            "QA-002",
            &format!(
                "Gate 2 PASS: Accuracy {:.1}% > {:.1}% threshold",
                spec.accuracy * 100.0,
                ACCURACY_THRESHOLD * 100.0
            ),
        );
        passed += 1;
    } else {
        stage.add_with_fix(
            Severity::Critical,
            "QA-002",
            &format!(
                "Gate 2 FAIL: Accuracy {:.1}% below {:.1}% threshold",
                spec.accuracy * 100.0,
                ACCURACY_THRESHOLD * 100.0
            ),
            "Retrain model or use a larger base model",
        );
    }

    // Gate 3: Memory footprint < limit
    if spec.memory_mb < MEMORY_LIMIT_MB {
        stage.add(
            Severity::Info,
            "QA-003",
            &format!(
                "Gate 3 PASS: Memory {:.0}MB < {:.0}MB limit",
                spec.memory_mb, MEMORY_LIMIT_MB
            ),
        );
        passed += 1;
    } else {
        stage.add_with_fix(
            Severity::Critical,
            "QA-003",
            &format!(
                "Gate 3 FAIL: Memory {:.0}MB exceeds {:.0}MB limit",
                spec.memory_mb, MEMORY_LIMIT_MB
            ),
            "Apply model sharding, quantization, or reduce context length",
        );
    }

    // Gate 4: Throughput > minimum
    if spec.throughput_tps > MIN_THROUGHPUT_TPS {
        stage.add(
            Severity::Info,
            "QA-004",
            &format!(
                "Gate 4 PASS: Throughput {:.1} tok/s > {:.1} tok/s minimum",
                spec.throughput_tps, MIN_THROUGHPUT_TPS
            ),
        );
        passed += 1;
    } else {
        stage.add_with_fix(
            Severity::Critical,
            "QA-004",
            &format!(
                "Gate 4 FAIL: Throughput {:.1} tok/s below {:.1} tok/s minimum",
                spec.throughput_tps, MIN_THROUGHPUT_TPS
            ),
            "Enable dynamic batching, use SIMD backend, or upgrade hardware",
        );
    }

    (stage, passed)
}

// ============================================================================
// Stage 5: Report
// ============================================================================

/// Determine the overall compliance verdict from stage results.
fn determine_verdict(qual_passed: u8, qa_passed: u8, stages: &[AuditStage]) -> ComplianceVerdict {
    let total_critical: usize = stages.iter().map(AuditStage::critical_count).sum();

    if total_critical == 0 && qual_passed == 8 && qa_passed == 4 {
        return ComplianceVerdict::Compliant;
    }

    // Conditional: minor issues only (no critical in qualify, some QA misses)
    let qualify_critical: usize = stages
        .iter()
        .filter(|s| s.name == "Qualify")
        .map(AuditStage::critical_count)
        .sum();

    if qualify_critical == 0 && qa_passed >= 3 {
        return ComplianceVerdict::Conditional;
    }

    ComplianceVerdict::NonCompliant
}

/// Run the full audit pipeline on a model spec and produce a report.
pub fn run_audit(spec: &ModelSpec) -> AuditReport {
    let inspect = stage_inspect(spec);
    let oracle = stage_oracle(spec);
    let (qualify, qual_passed) = stage_qualify(spec);
    let (qa, qa_passed) = stage_qa(spec);

    let stages = vec![inspect, oracle, qualify, qa];
    let verdict = determine_verdict(qual_passed, qa_passed, &stages);

    AuditReport {
        model_name: spec.name.clone(),
        stages,
        verdict,
        qualification_score: (qual_passed, 8),
        qa_score: (qa_passed, 4),
    }
}

/// Print a formatted compliance certificate to stdout.
fn print_report(report: &AuditReport) {
    println!();
    println!("================================================================");
    println!("            COMPLIANCE AUDIT CERTIFICATE");
    println!("================================================================");
    println!();
    println!("Model: {}", report.model_name);
    println!(
        "Qualification: {}/{} gates passed",
        report.qualification_score.0, report.qualification_score.1
    );
    println!(
        "QA:            {}/{} gates passed",
        report.qa_score.0, report.qa_score.1
    );
    println!("Verdict:       {}", report.verdict);
    println!();

    for stage in &report.stages {
        println!("--- Stage: {} ---", stage.name);
        for finding in &stage.findings {
            println!(
                "  [{}] {}: {}",
                finding.severity, finding.code, finding.message
            );
            if let Some(ref fix) = finding.remediation {
                println!("         Remediation: {fix}");
            }
        }
        println!();
    }

    if report.verdict == ComplianceVerdict::NonCompliant {
        println!("NON-COMPLIANT ITEMS:");
        for stage in &report.stages {
            for finding in &stage.findings {
                if finding.severity == Severity::Critical {
                    println!("  - [{}] {}", finding.code, finding.message);
                }
            }
        }
        println!();
    }

    println!("================================================================");
}

// ============================================================================
// Model Spec Generators
// ============================================================================

/// Generate a fully compliant model spec using deterministic RNG.
fn make_compliant_spec(ctx: &mut RecipeContext) -> ModelSpec {
    let rng = ctx.rng();
    let latency: f64 = rng.gen_range(20.0..80.0);
    let accuracy: f64 = rng.gen_range(0.90..0.98);
    let memory: f64 = rng.gen_range(500.0..3000.0);
    let throughput: f64 = rng.gen_range(100.0..500.0);
    let total_params: u64 = rng.gen_range(1_000_000..500_000_000);
    let tensor_count: usize = rng.gen_range(50..200);

    let mut dtype_distribution = HashMap::new();
    dtype_distribution.insert("FP16".to_string(), 0.85);
    dtype_distribution.insert("FP32".to_string(), 0.15);

    let mut metadata = HashMap::new();
    metadata.insert("model_type".to_string(), "transformer".to_string());
    metadata.insert("version".to_string(), "1.0.0".to_string());
    metadata.insert("author".to_string(), "compliance-team".to_string());

    let tensor_names = vec![
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        "model.layers.0.self_attn.v_proj.weight".to_string(),
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        "model.embed_tokens.weight".to_string(),
    ];
    let tensor_shapes = vec![
        vec![4096, 4096],
        vec![4096, 4096],
        vec![4096, 4096],
        vec![11008, 4096],
        vec![32000, 4096],
    ];

    ModelSpec {
        name: "llama-7b-compliant-v1".to_string(),
        magic: *b"APR2",
        format_version: "2.0".to_string(),
        tensor_count,
        total_params,
        dtype_distribution,
        compression: "zstd".to_string(),
        tensor_names,
        tensor_shapes,
        metadata,
        has_nan: false,
        has_inf: false,
        size_bytes: 7 * 1024 * 1024 * 1024,
        checksum_valid: true,
        latency_ms: latency,
        accuracy,
        memory_mb: memory,
        throughput_tps: throughput,
    }
}

/// Generate a non-compliant model spec with several issues.
fn make_noncompliant_spec(ctx: &mut RecipeContext) -> ModelSpec {
    let rng = ctx.rng();
    let latency: f64 = rng.gen_range(20.0..80.0);
    let accuracy: f64 = rng.gen_range(0.60..0.80); // Below threshold
    let memory: f64 = rng.gen_range(500.0..3000.0);
    let throughput: f64 = rng.gen_range(100.0..500.0);
    let total_params: u64 = rng.gen_range(1_000_000..500_000_000);
    let tensor_count: usize = rng.gen_range(50..200);

    let mut dtype_distribution = HashMap::new();
    dtype_distribution.insert("FP16".to_string(), 0.70);
    dtype_distribution.insert("BF16".to_string(), 0.30); // Unsupported dtype

    // Missing 'version' metadata key
    let mut metadata = HashMap::new();
    metadata.insert("model_type".to_string(), "transformer".to_string());

    let tensor_names = vec![
        "encoder.layer.0.attention.self.query.weight".to_string(),
        "encoder.layer.0.attention.self.key.weight".to_string(),
        "encoder.layer.0.attention.self.value.weight".to_string(),
        "encoder.layer.0.intermediate.dense.weight".to_string(),
        "embeddings.word_embeddings.weight".to_string(),
    ];
    let tensor_shapes = vec![
        vec![768, 768],
        vec![768, 768],
        vec![768, 768],
        vec![3072, 768],
        vec![30522, 768],
    ];

    ModelSpec {
        name: "bert-base-issues-v2".to_string(),
        magic: *b"APR2",
        format_version: "2.0".to_string(),
        tensor_count,
        total_params,
        dtype_distribution,
        compression: "none".to_string(),
        tensor_names,
        tensor_shapes,
        metadata,
        has_nan: true, // NaN in weights
        has_inf: false,
        size_bytes: 440 * 1024 * 1024,
        checksum_valid: true,
        latency_ms: latency,
        accuracy,
        memory_mb: memory,
        throughput_tps: throughput,
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() -> Result<()> {
    println!("================================================================");
    println!("  Compliance Audit Pipeline: inspect -> oracle -> qualify -> qa -> report");
    println!("  Toyota Way: Jidoka (built-in quality gates)");
    println!("================================================================");

    let mut ctx = RecipeContext::new("compliance_audit")?;

    // --- Audit Model 1: Fully compliant ---
    println!("\n[Audit 1/2] Fully compliant model");
    let compliant_spec = make_compliant_spec(&mut ctx);
    let report_1 = run_audit(&compliant_spec);
    print_report(&report_1);

    // --- Audit Model 2: Model with issues ---
    println!("\n[Audit 2/2] Model with compliance issues");
    let noncompliant_spec = make_noncompliant_spec(&mut ctx);
    let report_2 = run_audit(&noncompliant_spec);
    print_report(&report_2);

    // Record metrics
    ctx.record_metric("models_audited", 2);
    ctx.record_metric(
        "compliant_count",
        i64::from(report_1.verdict == ComplianceVerdict::Compliant)
            + i64::from(report_2.verdict == ComplianceVerdict::Compliant),
    );
    ctx.record_string_metric("verdict_model_1", format!("{}", report_1.verdict));
    ctx.record_string_metric("verdict_model_2", format!("{}", report_2.verdict));

    println!("\nCompliance audit pipeline complete.");
    ctx.report()?;
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal compliant spec for testing.
    fn minimal_compliant_spec() -> ModelSpec {
        let mut dtype_distribution = HashMap::new();
        dtype_distribution.insert("FP32".to_string(), 1.0);

        let mut metadata = HashMap::new();
        metadata.insert("model_type".to_string(), "linear".to_string());
        metadata.insert("version".to_string(), "1.0".to_string());

        ModelSpec {
            name: "test-compliant".to_string(),
            magic: *b"APR2",
            format_version: "2.0".to_string(),
            tensor_count: 10,
            total_params: 1_000_000,
            dtype_distribution,
            compression: "none".to_string(),
            tensor_names: vec!["model.layers.0.self_attn.q_proj.weight".to_string()],
            tensor_shapes: vec![vec![768, 768]],
            metadata,
            has_nan: false,
            has_inf: false,
            size_bytes: 4 * 1024 * 1024,
            checksum_valid: true,
            latency_ms: 25.0,
            accuracy: 0.92,
            memory_mb: 512.0,
            throughput_tps: 200.0,
        }
    }

    #[test]
    fn test_compliant_model_passes_all_gates() {
        let spec = minimal_compliant_spec();
        let report = run_audit(&spec);
        assert_eq!(report.verdict, ComplianceVerdict::Compliant);
        assert_eq!(report.qualification_score, (8, 8));
        assert_eq!(report.qa_score, (4, 4));
    }

    #[test]
    fn test_nan_fails_qualification_gate() {
        let mut spec = minimal_compliant_spec();
        spec.has_nan = true;
        let report = run_audit(&spec);
        assert!(report.qualification_score.0 < 8);
        assert_ne!(report.verdict, ComplianceVerdict::Compliant);
    }

    #[test]
    fn test_inf_fails_qualification_gate() {
        let mut spec = minimal_compliant_spec();
        spec.has_inf = true;
        let report = run_audit(&spec);
        assert!(report.qualification_score.0 < 8);
        assert_ne!(report.verdict, ComplianceVerdict::Compliant);
    }

    #[test]
    fn test_invalid_magic_fails_qualification() {
        let mut spec = minimal_compliant_spec();
        spec.magic = *b"XXXX";
        let report = run_audit(&spec);
        assert!(report.qualification_score.0 < 8);
        assert_ne!(report.verdict, ComplianceVerdict::Compliant);
    }

    #[test]
    fn test_missing_metadata_fails_qualification() {
        let mut spec = minimal_compliant_spec();
        spec.metadata.remove("version");
        let report = run_audit(&spec);
        // Should fail gate 7 (metadata) and also get flagged in inspect
        assert!(report.qualification_score.0 < 8);
    }

    #[test]
    fn test_accuracy_below_threshold_fails_qa() {
        let mut spec = minimal_compliant_spec();
        spec.accuracy = 0.50;
        let report = run_audit(&spec);
        assert!(report.qa_score.0 < 4);
    }

    #[test]
    fn test_architecture_detection_llama() {
        let names = vec![
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.layers.0.self_attn.k_proj.weight".to_string(),
        ];
        assert_eq!(detect_architecture(&names), "LLaMA");
    }

    #[test]
    fn test_architecture_detection_gpt() {
        let names = vec![
            "transformer.h.0.attn.c_attn.weight".to_string(),
            "transformer.h.0.attn.c_proj.weight".to_string(),
        ];
        assert_eq!(detect_architecture(&names), "GPT");
    }

    #[test]
    fn test_architecture_detection_bert() {
        let names = vec![
            "encoder.layer.0.attention.self.query.weight".to_string(),
            "encoder.layer.0.attention.self.key.weight".to_string(),
        ];
        assert_eq!(detect_architecture(&names), "BERT");
    }

    #[test]
    fn test_unknown_architecture_fails_oracle() {
        let mut spec = minimal_compliant_spec();
        spec.tensor_names = vec!["custom.layer.weight".to_string()];
        let report = run_audit(&spec);
        // Oracle should flag unknown architecture as critical
        let oracle_stage = report.stages.iter().find(|s| s.name == "Oracle");
        assert!(oracle_stage.is_some());
        let critical_findings = oracle_stage.map_or(0, |s| s.critical_count());
        assert!(
            critical_findings > 0,
            "Unknown arch should produce critical finding"
        );
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", Severity::Info), "INFO");
        assert_eq!(format!("{}", Severity::Warning), "WARN");
        assert_eq!(format!("{}", Severity::Critical), "CRIT");
    }

    #[test]
    fn test_verdict_display() {
        assert_eq!(format!("{}", ComplianceVerdict::Compliant), "COMPLIANT");
        assert_eq!(format!("{}", ComplianceVerdict::Conditional), "CONDITIONAL");
        assert_eq!(
            format!("{}", ComplianceVerdict::NonCompliant),
            "NON-COMPLIANT"
        );
    }

    #[test]
    fn test_conditional_verdict_on_qa_only_failure() {
        let mut spec = minimal_compliant_spec();
        // Fail only one QA gate (throughput) but pass all 8 qualification gates
        spec.throughput_tps = 10.0;
        let report = run_audit(&spec);
        assert_eq!(report.qualification_score, (8, 8));
        assert_eq!(report.qa_score.0, 3);
        assert_eq!(report.verdict, ComplianceVerdict::Conditional);
    }

    #[test]
    fn test_deterministic_compliant_spec() {
        let mut ctx1 = RecipeContext::new("compliance_audit_det").expect("context");
        let mut ctx2 = RecipeContext::new("compliance_audit_det").expect("context");
        let spec1 = make_compliant_spec(&mut ctx1);
        let spec2 = make_compliant_spec(&mut ctx2);
        assert_eq!(spec1.name, spec2.name);
        assert_eq!(spec1.total_params, spec2.total_params);
        assert!((spec1.latency_ms - spec2.latency_ms).abs() < f64::EPSILON);
    }

    #[test]
    fn test_report_has_all_stages() {
        let spec = minimal_compliant_spec();
        let report = run_audit(&spec);
        assert_eq!(report.stages.len(), 4);
        assert_eq!(report.stages[0].name, "Inspect");
        assert_eq!(report.stages[1].name, "Oracle");
        assert_eq!(report.stages[2].name, "Qualify");
        assert_eq!(report.stages[3].name, "QA");
    }

    #[test]
    fn test_oversize_model_fails_qualification() {
        let mut spec = minimal_compliant_spec();
        spec.size_bytes = 15 * 1024 * 1024 * 1024; // 15 GB
        let report = run_audit(&spec);
        assert!(report.qualification_score.0 < 8);
    }

    #[test]
    fn test_unsupported_dtype_fails_qualification() {
        let mut spec = minimal_compliant_spec();
        spec.dtype_distribution.insert("BF16".to_string(), 0.5);
        let report = run_audit(&spec);
        assert!(report.qualification_score.0 < 8);
    }

    #[test]
    fn test_invalid_checksum_fails_qualification() {
        let mut spec = minimal_compliant_spec();
        spec.checksum_valid = false;
        let report = run_audit(&spec);
        assert!(report.qualification_score.0 < 8);
    }

    #[test]
    fn test_latency_over_budget_fails_qa() {
        let mut spec = minimal_compliant_spec();
        spec.latency_ms = 200.0;
        let report = run_audit(&spec);
        assert!(report.qa_score.0 < 4);
    }

    #[test]
    fn test_memory_over_limit_fails_qa() {
        let mut spec = minimal_compliant_spec();
        spec.memory_mb = 8000.0;
        let report = run_audit(&spec);
        assert!(report.qa_score.0 < 4);
    }
}
