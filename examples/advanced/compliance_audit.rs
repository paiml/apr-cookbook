//! # Recipe: Compliance Audit Pipeline
//!
//! Demonstrates a full compliance audit pipeline for model deployment approval,
//! composing five stages: inspect, oracle, qualify, qa, and report.
//!
//! ```bash
//! cargo run --example compliance_audit
//! ```
//!
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::fmt;

// ---- Data Structures --------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Severity {
    Info,
    Warning,
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

#[derive(Debug, Clone)]
pub struct Finding {
    pub severity: Severity,
    pub code: String,
    pub message: String,
    pub remediation: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AuditStage {
    pub name: String,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplianceVerdict {
    Compliant,
    Conditional,
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

#[derive(Debug, Clone)]
pub struct AuditReport {
    pub model_name: String,
    pub stages: Vec<AuditStage>,
    pub verdict: ComplianceVerdict,
    pub qualification_score: (u8, u8),
    pub qa_score: (u8, u8),
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: String,
    pub magic: [u8; 4],
    pub format_version: String,
    pub tensor_count: usize,
    pub total_params: u64,
    pub dtype_distribution: HashMap<String, f32>,
    pub compression: String,
    pub tensor_names: Vec<String>,
    pub tensor_shapes: Vec<Vec<usize>>,
    pub metadata: HashMap<String, String>,
    pub has_nan: bool,
    pub has_inf: bool,
    pub size_bytes: u64,
    pub checksum_valid: bool,
    pub latency_ms: f64,
    pub accuracy: f64,
    pub memory_mb: f64,
    pub throughput_tps: f64,
}

// ---- Stage 1: Inspect -------------------------------------------------------

fn stage_inspect(spec: &ModelSpec) -> AuditStage {
    let mut s = AuditStage::new("Inspect");
    s.add(
        Severity::Info,
        "INS-001",
        &format!("Format version: {}", spec.format_version),
    );
    s.add(
        Severity::Info,
        "INS-002",
        &format!("Tensor count: {}", spec.tensor_count),
    );
    s.add(
        Severity::Info,
        "INS-003",
        &format!("Total parameters: {}", spec.total_params),
    );
    let dtypes: Vec<String> = spec
        .dtype_distribution
        .iter()
        .map(|(k, v)| format!("{}={:.0}%", k, v * 100.0))
        .collect();
    s.add(
        Severity::Info,
        "INS-004",
        &format!("Dtype distribution: {}", dtypes.join(", ")),
    );
    s.add(
        Severity::Info,
        "INS-005",
        &format!("Compression: {}", spec.compression),
    );
    if !spec.metadata.contains_key("model_type") {
        s.add_with_fix(
            Severity::Critical,
            "INS-010",
            "Missing required metadata field: model_type",
            "Add 'model_type' to model metadata before export",
        );
    }
    if !spec.metadata.contains_key("version") {
        s.add_with_fix(
            Severity::Critical,
            "INS-011",
            "Missing required metadata field: version",
            "Add 'version' to model metadata before export",
        );
    }
    s
}

// ---- Stage 2: Oracle --------------------------------------------------------

const APPROVED_ARCHITECTURES: &[&str] = &["LLaMA", "GPT", "BERT"];

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

fn stage_oracle(spec: &ModelSpec) -> AuditStage {
    let mut s = AuditStage::new("Oracle");
    let family = detect_architecture(&spec.tensor_names);
    s.add(
        Severity::Info,
        "ORA-001",
        &format!("Detected architecture family: {family}"),
    );
    if APPROVED_ARCHITECTURES.contains(&family) {
        s.add(
            Severity::Info,
            "ORA-002",
            &format!("{family} is on the approved architecture list"),
        );
    } else {
        s.add_with_fix(
            Severity::Critical,
            "ORA-010",
            &format!("{family} is NOT on the approved architecture list"),
            &format!(
                "Convert model to an approved architecture: {}",
                APPROVED_ARCHITECTURES.join(", ")
            ),
        );
    }
    s
}

// ---- Stage 3: Qualify (8 gates) ---------------------------------------------

const SUPPORTED_DTYPES: &[&str] = &["FP32", "FP16", "INT8"];
const MAX_SIZE_BYTES: u64 = 10 * 1024 * 1024 * 1024;

fn stage_qualify(spec: &ModelSpec) -> (AuditStage, u8) {
    let mut s = AuditStage::new("Qualify");
    let mut p: u8 = 0;
    macro_rules! gate {
        ($code:expr, $n:expr, $cond:expr, $pass:expr, $fail:expr, $fix:expr) => {
            if $cond {
                s.add(
                    Severity::Info,
                    $code,
                    &format!("Gate {} PASS: {}", $n, $pass),
                );
                p += 1;
            } else {
                s.add_with_fix(
                    Severity::Critical,
                    $code,
                    &format!("Gate {} FAIL: {}", $n, $fail),
                    $fix,
                );
            }
        };
    }
    gate!(
        "QAL-001",
        1,
        spec.magic == *b"APR2",
        "APR2 magic bytes valid",
        format!(
            "Invalid magic bytes (expected APR2, got {:?})",
            std::str::from_utf8(&spec.magic).unwrap_or("????")
        ),
        "Re-export model using aprender with APR2 format"
    );
    gate!(
        "QAL-002",
        2,
        !spec.format_version.is_empty() && spec.tensor_count > 0,
        "Header parseable",
        "Header not parseable",
        "Verify model export completed without errors"
    );
    gate!(
        "QAL-003",
        3,
        !spec.has_nan,
        "No NaN values in weights",
        "NaN values detected",
        "Re-train with gradient clipping or check for numerical instability"
    );
    gate!(
        "QAL-004",
        4,
        !spec.has_inf,
        "No Inf values in weights",
        "Inf values detected",
        "Re-train with loss scaling or reduce learning rate"
    );
    let sz_gb = spec.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    gate!(
        "QAL-005",
        5,
        spec.size_bytes < MAX_SIZE_BYTES,
        format!("Size {sz_gb:.2} GB within 10 GB budget"),
        format!("Size {sz_gb:.2} GB exceeds 10 GB budget"),
        "Apply quantization (INT8/INT4) or pruning to reduce model size"
    );
    let all_supported = spec
        .dtype_distribution
        .keys()
        .all(|dt| SUPPORTED_DTYPES.contains(&dt.as_str()));
    gate!(
        "QAL-006",
        6,
        all_supported,
        "All dtypes supported",
        "Unsupported dtypes found",
        &format!(
            "Convert to a supported dtype: {}",
            SUPPORTED_DTYPES.join(", ")
        )
    );
    let has_mt = spec.metadata.contains_key("model_type");
    let has_v = spec.metadata.contains_key("version");
    gate!(
        "QAL-007",
        7,
        has_mt && has_v,
        "Required metadata fields present",
        "Missing metadata fields",
        "Add missing metadata fields before model export"
    );
    gate!(
        "QAL-008",
        8,
        spec.checksum_valid,
        "Checksum valid",
        "Checksum mismatch",
        "Re-export model and verify file integrity after transfer"
    );
    (s, p)
}

// ---- Stage 4: QA (4 gates) --------------------------------------------------

const LATENCY_BUDGET_MS: f64 = 100.0;
const ACCURACY_THRESHOLD: f64 = 0.85;
const MEMORY_LIMIT_MB: f64 = 4096.0;
const MIN_THROUGHPUT_TPS: f64 = 50.0;

fn stage_qa(spec: &ModelSpec) -> (AuditStage, u8) {
    let mut s = AuditStage::new("QA");
    let mut p: u8 = 0;
    macro_rules! qa_gate {
        ($code:expr, $n:expr, $cond:expr, $pass:expr, $fail:expr, $fix:expr) => {
            if $cond {
                s.add(
                    Severity::Info,
                    $code,
                    &format!("Gate {} PASS: {}", $n, $pass),
                );
                p += 1;
            } else {
                s.add_with_fix(
                    Severity::Critical,
                    $code,
                    &format!("Gate {} FAIL: {}", $n, $fail),
                    $fix,
                );
            }
        };
    }
    qa_gate!(
        "QA-001",
        1,
        spec.latency_ms < LATENCY_BUDGET_MS,
        format!(
            "Latency {:.1}ms < {LATENCY_BUDGET_MS:.1}ms",
            spec.latency_ms
        ),
        format!(
            "Latency {:.1}ms exceeds {LATENCY_BUDGET_MS:.1}ms",
            spec.latency_ms
        ),
        "Apply quantization, reduce model size, or enable batched inference"
    );
    qa_gate!(
        "QA-002",
        2,
        spec.accuracy > ACCURACY_THRESHOLD,
        format!(
            "Accuracy {:.1}% > {:.1}%",
            spec.accuracy * 100.0,
            ACCURACY_THRESHOLD * 100.0
        ),
        format!(
            "Accuracy {:.1}% below {:.1}%",
            spec.accuracy * 100.0,
            ACCURACY_THRESHOLD * 100.0
        ),
        "Retrain model or use a larger base model"
    );
    qa_gate!(
        "QA-003",
        3,
        spec.memory_mb < MEMORY_LIMIT_MB,
        format!("Memory {:.0}MB < {MEMORY_LIMIT_MB:.0}MB", spec.memory_mb),
        format!(
            "Memory {:.0}MB exceeds {MEMORY_LIMIT_MB:.0}MB",
            spec.memory_mb
        ),
        "Apply model sharding, quantization, or reduce context length"
    );
    qa_gate!(
        "QA-004",
        4,
        spec.throughput_tps > MIN_THROUGHPUT_TPS,
        format!(
            "Throughput {:.1} tok/s > {MIN_THROUGHPUT_TPS:.1}",
            spec.throughput_tps
        ),
        format!(
            "Throughput {:.1} tok/s below {MIN_THROUGHPUT_TPS:.1}",
            spec.throughput_tps
        ),
        "Enable dynamic batching, use SIMD backend, or upgrade hardware"
    );
    (s, p)
}

// ---- Stage 5: Report --------------------------------------------------------

fn determine_verdict(qual_passed: u8, qa_passed: u8, stages: &[AuditStage]) -> ComplianceVerdict {
    let total_critical: usize = stages.iter().map(AuditStage::critical_count).sum();
    if total_critical == 0 && qual_passed == 8 && qa_passed == 4 {
        return ComplianceVerdict::Compliant;
    }
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

fn print_report(report: &AuditReport) {
    println!("\n================================================================");
    println!("            COMPLIANCE AUDIT CERTIFICATE");
    println!("================================================================\n");
    println!(
        "Model: {}  Qual: {}/{}  QA: {}/{}  Verdict: {}",
        report.model_name,
        report.qualification_score.0,
        report.qualification_score.1,
        report.qa_score.0,
        report.qa_score.1,
        report.verdict
    );
    for stage in &report.stages {
        println!("--- Stage: {} ---", stage.name);
        for f in &stage.findings {
            println!("  [{}] {}: {}", f.severity, f.code, f.message);
            if let Some(ref fix) = f.remediation {
                println!("         Remediation: {fix}");
            }
        }
    }
    println!("================================================================");
}

// ---- Model Spec Generators --------------------------------------------------

fn make_compliant_spec(ctx: &mut RecipeContext) -> ModelSpec {
    let rng = ctx.rng();
    let mut dtype_distribution = HashMap::new();
    dtype_distribution.insert("FP16".to_string(), 0.85);
    dtype_distribution.insert("FP32".to_string(), 0.15);
    let mut metadata = HashMap::new();
    metadata.insert("model_type".to_string(), "transformer".to_string());
    metadata.insert("version".to_string(), "1.0.0".to_string());
    metadata.insert("author".to_string(), "compliance-team".to_string());
    ModelSpec {
        name: "llama-7b-compliant-v1".to_string(),
        magic: *b"APR2",
        format_version: "2.0".to_string(),
        tensor_count: rng.gen_range(50..200),
        total_params: rng.gen_range(1_000_000..500_000_000),
        dtype_distribution,
        compression: "zstd".to_string(),
        tensor_names: vec![
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.layers.0.self_attn.k_proj.weight".to_string(),
        ],
        tensor_shapes: vec![vec![4096, 4096], vec![4096, 4096]],
        metadata,
        has_nan: false,
        has_inf: false,
        size_bytes: 7 * 1024 * 1024 * 1024,
        checksum_valid: true,
        latency_ms: rng.gen_range(20.0..80.0),
        accuracy: rng.gen_range(0.90..0.98),
        memory_mb: rng.gen_range(500.0..3000.0),
        throughput_tps: rng.gen_range(100.0..500.0),
    }
}

fn make_noncompliant_spec(ctx: &mut RecipeContext) -> ModelSpec {
    let rng = ctx.rng();
    let mut dtype_distribution = HashMap::new();
    dtype_distribution.insert("FP16".to_string(), 0.70);
    dtype_distribution.insert("BF16".to_string(), 0.30);
    let mut metadata = HashMap::new();
    metadata.insert("model_type".to_string(), "transformer".to_string());
    ModelSpec {
        name: "bert-base-issues-v2".to_string(),
        magic: *b"APR2",
        format_version: "2.0".to_string(),
        tensor_count: rng.gen_range(50..200),
        total_params: rng.gen_range(1_000_000..500_000_000),
        dtype_distribution,
        compression: "none".to_string(),
        tensor_names: vec![
            "encoder.layer.0.attention.self.query.weight".to_string(),
            "encoder.layer.0.attention.self.key.weight".to_string(),
        ],
        tensor_shapes: vec![vec![768, 768], vec![768, 768]],
        metadata,
        has_nan: true,
        has_inf: false,
        size_bytes: 440 * 1024 * 1024,
        checksum_valid: true,
        latency_ms: rng.gen_range(20.0..80.0),
        accuracy: rng.gen_range(0.60..0.80),
        memory_mb: rng.gen_range(500.0..3000.0),
        throughput_tps: rng.gen_range(100.0..500.0),
    }
}

// ---- Main -------------------------------------------------------------------

fn main() -> Result<()> {
    println!("================================================================");
    println!("  Compliance Audit Pipeline: inspect -> oracle -> qualify -> qa -> report");
    println!("================================================================");
    let mut ctx = RecipeContext::new("compliance_audit")?;

    println!("\n[Audit 1/2] Fully compliant model");
    let report_1 = run_audit(&make_compliant_spec(&mut ctx));
    print_report(&report_1);

    println!("\n[Audit 2/2] Model with compliance issues");
    let report_2 = run_audit(&make_noncompliant_spec(&mut ctx));
    print_report(&report_2);

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

// ---- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
        let report = run_audit(&minimal_compliant_spec());
        assert_eq!(report.verdict, ComplianceVerdict::Compliant);
        assert_eq!(report.qualification_score, (8, 8));
        assert_eq!(report.qa_score, (4, 4));
    }

    #[test]
    fn test_nan_and_inf_fail_qualification() {
        let mut spec = minimal_compliant_spec();
        spec.has_nan = true;
        assert!(run_audit(&spec).qualification_score.0 < 8);
        spec.has_nan = false;
        spec.has_inf = true;
        assert!(run_audit(&spec).qualification_score.0 < 8);
    }

    #[test]
    fn test_invalid_magic_fails_qualification() {
        let mut spec = minimal_compliant_spec();
        spec.magic = *b"XXXX";
        assert_ne!(run_audit(&spec).verdict, ComplianceVerdict::Compliant);
    }

    #[test]
    fn test_missing_metadata_fails_qualification() {
        let mut spec = minimal_compliant_spec();
        spec.metadata.remove("version");
        assert!(run_audit(&spec).qualification_score.0 < 8);
    }

    #[test]
    fn test_architecture_detection() {
        assert_eq!(
            detect_architecture(&["model.layers.0.self_attn.q_proj.weight".into()]),
            "LLaMA"
        );
        assert_eq!(
            detect_architecture(&["transformer.h.0.attn.c_attn.weight".into()]),
            "GPT"
        );
        assert_eq!(
            detect_architecture(&["encoder.layer.0.attention.self.query.weight".into()]),
            "BERT"
        );
        assert_eq!(
            detect_architecture(&["custom.layer.weight".into()]),
            "Unknown"
        );
    }

    #[test]
    fn test_unknown_architecture_fails_oracle() {
        let mut spec = minimal_compliant_spec();
        spec.tensor_names = vec!["custom.layer.weight".to_string()];
        let report = run_audit(&spec);
        let oracle_critical = report
            .stages
            .iter()
            .find(|s| s.name == "Oracle")
            .map_or(0, |s| s.critical_count());
        assert!(oracle_critical > 0);
    }

    #[test]
    fn test_conditional_verdict_on_qa_only_failure() {
        let mut spec = minimal_compliant_spec();
        spec.throughput_tps = 10.0;
        let report = run_audit(&spec);
        assert_eq!(report.qualification_score, (8, 8));
        assert_eq!(report.qa_score.0, 3);
        assert_eq!(report.verdict, ComplianceVerdict::Conditional);
    }

    #[test]
    fn test_qa_failures() {
        let mut spec = minimal_compliant_spec();
        spec.accuracy = 0.50;
        assert!(run_audit(&spec).qa_score.0 < 4);
        spec.accuracy = 0.92;
        spec.latency_ms = 200.0;
        assert!(run_audit(&spec).qa_score.0 < 4);
        spec.latency_ms = 25.0;
        spec.memory_mb = 8000.0;
        assert!(run_audit(&spec).qa_score.0 < 4);
    }

    #[test]
    fn test_oversize_and_unsupported_dtype_fail() {
        let mut spec = minimal_compliant_spec();
        spec.size_bytes = 15 * 1024 * 1024 * 1024;
        assert!(run_audit(&spec).qualification_score.0 < 8);
        spec.size_bytes = 4 * 1024 * 1024;
        spec.dtype_distribution.insert("BF16".to_string(), 0.5);
        assert!(run_audit(&spec).qualification_score.0 < 8);
    }

    #[test]
    fn test_report_has_all_stages() {
        let report = run_audit(&minimal_compliant_spec());
        assert_eq!(report.stages.len(), 4);
        let names: Vec<&str> = report.stages.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["Inspect", "Oracle", "Qualify", "QA"]);
    }
}
