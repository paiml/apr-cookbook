#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use super::types::*;

#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::fmt;

pub fn stage_inspect(spec: &ModelSpec) -> AuditStage {
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

pub fn detect_architecture(tensor_names: &[String]) -> &'static str {
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

pub fn stage_oracle(spec: &ModelSpec) -> AuditStage {
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

pub fn stage_qualify(spec: &ModelSpec) -> (AuditStage, u8) {
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

pub fn stage_qa(spec: &ModelSpec) -> (AuditStage, u8) {
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

pub fn determine_verdict(
    qual_passed: u8,
    qa_passed: u8,
    stages: &[AuditStage],
) -> ComplianceVerdict {
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

pub fn print_report(report: &AuditReport) {
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

pub fn make_compliant_spec(ctx: &mut RecipeContext) -> ModelSpec {
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

pub fn make_noncompliant_spec(ctx: &mut RecipeContext) -> ModelSpec {
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
