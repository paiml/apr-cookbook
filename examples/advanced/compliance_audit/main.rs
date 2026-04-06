#![allow(unused_imports)]
//! # Recipe: Compliance Audit Pipeline
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Demonstrates a full compliance audit pipeline for model deployment approval,
//! composing five stages: inspect, oracle, qualify, qa, and report.
//!
//! ```bash
//! cargo run --example compliance_audit
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::fmt;

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
