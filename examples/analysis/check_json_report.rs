//! # Recipe: Check with JSON Machine-Readable Report
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr check model.apr --format json --out report.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example check_json_report` exits 0
//! 2. [x] `cargo test --example check_json_report` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr check` JSON output in-process (no shell-out)
//! 10. [x] Unit tests cover schema round-trip, severity ordering, exit-code mapping
//!
//! ## Learning Objective
//! Produces a structured JSON health report from a single-model check, including
//! severity-ordered findings and a deterministic schema. This is the machine-
//! readable form CI pipelines consume to gate deploys.
//!
//! ## Run Command
//! ```bash
//! cargo run --example check_json_report
//! ```
//!
//! ## References
//! - Amershi, S. et al. (2019). *Software Engineering for Machine Learning: A Case Study*. ICSE-SEIP. DOI: 10.1109/ICSE-SEIP.2019.00042

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Severity {
    Info,
    Warn,
    Error,
    Critical,
}

impl Severity {
    #[allow(dead_code)]
    fn label(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
            Self::Critical => "critical",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Finding {
    code: String,
    severity: Severity,
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckReport {
    schema_version: u32,
    model_name: String,
    file_size_bytes: usize,
    findings: Vec<Finding>,
    exit_code: i32,
}

// ---------------------------------------------------------------------------
// Logic
// ---------------------------------------------------------------------------

fn check_model_bundle(name: &str, data: &[u8]) -> Vec<Finding> {
    let mut out = Vec::new();
    if data.len() < 4 || &data[0..4] != b"APR2" {
        out.push(Finding {
            code: "CHK-001".into(),
            severity: Severity::Critical,
            message: "missing or invalid APR2 magic bytes".into(),
        });
    }
    if data.len() < 64 {
        out.push(Finding {
            code: "CHK-002".into(),
            severity: Severity::Error,
            message: format!("model too small: {} bytes", data.len()),
        });
    }
    if data.len() > 10_000_000 {
        out.push(Finding {
            code: "CHK-003".into(),
            severity: Severity::Warn,
            message: format!("model unusually large: {} bytes", data.len()),
        });
    }
    if name.len() > 63 {
        out.push(Finding {
            code: "CHK-004".into(),
            severity: Severity::Warn,
            message: "model name exceeds recommended 63-char limit".into(),
        });
    }
    // Info-level finding: always report tensor-count presence.
    out.push(Finding {
        code: "CHK-I01".into(),
        severity: Severity::Info,
        message: "scan complete".into(),
    });
    out
}

/// Map the worst severity -> exit code.
fn exit_code_for(findings: &[Finding]) -> i32 {
    let worst = findings
        .iter()
        .map(|f| f.severity)
        .max()
        .unwrap_or(Severity::Info);
    match worst {
        Severity::Info => 0,
        Severity::Warn => 0, // warnings do not fail CI
        Severity::Error => 1,
        Severity::Critical => 2,
    }
}

fn build_report(name: &str, data: &[u8]) -> CheckReport {
    let mut findings = check_model_bundle(name, data);
    findings.sort_by(|a, b| b.severity.cmp(&a.severity).then(a.code.cmp(&b.code)));
    let exit_code = exit_code_for(&findings);
    CheckReport {
        schema_version: 1,
        model_name: name.to_string(),
        file_size_bytes: data.len(),
        findings,
        exit_code,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("check_json_report")?;
    println!("=== Recipe: {} ===", ctx.name());

    // Build a valid model and a deliberately broken one.
    let dim = 32;
    let seed = hash_name_to_seed("check-json-healthy");
    let healthy = ModelBundleV2::new()
        .with_name("healthy-m")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor(
            "weight",
            vec![dim, dim],
            generate_model_payload(seed, dim * dim),
        )
        .build();

    let broken: Vec<u8> = b"XXXX".to_vec();

    let healthy_path = ctx.path("healthy.apr");
    let broken_path = ctx.path("broken.apr");
    std::fs::write(&healthy_path, &healthy)?;
    std::fs::write(&broken_path, &broken)?;

    let healthy_report = build_report("healthy-m", &healthy);
    let broken_report = build_report("broken-m", &broken);

    let healthy_json = serde_json::to_string_pretty(&healthy_report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    let broken_json = serde_json::to_string_pretty(&broken_report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;

    let healthy_report_path = ctx.path("healthy-report.json");
    let broken_report_path = ctx.path("broken-report.json");
    std::fs::write(&healthy_report_path, &healthy_json)?;
    std::fs::write(&broken_report_path, &broken_json)?;

    println!("\n--- Healthy Report ---");
    println!("{}", healthy_json);
    println!("\n--- Broken Report ---");
    println!("{}", broken_json);

    // Sanity.
    assert_eq!(healthy_report.exit_code, 0);
    assert_eq!(broken_report.exit_code, 2);
    assert!(broken_report.findings.iter().any(|f| f.code == "CHK-001"));

    // Round-trip: the serialized JSON must deserialize to an equivalent report.
    let parsed: CheckReport = serde_json::from_str(&healthy_json)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    assert_eq!(parsed.schema_version, healthy_report.schema_version);
    assert_eq!(parsed.model_name, healthy_report.model_name);

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warn);
        assert!(Severity::Warn < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }

    #[test]
    fn test_bad_magic_produces_critical() {
        let findings = check_model_bundle("m", b"XXXX");
        assert!(findings
            .iter()
            .any(|f| f.code == "CHK-001" && f.severity == Severity::Critical));
    }

    #[test]
    fn test_valid_bundle_has_no_errors() {
        let mut data = b"APR2".to_vec();
        data.extend(vec![0_u8; 200]);
        let r = build_report("ok", &data);
        // No error-or-worse findings.
        assert!(r.findings.iter().all(|f| f.severity < Severity::Error));
        assert_eq!(r.exit_code, 0);
    }

    #[test]
    fn test_findings_sorted_by_severity_descending() {
        let r = build_report("m", b"XXXX");
        let sevs: Vec<Severity> = r.findings.iter().map(|f| f.severity).collect();
        // First finding must be the worst.
        let max = *sevs.iter().max().unwrap_or(&Severity::Info);
        assert_eq!(sevs.first().copied().unwrap_or(Severity::Info), max);
    }

    #[test]
    fn test_exit_code_for_critical_is_2() {
        let f = vec![Finding {
            code: "x".into(),
            severity: Severity::Critical,
            message: "".into(),
        }];
        assert_eq!(exit_code_for(&f), 2);
    }

    #[test]
    fn test_exit_code_for_warn_only_is_0() {
        let f = vec![Finding {
            code: "x".into(),
            severity: Severity::Warn,
            message: "".into(),
        }];
        assert_eq!(exit_code_for(&f), 0);
    }

    #[test]
    fn test_report_roundtrips() {
        let r = build_report("roundtrip", b"XXXX");
        let json = serde_json::to_string(&r).expect("serialize");
        let back: CheckReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.model_name, r.model_name);
        assert_eq!(back.exit_code, r.exit_code);
        assert_eq!(back.findings.len(), r.findings.len());
    }
}
