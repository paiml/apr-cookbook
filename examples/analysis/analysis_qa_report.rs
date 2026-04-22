//! # Recipe: QA Aggregate Report Across Models
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr qa <model.apr> --report markdown`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example analysis_qa_report` exits 0
//! 2. [x] `cargo test --example analysis_qa_report` passes
//! 3. [x] Deterministic output (same seed → same gate matrix)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Markdown report is valid GFM table (row/column alignment)
//!
//! ## Learning Objective
//! Demonstrates how to aggregate `apr qa` across N candidate models in a CI
//! pipeline — run the 6 canonical gates (format, integrity, performance,
//! size, accuracy, security) on three deterministic synthetic models, then
//! render a pass/fail matrix as a GitHub-flavoured-Markdown table. Teaches
//! the operational pattern Sculley et al. call "monitoring beyond model
//! accuracy" — the gate matrix is the artifact a release manager reads.
//!
//! ## Run Command
//! ```bash
//! cargo run --example analysis_qa_report
//! ```
//!
//! ## Format Variants
//! ```bash
//! apr qa model.apr          # APR native
//! apr qa model.gguf         # GGUF
//! apr qa model.safetensors  # HF SafeTensors
//! ```
//!
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};

/// The six canonical QA gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    Format,
    Integrity,
    Performance,
    Size,
    Accuracy,
    Security,
}

impl Gate {
    pub fn all() -> [Gate; 6] {
        [
            Gate::Format,
            Gate::Integrity,
            Gate::Performance,
            Gate::Size,
            Gate::Accuracy,
            Gate::Security,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            Gate::Format => "Format",
            Gate::Integrity => "Integrity",
            Gate::Performance => "Performance",
            Gate::Size => "Size",
            Gate::Accuracy => "Accuracy",
            Gate::Security => "Security",
        }
    }
}

/// A single gate outcome for a single model.
#[derive(Debug, Clone, PartialEq)]
pub struct GateOutcome {
    pub gate: Gate,
    pub passed: bool,
    pub detail: String,
}

/// QA outcome for one candidate model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelQa {
    pub name: String,
    pub size_bytes: usize,
    pub outcomes: Vec<GateOutcome>,
}

impl ModelQa {
    #[must_use]
    pub fn overall_pass(&self) -> bool {
        self.outcomes.iter().all(|o| o.passed)
    }

    #[must_use]
    pub fn passed_count(&self) -> usize {
        self.outcomes.iter().filter(|o| o.passed).count()
    }
}

/// Build a deterministic synthetic APR model bundle with the given name.
fn build_synthetic_model(name: &str, dim: usize) -> Vec<u8> {
    let seed = hash_name_to_seed(name);
    let weights = generate_model_payload(seed, dim * dim);
    let bias = generate_model_payload(seed + 1, dim);
    ModelBundleV2::new()
        .with_name(name)
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("weight", vec![dim, dim], weights)
        .add_tensor("bias", vec![dim], bias)
        .build()
}

/// Corrupt a clean bundle in a specific way to force a gate to fail.
fn corrupt_bundle(bundle: &[u8], mode: &str) -> Vec<u8> {
    let mut bad = bundle.to_vec();
    match mode {
        // Break APR2 magic — Format gate fails
        "format" => bad[0] = b'X',
        // Inject a URL pattern — Security gate fails
        "security" => {
            let url = b"http://evil.com";
            let off = 100.min(bad.len().saturating_sub(url.len()));
            if off + url.len() <= bad.len() {
                bad[off..off + url.len()].copy_from_slice(url);
            }
        }
        // Leave clean
        _ => {}
    }
    bad
}

/// Run the 6-gate matrix against one model bundle and record outcomes.
#[must_use]
pub fn run_qa_matrix(name: &str, bundle: &[u8]) -> ModelQa {
    let mut outcomes = Vec::new();

    // Gate 1: Format — APR2 magic + minimum length.
    let format_ok = bundle.len() >= 8 && &bundle[0..4] == b"APR2";
    outcomes.push(GateOutcome {
        gate: Gate::Format,
        passed: format_ok,
        detail: if format_ok {
            "APR2 magic ok".into()
        } else {
            "magic mismatch".into()
        },
    });

    // Gate 2: Integrity — no long runs of zeros (proxy for truncated payload).
    let max_zero = count_max_zero_run(bundle);
    let integrity_ok = max_zero < 2048;
    outcomes.push(GateOutcome {
        gate: Gate::Integrity,
        passed: integrity_ok,
        detail: format!("max zero run = {max_zero}"),
    });

    // Gate 3: Performance — size-based latency proxy; 64 KiB ≈ 1 ms budget.
    let est_ms = bundle.len() as f64 / 65536.0;
    let perf_ok = est_ms <= 10.0;
    outcomes.push(GateOutcome {
        gate: Gate::Performance,
        passed: perf_ok,
        detail: format!("est {est_ms:.2} ms"),
    });

    // Gate 4: Size — must be under 10 MiB.
    let size_ok = bundle.len() < 10 * 1024 * 1024;
    outcomes.push(GateOutcome {
        gate: Gate::Size,
        passed: size_ok,
        detail: format!("{} bytes", bundle.len()),
    });

    // Gate 5: Accuracy — synthetic; we accept any non-empty model.
    let acc_ok = !bundle.is_empty();
    outcomes.push(GateOutcome {
        gate: Gate::Accuracy,
        passed: acc_ok,
        detail: "synthetic model: n/a".into(),
    });

    // Gate 6: Security — no embedded URL.
    let has_url = bundle
        .windows(7)
        .any(|w| w.starts_with(b"http://") || w.starts_with(b"https:/"));
    let sec_ok = !has_url;
    outcomes.push(GateOutcome {
        gate: Gate::Security,
        passed: sec_ok,
        detail: if sec_ok {
            "no embedded URLs".into()
        } else {
            "URL found in payload".into()
        },
    });

    ModelQa {
        name: name.to_string(),
        size_bytes: bundle.len(),
        outcomes,
    }
}

fn count_max_zero_run(data: &[u8]) -> usize {
    let mut best = 0usize;
    let mut run = 0usize;
    for b in data {
        if *b == 0 {
            run += 1;
            if run > best {
                best = run;
            }
        } else {
            run = 0;
        }
    }
    best
}

/// Render a GFM markdown report: one row per model, one column per gate.
#[must_use]
pub fn render_markdown(reports: &[ModelQa]) -> String {
    let mut s = String::new();
    s.push_str("# QA Report\n\n");
    s.push_str(
        "| Model | Format | Integrity | Performance | Size | Accuracy | Security | Overall |\n",
    );
    s.push_str(
        "|-------|--------|-----------|-------------|------|----------|----------|---------|\n",
    );
    for r in reports {
        s.push_str(&format!("| {} ", r.name));
        for g in Gate::all() {
            let cell = r.outcomes.iter().find(|o| o.gate == g).map_or("?", |o| {
                if o.passed {
                    "PASS"
                } else {
                    "FAIL"
                }
            });
            s.push_str(&format!("| {cell} "));
        }
        s.push_str(&format!(
            "| {} |\n",
            if r.overall_pass() { "PASS" } else { "FAIL" }
        ));
    }
    s
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("analysis_qa_report")?;
    println!("=== Recipe: {} ===\n", ctx.name());

    // --- Section 1: Build three synthetic candidate models ----------------
    println!("--- Building 3 synthetic models ---");
    let clean = build_synthetic_model("model-clean", 32);
    let bad_format = corrupt_bundle(&clean, "format");
    let bad_sec = corrupt_bundle(&clean, "security");
    let bad_sec_named = bad_sec.clone();

    println!("  model-clean       : {} bytes", clean.len());
    println!(
        "  model-bad-format  : {} bytes (APR2 magic broken)",
        bad_format.len()
    );
    println!(
        "  model-bad-security: {} bytes (URL injected)",
        bad_sec_named.len()
    );

    // Persist all three to the tempdir so a real `apr qa` invocation could pick them up.
    for (name, bytes) in [
        ("model-clean.apr", &clean),
        ("model-bad-format.apr", &bad_format),
        ("model-bad-security.apr", &bad_sec_named),
    ] {
        let p = ctx.path(name);
        std::fs::write(&p, bytes)?;
    }
    println!();

    // --- Section 2: Run the 6-gate matrix on each ------------------------
    println!("--- Running QA matrix ---");
    let reports = vec![
        run_qa_matrix("model-clean", &clean),
        run_qa_matrix("model-bad-format", &bad_format),
        run_qa_matrix("model-bad-security", &bad_sec),
    ];
    for r in &reports {
        println!(
            "  {:<22} — {}/{} gates pass ({})",
            r.name,
            r.passed_count(),
            r.outcomes.len(),
            if r.overall_pass() { "PASS" } else { "FAIL" },
        );
    }
    println!();

    // --- Section 3: Render and persist the markdown report ---------------
    let md = render_markdown(&reports);
    let md_path = ctx.path("qa_report.md");
    std::fs::write(&md_path, &md)?;
    println!("--- Markdown Report ---\n");
    print!("{md}");
    println!();
    println!("Wrote {}", md_path.display());
    println!();

    // --- Section 4: Aggregate metrics -----------------------------------
    let total_models = reports.len();
    let total_gate_cells = reports.iter().map(|r| r.outcomes.len()).sum::<usize>();
    let passing_cells = reports.iter().map(ModelQa::passed_count).sum::<usize>();
    let overall_pass = reports.iter().filter(|r| r.overall_pass()).count();

    ctx.record_metric("models_total", total_models as i64);
    ctx.record_metric("models_pass", overall_pass as i64);
    ctx.record_metric("gate_cells_total", total_gate_cells as i64);
    ctx.record_metric("gate_cells_pass", passing_cells as i64);
    ctx.record_string_metric(
        "verdict",
        if overall_pass == total_models {
            "ALL_PASS"
        } else {
            "SOME_FAIL"
        },
    );

    // Sanity: the JSON summary is also available for CI dashboards.
    let summary = serde_json::json!({
        "schema_version": 1,
        "sub": "qa",
        "models": reports.iter().map(|r| serde_json::json!({
            "name": r.name,
            "size_bytes": r.size_bytes,
            "passed": r.passed_count(),
            "total": r.outcomes.len(),
            "overall_pass": r.overall_pass(),
        })).collect::<Vec<_>>(),
    });
    let summary_path = ctx.path("qa_report.json");
    std::fs::write(
        &summary_path,
        serde_json::to_vec_pretty(&summary)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;
    println!("Wrote {}", summary_path.display());

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_model_passes_all_gates() {
        let bundle = build_synthetic_model("t-clean", 16);
        let r = run_qa_matrix("t-clean", &bundle);
        assert!(
            r.overall_pass(),
            "clean model should pass; outcomes={:?}",
            r.outcomes
        );
        assert_eq!(r.outcomes.len(), 6);
    }

    #[test]
    fn test_bad_format_fails_format_gate() {
        let bundle = build_synthetic_model("t-bad-format", 16);
        let bad = corrupt_bundle(&bundle, "format");
        let r = run_qa_matrix("t-bad-format", &bad);
        let fmt = r
            .outcomes
            .iter()
            .find(|o| o.gate == Gate::Format)
            .expect("must have format gate");
        assert!(
            !fmt.passed,
            "format gate should fail after magic corruption"
        );
        assert!(!r.overall_pass());
    }

    #[test]
    fn test_bad_security_fails_security_gate() {
        let bundle = build_synthetic_model("t-bad-sec", 16);
        let bad = corrupt_bundle(&bundle, "security");
        let r = run_qa_matrix("t-bad-sec", &bad);
        let sec = r
            .outcomes
            .iter()
            .find(|o| o.gate == Gate::Security)
            .expect("must have security gate");
        assert!(!sec.passed);
        assert!(!r.overall_pass());
    }

    #[test]
    fn test_markdown_has_header_and_rows() {
        let bundle = build_synthetic_model("t-md", 16);
        let reports = vec![run_qa_matrix("a", &bundle), run_qa_matrix("b", &bundle)];
        let md = render_markdown(&reports);
        assert!(md.contains("| Model |"));
        assert!(md.contains("| a |"));
        assert!(md.contains("| b |"));
        assert!(md.contains("Overall"));
    }

    #[test]
    fn test_gate_all_returns_six() {
        assert_eq!(Gate::all().len(), 6);
    }
}
