//! # Recipe: Batch Model-Registry Validation
//!
//! **Category**: cli
//! **CLI Equivalent**: `apr validate --batch registry/*.apr`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example validate_batch` exits 0
//! 2. [x] `cargo test --example validate_batch` passes
//! 3. [x] Deterministic output (synthetic registry, fixed hashes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped via `CookbookError::Serialization`
//! 9. [x] Simulates `apr validate --batch` in-process (no shell-out)
//! 10. [x] Aggregate summary + per-model status table
//!
//! ## Learning Objective
//! Demonstrates the `apr validate --batch` flow: iterate a synthetic model
//! registry, run 5 structural checks against each, aggregate PASS/FAIL/WARN
//! counts, and emit a per-model status table that mirrors the CLI's output.
//! The recipe teaches how registry-wide validation gates release decisions.
//!
//! ## Run Command
//! ```bash
//! cargo run --example validate_batch
//! ```
//!
//! ## Format Variants
//! ```bash
//! apr validate --batch registry/*.apr          # APR native
//! apr validate --batch registry/*.gguf         # GGUF
//! apr validate --batch registry/*.safetensors  # SafeTensors
//! ```
//!
//! ## References
//! - Amershi, S. et al. (2019). *Software Engineering for Machine Learning: A Case Study*. ICSE-SEIP. DOI: 10.1109/ICSE-SEIP.2019.00042

use apr_cookbook::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Warn,
    Fail,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
        }
    }

    pub fn rank(self) -> u8 {
        match self {
            Self::Pass => 0,
            Self::Warn => 1,
            Self::Fail => 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub name: String,
    pub size_mb: u64,
    pub checksum_ok: bool,
    pub has_metadata: bool,
    pub quantization: String,
    pub signature_present: bool,
}

#[derive(Debug, Clone)]
pub struct CheckResult {
    pub rule: &'static str,
    pub verdict: Verdict,
    pub detail: String,
}

/// Five structural checks, one per rule.
pub fn run_checks(m: &ModelEntry) -> Vec<CheckResult> {
    let mut out = Vec::with_capacity(5);

    out.push(CheckResult {
        rule: "size_within_limits",
        verdict: if m.size_mb <= 20_000 {
            Verdict::Pass
        } else {
            Verdict::Warn
        },
        detail: format!("{} MB", m.size_mb),
    });

    out.push(CheckResult {
        rule: "checksum_matches",
        verdict: if m.checksum_ok {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        detail: if m.checksum_ok {
            "sha256 ok".into()
        } else {
            "sha256 mismatch".into()
        },
    });

    out.push(CheckResult {
        rule: "metadata_present",
        verdict: if m.has_metadata {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        detail: if m.has_metadata {
            "metadata.json found".into()
        } else {
            "metadata.json missing".into()
        },
    });

    out.push(CheckResult {
        rule: "known_quantization",
        verdict: match m.quantization.as_str() {
            "fp32" | "fp16" | "bf16" | "q8_0" | "q4_k_m" | "q4_0" => Verdict::Pass,
            "" | "unknown" => Verdict::Fail,
            _ => Verdict::Warn,
        },
        detail: format!("quant={}", m.quantization),
    });

    out.push(CheckResult {
        rule: "signature_present",
        verdict: if m.signature_present {
            Verdict::Pass
        } else {
            Verdict::Warn
        },
        detail: if m.signature_present {
            "ed25519 sig ok".into()
        } else {
            "unsigned (allowed, advisory)".into()
        },
    });

    out
}

pub fn rollup(results: &[CheckResult]) -> Verdict {
    results
        .iter()
        .map(|r| r.verdict)
        .max_by_key(|v| v.rank())
        .unwrap_or(Verdict::Pass)
}

pub fn synthetic_registry() -> Vec<ModelEntry> {
    vec![
        ModelEntry {
            name: "llama-3-8b-q4".into(),
            size_mb: 4_700,
            checksum_ok: true,
            has_metadata: true,
            quantization: "q4_k_m".into(),
            signature_present: true,
        },
        ModelEntry {
            name: "phi-3-mini-fp16".into(),
            size_mb: 7_400,
            checksum_ok: true,
            has_metadata: true,
            quantization: "fp16".into(),
            signature_present: true,
        },
        ModelEntry {
            name: "whisper-tiny-q8".into(),
            size_mb: 40,
            checksum_ok: true,
            has_metadata: true,
            quantization: "q8_0".into(),
            signature_present: false,
        },
        ModelEntry {
            name: "falcon-180b-fp16".into(),
            size_mb: 360_000,
            checksum_ok: true,
            has_metadata: true,
            quantization: "fp16".into(),
            signature_present: true,
        },
        ModelEntry {
            name: "experimental-kv".into(),
            size_mb: 800,
            checksum_ok: false,
            has_metadata: false,
            quantization: "unknown".into(),
            signature_present: false,
        },
    ]
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("validate_batch")?;
    let registry = synthetic_registry();

    println!("=== apr validate --batch (synthetic registry) ===\n");
    println!(
        "{:<24} {:<10} {:<10} {:<10} {:<10}",
        "MODEL", "VERDICT", "SIZE", "QUANT", "SIGNED"
    );
    println!("{}", "-".repeat(68));

    let mut summary = [0u32; 3];
    let mut fail_rules: Vec<(String, String)> = Vec::new();

    for m in &registry {
        let results = run_checks(m);
        let verdict = rollup(&results);
        summary[verdict.rank() as usize] += 1;

        for r in &results {
            if r.verdict == Verdict::Fail {
                fail_rules.push((m.name.clone(), format!("{}: {}", r.rule, r.detail)));
            }
        }

        println!(
            "{:<24} {:<10} {:>6} MB  {:<10} {:<10}",
            m.name,
            verdict.label(),
            m.size_mb,
            m.quantization,
            if m.signature_present { "yes" } else { "no" },
        );
    }

    println!("\n--- Summary ---");
    println!(
        "  PASS: {}   WARN: {}   FAIL: {}",
        summary[0], summary[1], summary[2]
    );

    if !fail_rules.is_empty() {
        println!("\n--- FAILED checks ---");
        for (model, rule) in &fail_rules {
            println!("  {model}  -> {rule}");
        }
    }

    ctx.record_metric("registry_size", registry.len() as i64);
    ctx.record_metric("passed", i64::from(summary[0]));
    ctx.record_metric("warned", i64::from(summary[1]));
    ctx.record_metric("failed", i64::from(summary[2]));
    ctx.record_metric("failed_rules", fail_rules.len() as i64);

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_happy_model_passes_all() {
        let m = ModelEntry {
            name: "ok".into(),
            size_mb: 10,
            checksum_ok: true,
            has_metadata: true,
            quantization: "fp16".into(),
            signature_present: true,
        };
        let results = run_checks(&m);
        assert!(results.iter().all(|r| r.verdict == Verdict::Pass));
        assert_eq!(rollup(&results), Verdict::Pass);
    }

    #[test]
    fn test_bad_checksum_fails_rollup() {
        let m = ModelEntry {
            name: "bad".into(),
            size_mb: 10,
            checksum_ok: false,
            has_metadata: true,
            quantization: "fp16".into(),
            signature_present: true,
        };
        assert_eq!(rollup(&run_checks(&m)), Verdict::Fail);
    }

    #[test]
    fn test_unsigned_warns_but_does_not_fail() {
        let m = ModelEntry {
            name: "unsigned".into(),
            size_mb: 10,
            checksum_ok: true,
            has_metadata: true,
            quantization: "fp16".into(),
            signature_present: false,
        };
        assert_eq!(rollup(&run_checks(&m)), Verdict::Warn);
    }

    #[test]
    fn test_synthetic_registry_has_five_entries() {
        assert_eq!(synthetic_registry().len(), 5);
    }

    #[test]
    fn test_verdict_rank_order() {
        assert!(Verdict::Pass.rank() < Verdict::Warn.rank());
        assert!(Verdict::Warn.rank() < Verdict::Fail.rank());
    }
}
