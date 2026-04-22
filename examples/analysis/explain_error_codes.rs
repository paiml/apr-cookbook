//! # Recipe: Explain — Error Code Lookup (Multilingual)
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr explain --error APR-E042 --lang es`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example explain_error_codes` exits 0
//! 2. [x] `cargo test --example explain_error_codes` passes
//! 3. [x] Deterministic output (static lookup)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr explain --error` in-process (no shell-out)
//! 10. [x] Unit tests cover known codes, unknown codes, fallback lang
//!
//! ## Learning Objective
//! Implements a multilingual error-code explainer. Given an error code like
//! `APR-E042` and a language tag, returns a structured explanation in the
//! requested language, falling back to English if unavailable.
//!
//! ## Run Command
//! ```bash
//! cargo run --example explain_error_codes
//! ```
//!
//! ## References
//! - Ko, A.J. & Myers, B.A. (2008). *Debugging Reinvented*. ICSE. DOI: 10.1145/1368088.1368132

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
struct ErrorEntry {
    code: String,
    severity: &'static str,
    translations: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
struct Explanation {
    code: String,
    severity: String,
    lang: String,
    message: String,
    fallback: bool,
}

fn build_error_table() -> Vec<ErrorEntry> {
    let mut out = Vec::new();

    let mut t042 = BTreeMap::new();
    t042.insert(
        "en".into(),
        "Invalid .apr magic bytes; expected APR1/APR2".into(),
    );
    t042.insert(
        "es".into(),
        "Bytes mágicos .apr inválidos; se esperaba APR1/APR2".into(),
    );
    t042.insert(
        "fr".into(),
        "Octets magiques .apr invalides; APR1/APR2 attendu".into(),
    );
    out.push(ErrorEntry {
        code: "APR-E042".into(),
        severity: "error",
        translations: t042,
    });

    let mut t101 = BTreeMap::new();
    t101.insert("en".into(), "Tensor shape mismatch during load".into());
    t101.insert(
        "es".into(),
        "Discordancia en forma del tensor durante la carga".into(),
    );
    out.push(ErrorEntry {
        code: "APR-E101".into(),
        severity: "error",
        translations: t101,
    });

    let mut t210 = BTreeMap::new();
    t210.insert("en".into(), "Quantization scale underflow".into());
    t210.insert(
        "fr".into(),
        "Dépassement bas de l'échelle de quantification".into(),
    );
    out.push(ErrorEntry {
        code: "APR-W210".into(),
        severity: "warn",
        translations: t210,
    });

    out
}

/// Look up an error code and explain it in the requested language.
/// Falls back to English if the language is missing.
fn explain(table: &[ErrorEntry], code: &str, lang: &str) -> Option<Explanation> {
    let entry = table.iter().find(|e| e.code == code)?;
    let (msg, fallback, used_lang) = entry.translations.get(lang).map_or_else(
        || {
            let fallback_msg = entry.translations.get("en").cloned().unwrap_or_default();
            (fallback_msg, true, "en".to_string())
        },
        |m| (m.clone(), false, lang.to_string()),
    );
    Some(Explanation {
        code: entry.code.clone(),
        severity: entry.severity.to_string(),
        lang: used_lang,
        message: msg,
        fallback,
    })
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("explain_error_codes")?;
    println!("=== Recipe: {} ===", ctx.name());

    let table = build_error_table();
    let queries = [
        ("APR-E042", "es"),
        ("APR-E042", "fr"),
        ("APR-E101", "de"), // fallback -> en
        ("APR-W210", "en"),
        ("APR-E999", "en"), // unknown
    ];

    let mut explanations = Vec::new();
    println!("\n--- Explanations ---");
    for (code, lang) in queries {
        match explain(&table, code, lang) {
            Some(exp) => {
                println!(
                    "[{}] {} ({}): {} {}",
                    exp.severity,
                    exp.code,
                    exp.lang,
                    exp.message,
                    if exp.fallback { "(fallback)" } else { "" }
                );
                explanations.push(exp);
            }
            None => {
                println!("[?] {}: unknown error code", code);
            }
        }
    }

    let report = json!({
        "recipe": ctx.name(),
        "queries": queries.iter().map(|(c, l)| json!({"code": c, "lang": l})).collect::<Vec<_>>(),
        "explanations": explanations.iter().map(|e| json!({
            "code": e.code,
            "severity": e.severity,
            "lang": e.lang,
            "message": e.message,
            "fallback": e.fallback,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("explain-errors.json");
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out, bytes)?;

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_code_with_translation() {
        let t = build_error_table();
        let e = explain(&t, "APR-E042", "es").expect("entry");
        assert!(!e.fallback);
        assert_eq!(e.lang, "es");
        assert!(e.message.contains("inválidos"));
    }

    #[test]
    fn known_code_falls_back_to_english() {
        let t = build_error_table();
        let e = explain(&t, "APR-E042", "zz").expect("entry");
        assert!(e.fallback);
        assert_eq!(e.lang, "en");
    }

    #[test]
    fn unknown_code_returns_none() {
        let t = build_error_table();
        assert!(explain(&t, "APR-E999", "en").is_none());
    }

    #[test]
    fn warning_severity_preserved() {
        let t = build_error_table();
        let e = explain(&t, "APR-W210", "en").expect("entry");
        assert_eq!(e.severity, "warn");
    }
}
