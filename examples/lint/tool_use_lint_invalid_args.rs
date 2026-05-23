//! # Recipe: Tool-Use Lint — Invalid Arguments JSON
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr tool-use-lint response.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example tool_use_lint_invalid_args` exits 0
//! 2. [x] `cargo test --example tool_use_lint_invalid_args` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! The most common real-world failure: the model emits a SYNTAX-BROKEN JSON
//! string in `arguments` (trailing comma, unquoted key, literal newline in
//! string). Shows how the lint catches each class and how a repair pass could
//! optionally try lax parsing before failing.
//!
//! ## Run Command
//! ```bash
//! cargo run --example tool_use_lint_invalid_args
//! ```
//!
//! ## References
//! - Schick, T. et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. arXiv:2302.04761

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Finding {
    pub rule: &'static str,
    pub severity: &'static str,
    pub message: String,
}

pub fn lint_arguments(args: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    match serde_json::from_str::<Value>(args) {
        Ok(v) if v.is_object() => {}
        Ok(_) => out.push(Finding {
            rule: "TOOL-005",
            severity: "error",
            message: "arguments is not a JSON object".into(),
        }),
        Err(e) => out.push(Finding {
            rule: "TOOL-005",
            severity: "error",
            message: format!("arguments JSON parse error: {e}"),
        }),
    }
    // Heuristic: trailing commas are a red flag for model-emitted JSON.
    if args.contains(",}") || args.contains(", }") || args.contains(",]") {
        out.push(Finding {
            rule: "TOOL-007",
            severity: "warn",
            message: "arguments contains a trailing comma before } or ]".into(),
        });
    }
    out
}

fn build_broken_samples() -> Vec<(&'static str, &'static str)> {
    vec![
        ("trailing_comma", r#"{"city":"Tokyo","units":"metric",}"#),
        ("unquoted_key", r#"{city:"Tokyo","units":"metric"}"#),
        ("literal_nl", "{\"city\":\"Tok\nyo\",\"units\":\"metric\"}"),
        ("non_object", r#""just a string""#),
    ]
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("tool_use_lint_invalid_args")?;
    let samples = build_broken_samples();

    let mut total_err = 0usize;
    let mut total_warn = 0usize;
    println!("=== Recipe: {} ===", ctx.name());
    for (label, raw) in &samples {
        let path = ctx.path(&format!("{label}.json"));
        std::fs::write(&path, raw)?;
        let findings = lint_arguments(raw);
        let errs = findings.iter().filter(|f| f.severity == "error").count();
        let warns = findings.iter().filter(|f| f.severity == "warn").count();
        total_err += errs;
        total_warn += warns;
        println!("{label} ({} errors, {} warnings)", errs, warns);
        for f in &findings {
            println!("  [{}] {} — {}", f.severity, f.rule, f.message);
        }
    }

    ctx.record_metric("total_errors", total_err as i64);
    ctx.record_metric("total_warnings", total_warn as i64);
    ctx.record_string_metric("verdict", if total_err == 0 { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trailing_comma_warns() {
        let f = lint_arguments(r#"{"city":"Tokyo",}"#);
        assert!(f.iter().any(|x| x.rule == "TOOL-007"));
    }

    #[test]
    fn unquoted_key_errors() {
        let f = lint_arguments(r#"{city:"Tokyo"}"#);
        assert!(f.iter().any(|x| x.rule == "TOOL-005"));
    }

    #[test]
    fn non_object_errors() {
        let f = lint_arguments(r#""just a string""#);
        assert!(f.iter().any(|x| x.rule == "TOOL-005"));
    }

    #[test]
    fn valid_json_clean() {
        assert!(lint_arguments(r#"{"city":"Tokyo"}"#).is_empty());
    }
}
