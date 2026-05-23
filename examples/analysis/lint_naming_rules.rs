//! # Recipe: Lint — Custom Naming Convention Rule Engine
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr lint model.apr --rules naming.yaml`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example lint_naming_rules` exits 0
//! 2. [x] `cargo test --example lint_naming_rules` passes
//! 3. [x] Deterministic output (pure)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr lint --rules` in-process (no shell-out)
//! 10. [x] Unit tests cover snake_case, dotted, numeric-suffix, allowlist
//!
//! ## Learning Objective
//! Builds a small lint rule engine that checks tensor names against naming
//! conventions: snake_case identifiers, dotted layer paths, and a case-
//! sensitive allowlist of permitted top-level prefixes.
//!
//! ## Run Command
//! ```bash
//! cargo run --example lint_naming_rules
//! ```
//!
//! ## References
//! - Johnson, S.C. (1977). *lint, a C Program Checker*. Bell Labs CSTR 65.

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone)]
struct LintRule {
    id: &'static str,
    description: &'static str,
}

#[derive(Debug, Clone)]
struct LintFinding {
    rule: &'static str,
    tensor: String,
    severity: &'static str,
    message: String,
}

fn rules() -> Vec<LintRule> {
    vec![
        LintRule {
            id: "NAME-001",
            description: "Each segment must be lowercase snake_case (a-z, 0-9, _)",
        },
        LintRule {
            id: "NAME-002",
            description: "Layer paths must use dotted notation 'layer.<idx>.<sub>'",
        },
        LintRule {
            id: "NAME-003",
            description: "Top-level prefix must be in the allowlist",
        },
    ]
}

fn is_snake_segment(seg: &str) -> bool {
    if seg.is_empty() {
        return false;
    }
    seg.chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

fn check_name(name: &str, allowlist: &[&str]) -> Vec<LintFinding> {
    let mut out = Vec::new();
    let segments: Vec<&str> = name.split('.').collect();

    // NAME-001: each segment snake_case.
    for seg in &segments {
        if !is_snake_segment(seg) {
            out.push(LintFinding {
                rule: "NAME-001",
                tensor: name.to_string(),
                severity: "error",
                message: format!("segment '{}' is not snake_case", seg),
            });
            break; // one finding per tensor
        }
    }

    // NAME-002: layer paths must be 'layer.<digits>.<something>'
    if segments.first() == Some(&"layer") {
        let idx_ok = segments
            .get(1)
            .is_some_and(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()));
        if !idx_ok || segments.len() < 3 {
            out.push(LintFinding {
                rule: "NAME-002",
                tensor: name.to_string(),
                severity: "error",
                message: "layer path must match 'layer.<idx>.<sub>'".into(),
            });
        }
    }

    // NAME-003: top-level prefix allowlist.
    let top = segments.first().copied().unwrap_or("");
    if !allowlist.iter().any(|p| p == &top) {
        out.push(LintFinding {
            rule: "NAME-003",
            tensor: name.to_string(),
            severity: "warn",
            message: format!("top-level prefix '{}' not in allowlist", top),
        });
    }

    out
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("lint_naming_rules")?;
    println!("=== Recipe: {} ===", ctx.name());

    let allowlist = ["embed", "layer", "head", "norm"];
    let names = [
        "embed.weight",
        "layer.0.attn.qkv",
        "layer.X.attn.qkv", // NAME-002 fail
        "LAYER.0.attn.qkv", // NAME-001 + NAME-003 fail
        "head.weight",
        "rogue_tensor", // NAME-003 fail
    ];

    let rs = rules();
    println!("\nRules in force:");
    for r in &rs {
        println!("  {} — {}", r.id, r.description);
    }

    let mut all_findings = Vec::new();
    println!("\n--- Findings ---");
    for name in names {
        let f = check_name(name, &allowlist);
        if f.is_empty() {
            println!("  OK   {}", name);
        } else {
            for ff in &f {
                println!(
                    "  [{}] {}: {} — {}",
                    ff.severity, ff.rule, ff.tensor, ff.message
                );
            }
        }
        all_findings.extend(f);
    }

    let errors = all_findings
        .iter()
        .filter(|f| f.severity == "error")
        .count();
    let warnings = all_findings.iter().filter(|f| f.severity == "warn").count();
    println!("\nErrors: {}  Warnings: {}", errors, warnings);

    let report = json!({
        "recipe": ctx.name(),
        "allowlist": allowlist,
        "rules": rs.iter().map(|r| json!({"id": r.id, "description": r.description})).collect::<Vec<_>>(),
        "findings": all_findings.iter().map(|f| json!({
            "rule": f.rule,
            "tensor": f.tensor,
            "severity": f.severity,
            "message": f.message,
        })).collect::<Vec<_>>(),
        "errors": errors,
        "warnings": warnings,
    });
    let out = ctx.path("lint-naming.json");
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
    fn snake_case_segment_accepts() {
        assert!(is_snake_segment("attn_qkv"));
        assert!(is_snake_segment("layer0"));
    }

    #[test]
    fn snake_case_rejects_uppercase() {
        assert!(!is_snake_segment("Attn"));
    }

    #[test]
    fn allowed_name_has_no_findings() {
        let f = check_name("embed.weight", &["embed", "layer", "head"]);
        assert!(f.is_empty(), "got: {:?}", f);
    }

    #[test]
    fn layer_without_digit_idx_flags_name_002() {
        let f = check_name("layer.X.attn", &["embed", "layer", "head"]);
        assert!(f.iter().any(|x| x.rule == "NAME-002"));
    }

    #[test]
    fn uppercase_segment_flags_name_001() {
        let f = check_name("LAYER.0.a", &["embed", "layer", "head"]);
        assert!(f.iter().any(|x| x.rule == "NAME-001"));
    }

    #[test]
    fn unknown_prefix_flags_name_003() {
        let f = check_name("rogue.x", &["embed", "layer"]);
        assert!(f.iter().any(|x| x.rule == "NAME-003"));
    }
}
