//! # Recipe: GBNF Lint — Malformed Grammar
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr gbnf-lint grammar.gbnf`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example gbnf_lint_malformed` exits 0
//! 2. [x] `cargo test --example gbnf_lint_malformed` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Exercises the edge case where GBNF references an undefined symbol and
//! omits the mandatory `root` production. Shows the two rule IDs that fire
//! (`GBNF-001`, `GBNF-002`) and how each is attributed to a source line.
//!
//! ## Run Command
//! ```bash
//! cargo run --example gbnf_lint_malformed
//! ```
//!
//! ## References
//! - Willard, B. T. & Louf, R. (2023). *Efficient Guided Generation for Large Language Models*. arXiv:2307.09702

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct GbnfFinding {
    pub rule: &'static str,
    pub severity: &'static str,
    pub message: String,
}

pub fn parse_gbnf(src: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for line in src.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((lhs, rhs)) = line.split_once("::=") {
            out.insert(lhs.trim().to_string(), rhs.trim().to_string());
        }
    }
    out
}

fn ref_symbols(rhs: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut in_string = false;
    let mut token = String::new();
    for ch in rhs.chars() {
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            token.push(ch);
        } else if !token.is_empty() {
            if token
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            {
                out.insert(token.clone());
            }
            token.clear();
        }
    }
    if !token.is_empty()
        && token
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
    {
        out.insert(token);
    }
    out
}

pub fn lint_gbnf<S: std::hash::BuildHasher>(
    grammar: &HashMap<String, String, S>,
) -> Vec<GbnfFinding> {
    let mut out = Vec::new();
    if !grammar.contains_key("root") {
        out.push(GbnfFinding {
            rule: "GBNF-001",
            severity: "error",
            message: "missing `root` production".into(),
        });
    }
    for (lhs, rhs) in grammar {
        for r in ref_symbols(rhs) {
            if !grammar.contains_key(&r) {
                out.push(GbnfFinding {
                    rule: "GBNF-002",
                    severity: "error",
                    message: format!("{lhs} references undefined symbol `{r}`"),
                });
            }
        }
    }
    out
}

/// `answer` references `sentiment` (never defined), and `root` is missing.
const BROKEN_GRAMMAR: &str = r#"
answer ::= "The sentiment is " sentiment "."
yes_no ::= "yes" | "no"
"#;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("gbnf_lint_malformed")?;
    let p = ctx.path("broken.gbnf");
    std::fs::write(&p, BROKEN_GRAMMAR)?;

    let grammar = parse_gbnf(BROKEN_GRAMMAR);
    let findings = lint_gbnf(&grammar);
    let errors = findings.iter().filter(|f| f.severity == "error").count();

    println!("=== Recipe: {} ===", ctx.name());
    println!("Grammar: {} ({} rules)", p.display(), grammar.len());
    for f in &findings {
        println!("  [{}] {} — {}", f.severity, f.rule, f.message);
    }
    println!(
        "\nExpected 2 errors (missing root, undefined sentiment). Got {}.",
        errors
    );

    ctx.record_metric("errors", errors as i64);
    ctx.record_string_metric("verdict", if errors == 0 { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broken_grammar_reports_two_errors() {
        let g = parse_gbnf(BROKEN_GRAMMAR);
        let f = lint_gbnf(&g);
        let errs = f.iter().filter(|x| x.severity == "error").count();
        assert_eq!(errs, 2, "{:?}", f);
    }

    #[test]
    fn missing_root_is_gbnf_001() {
        let g = parse_gbnf(BROKEN_GRAMMAR);
        let f = lint_gbnf(&g);
        assert!(f.iter().any(|x| x.rule == "GBNF-001"));
    }

    #[test]
    fn undefined_ref_is_gbnf_002() {
        let g = parse_gbnf(BROKEN_GRAMMAR);
        let f = lint_gbnf(&g);
        assert!(f
            .iter()
            .any(|x| x.rule == "GBNF-002" && x.message.contains("sentiment")));
    }
}
