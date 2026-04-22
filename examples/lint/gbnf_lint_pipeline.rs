//! # Recipe: GBNF Lint — Pipeline Composition with Conformance Test
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr gbnf-lint grammar.gbnf --conformance samples.jsonl`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example gbnf_lint_pipeline` exits 0
//! 2. [x] `cargo test --example gbnf_lint_pipeline` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Combines GBNF lint with a conformance sweep. After static linting passes,
//! the recipe runs a tiny in-memory conformance test that matches a collection
//! of candidate strings against the grammar's terminal set, so a CI job can
//! fail on either STATIC (lint) or DYNAMIC (conformance) regressions.
//!
//! ## Run Command
//! ```bash
//! cargo run --example gbnf_lint_pipeline
//! ```
//!
//! ## References
//! - Willard, B. T. & Louf, R. (2023). *Efficient Guided Generation for Large Language Models*. arXiv:2307.09702

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use std::collections::HashMap;

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

/// Very small conformance check: the grammar defines a union of literal
/// answers — we accept a candidate if it matches any quoted terminal from
/// the `root` RHS.
pub fn accepts<S: std::hash::BuildHasher>(
    grammar: &HashMap<String, String, S>,
    candidate: &str,
) -> bool {
    let Some(rhs) = grammar.get("root") else {
        return false;
    };
    let mut literals = Vec::new();
    let mut in_string = false;
    let mut buf = String::new();
    for ch in rhs.chars() {
        if ch == '"' {
            if in_string {
                literals.push(buf.clone());
                buf.clear();
            }
            in_string = !in_string;
            continue;
        }
        if in_string {
            buf.push(ch);
        }
    }
    literals.iter().any(|lit| lit == candidate)
}

pub fn lint<S: std::hash::BuildHasher>(grammar: &HashMap<String, String, S>) -> Vec<&'static str> {
    let mut out = Vec::new();
    if !grammar.contains_key("root") {
        out.push("GBNF-001");
    }
    out
}

const GRAMMAR: &str = r#"
root ::= "yes" | "no" | "maybe"
"#;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("gbnf_lint_pipeline")?;
    let gp = ctx.path("answers.gbnf");
    std::fs::write(&gp, GRAMMAR)?;

    let grammar = parse_gbnf(GRAMMAR);
    let static_errs = lint(&grammar);

    let candidates = ["yes", "no", "maybe", "perhaps", ""];
    let mut passed = 0usize;
    let mut rejected = 0usize;
    let mut conformance: Vec<(String, bool)> = Vec::new();
    for c in candidates {
        let ok = accepts(&grammar, c);
        if ok {
            passed += 1;
        } else {
            rejected += 1;
        }
        conformance.push((c.to_string(), ok));
    }

    println!("=== Recipe: {} ===", ctx.name());
    println!("Static lint errors: {}", static_errs.len());
    println!("Conformance:");
    for (c, ok) in &conformance {
        println!(
            "  {:<10} -> {}",
            format!("{:?}", c),
            if *ok { "accept" } else { "reject" }
        );
    }
    println!(
        "\nSummary: {} passed, {} rejected (expected reject: \"perhaps\", \"\")",
        passed, rejected
    );

    ctx.record_metric("static_errors", static_errs.len() as i64);
    ctx.record_metric("accepted", passed as i64);
    ctx.record_metric("rejected", rejected as i64);
    ctx.record_string_metric(
        "verdict",
        if static_errs.is_empty() && passed == 3 && rejected == 2 {
            "PASS"
        } else {
            "FAIL"
        },
    );
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_known_literal() {
        let g = parse_gbnf(GRAMMAR);
        assert!(accepts(&g, "yes"));
        assert!(accepts(&g, "no"));
        assert!(accepts(&g, "maybe"));
    }

    #[test]
    fn rejects_unknown() {
        let g = parse_gbnf(GRAMMAR);
        assert!(!accepts(&g, "perhaps"));
    }

    #[test]
    fn static_lint_clean() {
        let g = parse_gbnf(GRAMMAR);
        assert!(lint(&g).is_empty());
    }
}
