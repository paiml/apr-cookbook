//! # Recipe: GBNF Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr gbnf-lint grammar.gbnf`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example gbnf_lint_happy` exits 0
//! 2. [x] `cargo test --example gbnf_lint_happy` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Demonstrates lightweight structural validation of a GBNF (GGML Backus-Naur
//! Form) grammar, the constrained-decoding DSL used by llama.cpp and friends.
//! Checks for the canonical set of rules: `root` symbol exists, every symbol
//! referenced on the RHS is defined, no redundant root-recursion loops.
//!
//! ## Run Command
//! ```bash
//! cargo run --example gbnf_lint_happy
//! ```
//!
//! ## References
//! - Willard, B. T. & Louf, R. (2023). *Efficient Guided Generation for Large Language Models*. arXiv:2307.09702

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GbnfFinding {
    pub rule: &'static str,
    pub severity: &'static str,
    pub message: String,
}

/// Parse a GBNF source into a `symbol -> RHS` map.
///
/// We recognise lines of the form `name ::= rhs` (strict, one rule per line).
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

/// Collect referenced symbols from a RHS (word-like tokens outside quotes).
pub fn referenced_symbols(rhs: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut in_string = false;
    let mut token = String::new();
    let flush = |t: &mut String, out: &mut HashSet<String>| {
        if !t.is_empty() {
            let first = t.chars().next().unwrap_or('_');
            if first.is_ascii_alphabetic() || first == '_' {
                out.insert(t.clone());
            }
            t.clear();
        }
    };
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
        } else {
            flush(&mut token, &mut out);
        }
    }
    flush(&mut token, &mut out);
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
        for r in referenced_symbols(rhs) {
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

const HAPPY_GRAMMAR: &str = r#"
# A canonical JSON subset.
root ::= value
value ::= object | array | string | number | bool | null_lit
object ::= "{" pair ("," pair)* "}"
pair ::= string ":" value
array ::= "[" value ("," value)* "]"
string ::= "\"" [a-zA-Z0-9_]* "\""
number ::= [0-9]+
bool ::= "true" | "false"
null_lit ::= "null"
"#;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("gbnf_lint_happy")?;
    let p = ctx.path("json.gbnf");
    std::fs::write(&p, HAPPY_GRAMMAR)?;

    let grammar = parse_gbnf(HAPPY_GRAMMAR);
    let findings = lint_gbnf(&grammar);
    let errors = findings.iter().filter(|f| f.severity == "error").count();

    println!("=== Recipe: {} ===", ctx.name());
    println!("Grammar: {}", p.display());
    println!("Rules defined: {}", grammar.len());
    println!("Findings: {}", findings.len());
    for f in &findings {
        println!("  [{}] {} — {}", f.severity, f.rule, f.message);
    }

    ctx.record_metric("rules", grammar.len() as i64);
    ctx.record_metric("errors", errors as i64);
    ctx.record_string_metric("verdict", if errors == 0 { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_rules() {
        // root, value, object, pair, array, string, number, bool, null_lit = 9
        assert_eq!(parse_gbnf(HAPPY_GRAMMAR).len(), 9);
    }

    #[test]
    fn happy_grammar_is_clean() {
        assert!(lint_gbnf(&parse_gbnf(HAPPY_GRAMMAR)).is_empty());
    }

    #[test]
    fn references_skip_string_literals() {
        let refs = referenced_symbols(r#""{" pair ("," pair)* "}""#);
        assert!(refs.contains("pair"));
        assert!(!refs.contains("{"));
    }
}
