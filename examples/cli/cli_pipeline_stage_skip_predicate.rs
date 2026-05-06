//! # apr pipeline --skip-when — Stage Skip Predicate
//!
//! Stages can declare skip conditions: `--skip-when <predicate>`
//! evaluates flags to determine if a stage should be bypassed (e.g.,
//! `--skip-when "branch != main && env != prod"`). This recipe
//! builds the predicate evaluator over a small DSL: `flag == value`,
//! `flag != value`, `&&`, `||`.
//!
//! Demonstrates the **PIPE.6** recipe for PMAT-121 (apr pipeline coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PIPE-001 + Concourse / GitHub Actions when conventions
//!
//! Run with: cargo run --example cli_pipeline_stage_skip_predicate
//!
//! Added by PMAT-121 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
pub enum EvalVerdict {
    Skip,
    Run,
    InvalidPredicate { reason: String },
}

pub fn evaluate<S: ::std::hash::BuildHasher>(
    predicate: &str,
    flags: &HashMap<&str, &str, S>,
) -> EvalVerdict {
    let trimmed = predicate.trim();
    if trimmed.is_empty() {
        return EvalVerdict::InvalidPredicate {
            reason: "empty predicate".into(),
        };
    }
    match eval_or(trimmed, flags) {
        Ok(true) => EvalVerdict::Skip,
        Ok(false) => EvalVerdict::Run,
        Err(reason) => EvalVerdict::InvalidPredicate { reason },
    }
}

fn eval_or<S: ::std::hash::BuildHasher>(
    input: &str,
    flags: &HashMap<&str, &str, S>,
) -> std::result::Result<bool, String> {
    let mut acc = false;
    for clause in input.split("||") {
        let v = eval_and(clause, flags)?;
        acc = acc || v;
    }
    Ok(acc)
}

fn eval_and<S: ::std::hash::BuildHasher>(
    input: &str,
    flags: &HashMap<&str, &str, S>,
) -> std::result::Result<bool, String> {
    let mut acc = true;
    for clause in input.split("&&") {
        let v = eval_atom(clause.trim(), flags)?;
        acc = acc && v;
    }
    Ok(acc)
}

fn eval_atom<S: ::std::hash::BuildHasher>(
    input: &str,
    flags: &HashMap<&str, &str, S>,
) -> std::result::Result<bool, String> {
    if let Some((lhs, rhs)) = input.split_once("!=") {
        let l = flags.get(lhs.trim()).copied().unwrap_or("");
        let r = unquote(rhs.trim());
        return Ok(l != r);
    }
    if let Some((lhs, rhs)) = input.split_once("==") {
        let l = flags.get(lhs.trim()).copied().unwrap_or("");
        let r = unquote(rhs.trim());
        return Ok(l == r);
    }
    Err(format!("no comparator in '{input}'"))
}

fn unquote(s: &str) -> &str {
    s.trim_matches(|c| c == '"' || c == '\'')
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_pipeline_stage_skip_predicate")?;

    let mut flags = HashMap::new();
    flags.insert("branch", "main");
    flags.insert("env", "staging");
    let cases = [
        "branch == main",
        "env == prod",
        "branch == main && env == prod",
        "branch == main || env == staging",
        "no comparator",
    ];
    for p in cases {
        println!("{p:<40}  →  {:?}", evaluate(p, &flags));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flags() -> HashMap<&'static str, &'static str> {
        let mut f = HashMap::new();
        f.insert("branch", "main");
        f.insert("env", "staging");
        f
    }

    #[test]
    fn evaluator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn equal_match_returns_skip() {
        assert_eq!(evaluate("branch == main", &flags()), EvalVerdict::Skip);
    }

    #[test]
    fn equal_mismatch_returns_run() {
        assert_eq!(evaluate("branch == feature", &flags()), EvalVerdict::Run);
    }

    #[test]
    fn neq_returns_skip_when_different() {
        assert_eq!(evaluate("env != prod", &flags()), EvalVerdict::Skip);
    }

    #[test]
    fn and_combines_clauses() {
        // Both clauses true → skip.
        assert_eq!(
            evaluate("branch == main && env == staging", &flags()),
            EvalVerdict::Skip
        );
        // One false → run.
        assert_eq!(
            evaluate("branch == main && env == prod", &flags()),
            EvalVerdict::Run
        );
    }

    #[test]
    fn or_short_circuits_on_first_true() {
        assert_eq!(
            evaluate("branch == main || env == prod", &flags()),
            EvalVerdict::Skip
        );
    }

    #[test]
    fn or_returns_run_when_all_false() {
        assert_eq!(
            evaluate("branch == feature || env == prod", &flags()),
            EvalVerdict::Run
        );
    }

    #[test]
    fn missing_flag_treated_as_empty() {
        // unknown flag compared to nonempty value → Run.
        assert_eq!(evaluate("missing == foo", &flags()), EvalVerdict::Run);
        // unknown flag != nonempty → Skip.
        assert_eq!(evaluate("missing != foo", &flags()), EvalVerdict::Skip);
    }

    #[test]
    fn empty_predicate_invalid() {
        let v = evaluate("", &flags());
        assert!(matches!(v, EvalVerdict::InvalidPredicate { .. }));
    }

    #[test]
    fn no_comparator_invalid() {
        let v = evaluate("just_a_flag", &flags());
        assert!(matches!(v, EvalVerdict::InvalidPredicate { .. }));
    }

    #[test]
    fn quoted_values_trimmed() {
        assert_eq!(evaluate(r#"branch == "main""#, &flags()), EvalVerdict::Skip);
        assert_eq!(evaluate("branch == 'main'", &flags()), EvalVerdict::Skip);
    }
}
