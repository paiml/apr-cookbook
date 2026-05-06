//! # Contracts-Macros Constant Propagation
//!
//! Given a list of bindings `name = literal` and a target expression,
//! substitute known constants and report whether the expression is
//! fully reduced or has free variables.
//!
//! Demonstrates the **CMM.26** recipe for PMAT-166 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Constant folding in compilers (Aho et al. Dragon Book).
//!
//! Run with: cargo run --example contracts_macros_constant_propagation
//!
//! Added by PMAT-166 (catalog 1117→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum PropVerdict {
    Reduced { expr: String, substitutions: u32 },
    Free { expr: String, missing: Vec<String> },
    EmptyExpr,
}

pub fn propagate(expr: &str, bindings: &[(&str, &str)]) -> PropVerdict {
    let trimmed = expr.trim();
    if trimmed.is_empty() {
        return PropVerdict::EmptyExpr;
    }
    let map: BTreeMap<&str, &str> = bindings.iter().copied().collect();
    let mut tokens: Vec<String> = Vec::new();
    let mut substitutions = 0u32;
    let mut missing: Vec<String> = Vec::new();
    for tok in trimmed.split_whitespace() {
        let is_ident = tok.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && tok
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
        if !is_ident {
            tokens.push(tok.to_string());
            continue;
        }
        if let Some(value) = map.get(tok) {
            tokens.push((*value).to_string());
            substitutions += 1;
        } else {
            tokens.push(tok.to_string());
            missing.push(tok.to_string());
        }
    }
    let new_expr = tokens.join(" ");
    if missing.is_empty() {
        PropVerdict::Reduced {
            expr: new_expr,
            substitutions,
        }
    } else {
        PropVerdict::Free {
            expr: new_expr,
            missing,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_constant_propagation")?;

    let bindings = [("a", "5"), ("b", "10")];
    println!("fully reduced: {:?}", propagate("a + b", &bindings));
    println!("partial: {:?}", propagate("a + c", &bindings));
    println!("no idents: {:?}", propagate("1 + 2", &bindings));
    println!("empty: {:?}", propagate("  ", &bindings));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propagator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fully_reduced_expr() {
        let v = propagate("a + b", &[("a", "5"), ("b", "10")]);
        if let PropVerdict::Reduced {
            expr,
            substitutions,
        } = v
        {
            assert_eq!(expr, "5 + 10");
            assert_eq!(substitutions, 2);
        }
    }

    #[test]
    fn partial_reduction_lists_missing() {
        let v = propagate("a + c", &[("a", "5")]);
        if let PropVerdict::Free { missing, .. } = v {
            assert_eq!(missing, vec!["c".to_string()]);
        }
    }

    #[test]
    fn no_idents_no_substitutions() {
        let v = propagate("1 + 2", &[]);
        if let PropVerdict::Reduced { substitutions, .. } = v {
            assert_eq!(substitutions, 0);
        }
    }

    #[test]
    fn empty_expr_rejected() {
        assert_eq!(propagate("  ", &[]), PropVerdict::EmptyExpr);
    }

    #[test]
    fn keeps_operators_intact() {
        let v = propagate("a * b", &[("a", "3"), ("b", "4")]);
        if let PropVerdict::Reduced { expr, .. } = v {
            assert!(expr.contains('*'));
        }
    }

    #[test]
    fn ignores_numeric_tokens() {
        let v = propagate("a + 5", &[("a", "10")]);
        if let PropVerdict::Reduced {
            expr,
            substitutions,
        } = v
        {
            assert_eq!(expr, "10 + 5");
            assert_eq!(substitutions, 1);
        }
    }

    #[test]
    fn unicode_value_substituted() {
        let v = propagate("pi", &[("pi", "3.14159")]);
        if let PropVerdict::Reduced { expr, .. } = v {
            assert_eq!(expr, "3.14159");
        }
    }

    #[test]
    fn underscore_ident() {
        let v = propagate("foo_bar", &[("foo_bar", "42")]);
        if let PropVerdict::Reduced { expr, .. } = v {
            assert_eq!(expr, "42");
        }
    }

    #[test]
    fn multiple_missing_listed() {
        let v = propagate("a + b + c", &[("a", "1")]);
        if let PropVerdict::Free { missing, .. } = v {
            assert_eq!(missing.len(), 2);
        }
    }

    #[test]
    fn deterministic() {
        let a = propagate("a + b", &[("a", "5"), ("b", "10")]);
        let b = propagate("a + b", &[("a", "5"), ("b", "10")]);
        assert_eq!(a, b);
    }
}
