//! # Contracts-Macros Invariant Negated Classify
//!
//! Classify invariants as Positive (e.g., `x > 0`) vs Negated (e.g.,
//! `not x`, `!P`, `x != 0`). Returns counts and sorted lists of each
//! kind.
//!
//! Demonstrates the **CMM.198** recipe for PMAT-223 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: BDD positive-vs-negative test framing; first-order
//!  logic literal-form classification.
//!
//! Run with: cargo run --example contracts_macros_invariant_negated_classify
//!
//! Added by PMAT-223 (catalog 1630→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum NegateVerdict {
    Ok {
        positive_ids: Vec<String>,
        negated_ids: Vec<String>,
    },
    InvalidConfig,
}

pub fn classify(items: &[(&str, &str)]) -> NegateVerdict {
    if items.is_empty() {
        return NegateVerdict::InvalidConfig;
    }
    let mut positive: Vec<String> = Vec::new();
    let mut negated: Vec<String> = Vec::new();
    for (id, expr) in items {
        if is_negated(expr) {
            negated.push((*id).to_string());
        } else {
            positive.push((*id).to_string());
        }
    }
    positive.sort();
    negated.sort();
    NegateVerdict::Ok {
        positive_ids: positive,
        negated_ids: negated,
    }
}

fn is_negated(expr: &str) -> bool {
    let trimmed = expr.trim();
    trimmed.starts_with("not ") || trimmed.starts_with('!') || trimmed.contains("!=")
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_invariant_negated_classify")?;

    let items = [
        ("a", "x > 0"),
        ("b", "not x"),
        ("c", "x != 0"),
        ("d", "P(x)"),
    ];
    println!("classify: {:?}", classify(&items));
    println!("invalid: {:?}", classify(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn positive_classified() {
        let v = classify(&[("a", "x > 0")]);
        if let NegateVerdict::Ok { positive_ids, .. } = v {
            assert_eq!(positive_ids, vec!["a".to_string()]);
        }
    }

    #[test]
    fn not_keyword_negated() {
        let v = classify(&[("a", "not x")]);
        if let NegateVerdict::Ok { negated_ids, .. } = v {
            assert_eq!(negated_ids, vec!["a".to_string()]);
        }
    }

    #[test]
    fn bang_prefix_negated() {
        let v = classify(&[("a", "!P")]);
        if let NegateVerdict::Ok { negated_ids, .. } = v {
            assert_eq!(negated_ids, vec!["a".to_string()]);
        }
    }

    #[test]
    fn neq_negated() {
        let v = classify(&[("a", "x != 0")]);
        if let NegateVerdict::Ok { negated_ids, .. } = v {
            assert_eq!(negated_ids, vec!["a".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(classify(&[]), NegateVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = classify(&[("a", "x > 0")]);
        let r2 = classify(&[("a", "x > 0")]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn lists_sorted() {
        let v = classify(&[("zeta", "x"), ("alpha", "x")]);
        if let NegateVerdict::Ok { positive_ids, .. } = v {
            assert_eq!(positive_ids, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn whitespace_handled() {
        let v = classify(&[("a", "  not x ")]);
        if let NegateVerdict::Ok { negated_ids, .. } = v {
            assert_eq!(negated_ids, vec!["a".to_string()]);
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, &str)> = (0..30).map(|_| ("a", "x > 0")).collect();
        let v = classify(&items);
        if let NegateVerdict::Ok { positive_ids, .. } = v {
            assert_eq!(positive_ids.len(), 30);
        }
    }

    #[test]
    fn unicode_id_supported() {
        let v = classify(&[("café", "x > 0")]);
        if let NegateVerdict::Ok { positive_ids, .. } = v {
            assert_eq!(positive_ids, vec!["café".to_string()]);
        }
    }

    #[test]
    fn mixed_classification() {
        let v = classify(&[("a", "x > 0"), ("b", "not y"), ("c", "z == 1")]);
        if let NegateVerdict::Ok {
            positive_ids,
            negated_ids,
        } = v
        {
            assert_eq!(positive_ids.len(), 2);
            assert_eq!(negated_ids.len(), 1);
        }
    }

    #[test]
    fn equality_is_positive() {
        let v = classify(&[("a", "x == 0")]);
        if let NegateVerdict::Ok { positive_ids, .. } = v {
            assert_eq!(positive_ids, vec!["a".to_string()]);
        }
    }
}
