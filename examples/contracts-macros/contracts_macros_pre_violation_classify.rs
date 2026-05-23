//! # Contracts-Macros Precondition Violation Classifier
//!
//! When a precondition fails at runtime, classify the failure: range
//! (value out of bounds), enum (unexpected variant), shape (size
//! mismatch), or domain (semantic constraint).
//!
//! Demonstrates the **CMM.13** recipe for PMAT-162 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Eiffel-style preconditions + LiquidHaskell categorization.
//!
//! Run with: cargo run --example contracts_macros_pre_violation_classify
//!
//! Added by PMAT-162 (catalog 1081→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationKind {
    Range,
    Enum,
    Shape,
    Domain,
    Unknown,
}

#[derive(Debug, PartialEq)]
pub enum ClassifyVerdict {
    Ok { kind: ViolationKind, key: String },
    EmptyMessage,
}

pub fn classify(failed_precond: &str) -> ClassifyVerdict {
    let msg = failed_precond.trim();
    if msg.is_empty() {
        return ClassifyVerdict::EmptyMessage;
    }
    let lower = msg.to_ascii_lowercase();
    let kind = if lower.contains("must be in")
        || lower.contains("between")
        || lower.contains("≥")
        || lower.contains("≤")
        || lower.contains(">=")
        || lower.contains("<=")
    {
        ViolationKind::Range
    } else if lower.contains("expected variant")
        || lower.contains("invalid enum")
        || lower.contains("unknown variant")
    {
        ViolationKind::Enum
    } else if lower.contains("shape mismatch")
        || lower.contains("dimension")
        || lower.contains("len(")
        || lower.contains("size")
    {
        ViolationKind::Shape
    } else if lower.contains("domain")
        || lower.contains("must satisfy")
        || lower.contains("constraint")
    {
        ViolationKind::Domain
    } else {
        ViolationKind::Unknown
    };
    let key = msg.split([':', '.']).next().unwrap_or(msg).to_string();
    ClassifyVerdict::Ok { kind, key }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_pre_violation_classify")?;

    println!("range: {:?}", classify("x must be in [0, 10]"));
    println!("enum: {:?}", classify("expected variant Foo"));
    println!("shape: {:?}", classify("shape mismatch"));
    println!("domain: {:?}", classify("must satisfy domain constraint"));
    println!("unknown: {:?}", classify("oops"));
    println!("empty: {:?}", classify("  "));
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
    fn range_phrase_classified() {
        let v = classify("x must be in [0, 10]");
        if let ClassifyVerdict::Ok { kind, .. } = v {
            assert_eq!(kind, ViolationKind::Range);
        }
    }

    #[test]
    fn between_phrase_range() {
        let v = classify("value between 1 and 100");
        if let ClassifyVerdict::Ok { kind, .. } = v {
            assert_eq!(kind, ViolationKind::Range);
        }
    }

    #[test]
    fn enum_phrase() {
        let v = classify("expected variant Foo");
        if let ClassifyVerdict::Ok { kind, .. } = v {
            assert_eq!(kind, ViolationKind::Enum);
        }
    }

    #[test]
    fn shape_phrase() {
        let v = classify("shape mismatch in tensor");
        if let ClassifyVerdict::Ok { kind, .. } = v {
            assert_eq!(kind, ViolationKind::Shape);
        }
    }

    #[test]
    fn domain_phrase() {
        let v = classify("input must satisfy non-negative");
        if let ClassifyVerdict::Ok { kind, .. } = v {
            assert_eq!(kind, ViolationKind::Domain);
        }
    }

    #[test]
    fn unknown_falls_through() {
        let v = classify("something went wrong");
        if let ClassifyVerdict::Ok { kind, .. } = v {
            assert_eq!(kind, ViolationKind::Unknown);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(classify("  "), ClassifyVerdict::EmptyMessage);
    }

    #[test]
    fn key_extracted_before_colon() {
        let v = classify("invalid enum: bad value");
        if let ClassifyVerdict::Ok { key, kind } = v {
            assert_eq!(key, "invalid enum");
            assert_eq!(kind, ViolationKind::Enum);
        }
    }

    #[test]
    fn case_insensitive() {
        let v = classify("MUST BE IN [0,1]");
        if let ClassifyVerdict::Ok { kind, .. } = v {
            assert_eq!(kind, ViolationKind::Range);
        }
    }

    #[test]
    fn deterministic() {
        let a = classify("shape mismatch");
        let b = classify("shape mismatch");
        assert_eq!(a, b);
    }
}
