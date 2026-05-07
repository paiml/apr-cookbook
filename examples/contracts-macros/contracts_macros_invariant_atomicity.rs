//! # Contracts-Macros Invariant Atomicity
//!
//! Verify each invariant is atomic: ≤1 conjunction (no `&&` or `,`),
//! ≤1 quantifier, depth ≤2. Returns sorted compound (non-atomic) IDs.
//!
//! Demonstrates the **CMM.186** recipe for PMAT-219 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: BDD-step atomicity guidance; SAT-solver literal-form
//!  atomic-clause requirements.
//!
//! Run with: cargo run --example contracts_macros_invariant_atomicity
//!
//! Added by PMAT-219 (catalog 1594→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AtomicityVerdict {
    Ok {
        compound_ids: Vec<String>,
        atomic_count: u32,
    },
    InvalidConfig,
}

/// Items: (id, expression).
pub fn check(items: &[(&str, &str)]) -> AtomicityVerdict {
    if items.is_empty() {
        return AtomicityVerdict::InvalidConfig;
    }
    let mut compound: Vec<String> = Vec::new();
    let mut atomic = 0u32;
    for (id, expr) in items {
        let conjunctions = expr.matches("&&").count() + expr.matches(", ").count();
        let quantifiers = expr.matches("forall").count() + expr.matches("exists").count();
        let nesting = expr.matches('(').count();
        if conjunctions > 0 || quantifiers > 1 || nesting > 2 {
            compound.push((*id).to_string());
        } else {
            atomic += 1;
        }
    }
    compound.sort();
    AtomicityVerdict::Ok {
        compound_ids: compound,
        atomic_count: atomic,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_invariant_atomicity")?;

    let items = [
        ("a", "x > 0"),
        ("b", "x > 0 && y > 0"),
        ("c", "forall i, exists j, P(i, j)"),
    ];
    println!("check: {:?}", check(&items));
    println!("invalid: {:?}", check(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn atomic_no_offender() {
        let v = check(&[("a", "x > 0")]);
        if let AtomicityVerdict::Ok { compound_ids, .. } = v {
            assert!(compound_ids.is_empty());
        }
    }

    #[test]
    fn double_ampersand_compound() {
        let v = check(&[("a", "x > 0 && y > 0")]);
        if let AtomicityVerdict::Ok { compound_ids, .. } = v {
            assert_eq!(compound_ids, vec!["a".to_string()]);
        }
    }

    #[test]
    fn comma_separated_compound() {
        let v = check(&[("a", "x > 0, y > 0")]);
        if let AtomicityVerdict::Ok { compound_ids, .. } = v {
            assert_eq!(compound_ids, vec!["a".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[]), AtomicityVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = check(&[("a", "x > 0")]);
        let r2 = check(&[("a", "x > 0")]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn compound_sorted() {
        let v = check(&[("zeta", "x && y"), ("alpha", "x && y")]);
        if let AtomicityVerdict::Ok { compound_ids, .. } = v {
            assert_eq!(compound_ids, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn double_quantifier_compound() {
        let v = check(&[("a", "forall x, exists y, P")]);
        if let AtomicityVerdict::Ok { compound_ids, .. } = v {
            assert!(compound_ids.contains(&"a".to_string()));
        }
    }

    #[test]
    fn deep_nesting_compound() {
        let v = check(&[("a", "((((x))))")]);
        if let AtomicityVerdict::Ok { compound_ids, .. } = v {
            assert!(compound_ids.contains(&"a".to_string()));
        }
    }

    #[test]
    fn atomic_count_correct() {
        let v = check(&[("a", "x"), ("b", "x && y"), ("c", "z")]);
        if let AtomicityVerdict::Ok { atomic_count, .. } = v {
            assert_eq!(atomic_count, 2);
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, &str)> = (0..30).map(|_| ("a", "x && y")).collect();
        let v = check(&items);
        if let AtomicityVerdict::Ok { compound_ids, .. } = v {
            // No dedup; all 30 entries flagged.
            assert_eq!(compound_ids.len(), 30);
        }
    }

    #[test]
    fn unicode_id_supported() {
        let v = check(&[("café", "x && y")]);
        if let AtomicityVerdict::Ok { compound_ids, .. } = v {
            assert_eq!(compound_ids, vec!["café".to_string()]);
        }
    }

    #[test]
    fn shallow_nesting_atomic() {
        let v = check(&[("a", "f(x)")]);
        if let AtomicityVerdict::Ok { atomic_count, .. } = v {
            assert_eq!(atomic_count, 1);
        }
    }
}
