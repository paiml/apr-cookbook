//! # Contracts-Macros Obligation Type Consistency
//!
//! Verify the same obligation id has consistent type across all
//! references. Returns conflicting ids and consistency rate.
//!
//! Demonstrates the **CMM.106** recipe for PMAT-193 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: type-checker disambiguation rules; SQL constraint
//!  consistency (Date, SQL and Relational Theory ch.7).
//!
//! Run with: cargo run --example contracts_macros_obligation_type_consistency
//!
//! Added by PMAT-193 (catalog 1360→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum ConsistencyVerdict {
    Ok {
        conflicts: Vec<String>,
        consistency_rate: f64,
    },
    InvalidConfig,
}

pub fn audit(refs: &[(&str, &str)]) -> ConsistencyVerdict {
    if refs.is_empty() {
        return ConsistencyVerdict::InvalidConfig;
    }
    let mut id_types: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (id, ty) in refs {
        id_types
            .entry((*id).to_string())
            .or_default()
            .insert((*ty).to_string());
    }
    let mut conflicts: Vec<String> = id_types
        .iter()
        .filter(|(_, types)| types.len() > 1)
        .map(|(id, _)| id.clone())
        .collect();
    conflicts.sort();
    let consistency_rate = if id_types.is_empty() {
        1.0
    } else {
        let consistent = id_types.len() - conflicts.len();
        consistent as f64 / id_types.len() as f64
    };
    ConsistencyVerdict::Ok {
        conflicts,
        consistency_rate,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_type_consistency")?;

    let refs = [("o1", "i32"), ("o2", "f64"), ("o1", "i32")];
    println!("consistent: {:?}", audit(&refs));
    let bad = [("o1", "i32"), ("o1", "u32")];
    println!("conflict: {:?}", audit(&bad));
    println!("invalid: {:?}", audit(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auditor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn consistent_no_conflicts() {
        let refs = [("o1", "i32"), ("o1", "i32")];
        let v = audit(&refs);
        if let ConsistencyVerdict::Ok { conflicts, .. } = v {
            assert!(conflicts.is_empty());
        }
    }

    #[test]
    fn type_conflict_flagged() {
        let refs = [("o1", "i32"), ("o1", "u32")];
        let v = audit(&refs);
        if let ConsistencyVerdict::Ok { conflicts, .. } = v {
            assert_eq!(conflicts, vec!["o1".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), ConsistencyVerdict::InvalidConfig);
    }

    #[test]
    fn distinct_ids_independent() {
        let refs = [("o1", "i32"), ("o2", "u32")];
        let v = audit(&refs);
        if let ConsistencyVerdict::Ok { conflicts, .. } = v {
            assert!(conflicts.is_empty());
        }
    }

    #[test]
    fn consistency_rate_one_when_all_ok() {
        let refs = [("o1", "i32")];
        let v = audit(&refs);
        if let ConsistencyVerdict::Ok {
            consistency_rate, ..
        } = v
        {
            assert!((consistency_rate - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn consistency_rate_zero_all_conflicts() {
        let refs = [("o", "a"), ("o", "b")];
        let v = audit(&refs);
        if let ConsistencyVerdict::Ok {
            consistency_rate, ..
        } = v
        {
            assert_eq!(consistency_rate, 0.0);
        }
    }

    #[test]
    fn deterministic() {
        let refs = [("o", "a")];
        let r1 = audit(&refs);
        let r2 = audit(&refs);
        assert_eq!(r1, r2);
    }

    #[test]
    fn conflicts_sorted() {
        let refs = [("zeta", "a"), ("zeta", "b"), ("alpha", "x"), ("alpha", "y")];
        let v = audit(&refs);
        if let ConsistencyVerdict::Ok { conflicts, .. } = v {
            assert_eq!(conflicts, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn rate_in_unit_range() {
        let refs = [("o", "a"), ("o", "b")];
        let v = audit(&refs);
        if let ConsistencyVerdict::Ok {
            consistency_rate, ..
        } = v
        {
            assert!((0.0..=1.0).contains(&consistency_rate));
        }
    }

    #[test]
    fn three_types_same_id_one_conflict() {
        let refs = [("o", "a"), ("o", "b"), ("o", "c")];
        let v = audit(&refs);
        if let ConsistencyVerdict::Ok { conflicts, .. } = v {
            assert_eq!(conflicts, vec!["o".to_string()]);
        }
    }

    #[test]
    fn case_sensitive_types() {
        let refs = [("o", "i32"), ("o", "I32")];
        let v = audit(&refs);
        if let ConsistencyVerdict::Ok { conflicts, .. } = v {
            assert_eq!(conflicts.len(), 1);
        }
    }
}
