//! # Contracts-Macros Obligation Count Audit
//!
//! Audit per-equation obligation counts: count preconditions and
//! postconditions, flag equations below `min_required` (default 1
//! pre + 1 post). Returns counts per equation and a list of
//! under-specified equations.
//!
//! Demonstrates the **CMM.64** recipe for PMAT-179 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hoare logic — pre/post obligation completeness.
//!
//! Run with: cargo run --example contracts_macros_obligation_count_audit
//!
//! Added by PMAT-179 (catalog 1234→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub struct EqStats {
    pub equation: String,
    pub preconditions: u32,
    pub postconditions: u32,
}

#[derive(Debug, PartialEq)]
pub enum AuditVerdict {
    Ok {
        per_equation: Vec<EqStats>,
        under_specified: Vec<String>,
    },
    InvalidConfig,
}

pub fn audit(obligations: &[(&str, &str)], min_pre: u32, min_post: u32) -> AuditVerdict {
    if obligations.is_empty() {
        return AuditVerdict::InvalidConfig;
    }
    let mut counts: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    for (eq, kind) in obligations {
        let entry = counts.entry((*eq).to_string()).or_insert((0, 0));
        match *kind {
            "pre" => entry.0 += 1,
            "post" => entry.1 += 1,
            _ => return AuditVerdict::InvalidConfig,
        }
    }
    let mut per_equation: Vec<EqStats> = Vec::with_capacity(counts.len());
    let mut under_specified: Vec<String> = Vec::new();
    for (eq, (pre, post)) in counts {
        if pre < min_pre || post < min_post {
            under_specified.push(eq.clone());
        }
        per_equation.push(EqStats {
            equation: eq,
            preconditions: pre,
            postconditions: post,
        });
    }
    AuditVerdict::Ok {
        per_equation,
        under_specified,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_count_audit")?;

    let obligations = [
        ("eq1", "pre"),
        ("eq1", "pre"),
        ("eq1", "post"),
        ("eq2", "post"),
    ];
    println!("audit: {:?}", audit(&obligations, 1, 1));
    println!("strict: {:?}", audit(&obligations, 2, 2));
    println!("invalid: {:?}", audit(&[], 1, 1));
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
    fn fully_specified_passes() {
        let obligations = [("eq1", "pre"), ("eq1", "post")];
        let v = audit(&obligations, 1, 1);
        if let AuditVerdict::Ok {
            under_specified, ..
        } = v
        {
            assert!(under_specified.is_empty());
        }
    }

    #[test]
    fn missing_postcondition_flagged() {
        let obligations = [("eq1", "pre")];
        let v = audit(&obligations, 1, 1);
        if let AuditVerdict::Ok {
            under_specified, ..
        } = v
        {
            assert_eq!(under_specified, vec!["eq1".to_string()]);
        }
    }

    #[test]
    fn missing_precondition_flagged() {
        let obligations = [("eq1", "post")];
        let v = audit(&obligations, 1, 1);
        if let AuditVerdict::Ok {
            under_specified, ..
        } = v
        {
            assert_eq!(under_specified, vec!["eq1".to_string()]);
        }
    }

    #[test]
    fn invalid_kind_rejected() {
        let obligations = [("eq1", "weird")];
        assert_eq!(audit(&obligations, 1, 1), AuditVerdict::InvalidConfig);
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[], 1, 1), AuditVerdict::InvalidConfig);
    }

    #[test]
    fn counts_are_correct() {
        let obligations = [("eq1", "pre"), ("eq1", "pre"), ("eq1", "post")];
        let v = audit(&obligations, 1, 1);
        if let AuditVerdict::Ok { per_equation, .. } = v {
            assert_eq!(per_equation.len(), 1);
            assert_eq!(per_equation[0].preconditions, 2);
            assert_eq!(per_equation[0].postconditions, 1);
        }
    }

    #[test]
    fn equations_alphabetically_sorted() {
        let obligations = [
            ("zeta", "pre"),
            ("zeta", "post"),
            ("alpha", "pre"),
            ("alpha", "post"),
        ];
        let v = audit(&obligations, 1, 1);
        if let AuditVerdict::Ok { per_equation, .. } = v {
            assert_eq!(per_equation[0].equation, "alpha");
            assert_eq!(per_equation[1].equation, "zeta");
        }
    }

    #[test]
    fn strict_thresholds_flag_more() {
        let obligations = [("eq1", "pre"), ("eq1", "post")];
        let v = audit(&obligations, 2, 2);
        if let AuditVerdict::Ok {
            under_specified, ..
        } = v
        {
            assert_eq!(under_specified, vec!["eq1".to_string()]);
        }
    }

    #[test]
    fn multiple_under_specified_collected() {
        let obligations = [("a", "pre"), ("b", "post"), ("c", "pre"), ("c", "post")];
        let v = audit(&obligations, 1, 1);
        if let AuditVerdict::Ok {
            under_specified, ..
        } = v
        {
            assert_eq!(under_specified.len(), 2);
            assert!(under_specified.contains(&"a".to_string()));
            assert!(under_specified.contains(&"b".to_string()));
        }
    }

    #[test]
    fn deterministic() {
        let obligations = [("e1", "pre"), ("e1", "post")];
        let a = audit(&obligations, 1, 1);
        let b = audit(&obligations, 1, 1);
        assert_eq!(a, b);
    }

    #[test]
    fn zero_thresholds_passes_all() {
        let obligations = [("e1", "pre")];
        let v = audit(&obligations, 0, 0);
        if let AuditVerdict::Ok {
            under_specified, ..
        } = v
        {
            assert!(under_specified.is_empty());
        }
    }
}
