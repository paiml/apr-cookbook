//! # Contracts-Macros Phase Dependency Check
//!
//! Verify each phase's `depends_on` references occur earlier in the
//! pipeline (no forward references, no self-loops). Returns
//! offending phases plus their bad refs.
//!
//! Demonstrates the **CMM.79** recipe for PMAT-184 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Make/Bazel BUILD-graph dependency-order rules; topo-sort
//!  invariant in build systems.
//!
//! Run with: cargo run --example contracts_macros_phase_dependency_check
//!
//! Added by PMAT-184 (catalog 1279→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub struct Violation {
    pub phase: String,
    pub bad_refs: Vec<String>,
}

#[derive(Debug, PartialEq)]
pub enum DependencyVerdict {
    Ok { violations: Vec<Violation> },
    InvalidConfig,
}

pub fn audit(phases: &[(&str, Vec<&str>)]) -> DependencyVerdict {
    if phases.is_empty() {
        return DependencyVerdict::InvalidConfig;
    }
    let mut order: BTreeMap<String, usize> = BTreeMap::new();
    for (i, (name, _)) in phases.iter().enumerate() {
        order.insert((*name).to_string(), i);
    }
    let mut violations: Vec<Violation> = Vec::new();
    for (i, (name, deps)) in phases.iter().enumerate() {
        let mut bad: Vec<String> = Vec::new();
        for d in deps {
            if d == name {
                bad.push((*d).to_string());
                continue;
            }
            match order.get(*d) {
                Some(&j) if j < i => {} // ok
                _ => bad.push((*d).to_string()),
            }
        }
        if !bad.is_empty() {
            bad.sort();
            violations.push(Violation {
                phase: (*name).to_string(),
                bad_refs: bad,
            });
        }
    }
    DependencyVerdict::Ok { violations }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_phase_dependency_check")?;

    let clean = vec![
        ("init", vec![]),
        ("validate", vec!["init"]),
        ("emit", vec!["validate", "init"]),
    ];
    println!("clean: {:?}", audit(&clean));
    let bad = vec![
        ("init", vec!["emit"]), // forward ref
        ("emit", vec!["init"]),
    ];
    println!("bad: {:?}", audit(&bad));
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
    fn clean_pipeline_no_violations() {
        let phases = vec![("a", vec![]), ("b", vec!["a"]), ("c", vec!["a", "b"])];
        let v = audit(&phases);
        if let DependencyVerdict::Ok { violations } = v {
            assert!(violations.is_empty());
        }
    }

    #[test]
    fn forward_reference_flagged() {
        let phases = vec![("a", vec!["b"]), ("b", vec![])];
        let v = audit(&phases);
        if let DependencyVerdict::Ok { violations } = v {
            assert_eq!(violations.len(), 1);
            assert_eq!(violations[0].phase, "a");
            assert_eq!(violations[0].bad_refs, vec!["b".to_string()]);
        }
    }

    #[test]
    fn self_reference_flagged() {
        let phases = vec![("a", vec!["a"])];
        let v = audit(&phases);
        if let DependencyVerdict::Ok { violations } = v {
            assert_eq!(violations.len(), 1);
            assert_eq!(violations[0].bad_refs, vec!["a".to_string()]);
        }
    }

    #[test]
    fn unknown_dep_flagged() {
        let phases = vec![("a", vec!["ghost"])];
        let v = audit(&phases);
        if let DependencyVerdict::Ok { violations } = v {
            assert_eq!(violations[0].bad_refs, vec!["ghost".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), DependencyVerdict::InvalidConfig);
    }

    #[test]
    fn no_deps_ok() {
        let phases = vec![("a", vec![]), ("b", vec![])];
        let v = audit(&phases);
        if let DependencyVerdict::Ok { violations } = v {
            assert!(violations.is_empty());
        }
    }

    #[test]
    fn bad_refs_sorted() {
        let phases = vec![
            ("first", vec![]),
            ("second", vec!["zeta", "alpha", "first"]),
        ];
        let v = audit(&phases);
        if let DependencyVerdict::Ok { violations } = v {
            assert_eq!(
                violations[0].bad_refs,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn deterministic() {
        let phases = vec![("a", vec![]), ("b", vec!["a"])];
        let r1 = audit(&phases);
        let r2 = audit(&phases);
        assert_eq!(r1, r2);
    }

    #[test]
    fn multi_phase_violations_collected() {
        let phases = vec![
            ("a", vec!["c"]), // forward
            ("b", vec!["d"]), // forward
            ("c", vec![]),
            ("d", vec![]),
        ];
        let v = audit(&phases);
        if let DependencyVerdict::Ok { violations } = v {
            assert_eq!(violations.len(), 2);
        }
    }

    #[test]
    fn all_three_violations_in_one_phase() {
        let phases = vec![("a", vec!["a", "ghost", "future"]), ("future", vec![])];
        let v = audit(&phases);
        if let DependencyVerdict::Ok { violations } = v {
            assert_eq!(violations[0].bad_refs.len(), 3);
        }
    }

    #[test]
    fn duplicate_bad_ref_kept() {
        let phases = vec![("a", vec!["a", "a"])];
        let v = audit(&phases);
        if let DependencyVerdict::Ok { violations } = v {
            // Both self-refs reported; we don't dedupe.
            assert_eq!(violations[0].bad_refs.len(), 2);
        }
    }
}
