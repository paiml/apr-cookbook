//! # Contracts-Macros Obligation Test Traceability
//!
//! Verify each obligation references at least `min_tests` named test
//! cases. Returns offenders (no tests / under-tested) plus traceable
//! count.
//!
//! Demonstrates the **CMM.94** recipe for PMAT-189 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: V-Model verification matrix; ISO/IEC/IEEE 29119-2 §7
//!  test-traceability conventions.
//!
//! Run with: cargo run --example contracts_macros_obligation_test_traceability
//!
//! Added by PMAT-189 (catalog 1324→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TraceVerdict {
    Ok {
        no_tests: Vec<String>,
        under_tested: Vec<String>,
        traceable_count: u32,
    },
    InvalidConfig,
}

pub fn audit(obligations: &[(&str, u32)], min_tests: u32) -> TraceVerdict {
    if obligations.is_empty() || min_tests == 0 {
        return TraceVerdict::InvalidConfig;
    }
    let mut no_tests: Vec<String> = Vec::new();
    let mut under_tested: Vec<String> = Vec::new();
    let mut traceable = 0u32;
    for (id, count) in obligations {
        if *count == 0 {
            no_tests.push((*id).to_string());
        } else if *count < min_tests {
            under_tested.push((*id).to_string());
        } else {
            traceable += 1;
        }
    }
    no_tests.sort();
    under_tested.sort();
    TraceVerdict::Ok {
        no_tests,
        under_tested,
        traceable_count: traceable,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_test_traceability")?;

    let obligations = [("o1", 5), ("o2", 0), ("o3", 1), ("o4", 10)];
    println!("audit: {:?}", audit(&obligations, 3));
    println!("invalid: {:?}", audit(&[], 3));
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
    fn well_tested_no_offenders() {
        let obligations = [("o1", 5), ("o2", 5)];
        let v = audit(&obligations, 3);
        if let TraceVerdict::Ok {
            no_tests,
            under_tested,
            ..
        } = v
        {
            assert!(no_tests.is_empty());
            assert!(under_tested.is_empty());
        }
    }

    #[test]
    fn no_tests_flagged() {
        let obligations = [("o1", 0)];
        let v = audit(&obligations, 3);
        if let TraceVerdict::Ok { no_tests, .. } = v {
            assert_eq!(no_tests, vec!["o1".to_string()]);
        }
    }

    #[test]
    fn under_tested_flagged() {
        let obligations = [("o1", 1)];
        let v = audit(&obligations, 3);
        if let TraceVerdict::Ok { under_tested, .. } = v {
            assert_eq!(under_tested, vec!["o1".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[], 3), TraceVerdict::InvalidConfig);
    }

    #[test]
    fn zero_min_rejected() {
        let obligations = [("o", 5)];
        assert_eq!(audit(&obligations, 0), TraceVerdict::InvalidConfig);
    }

    #[test]
    fn boundary_min_passes() {
        let obligations = [("o", 3)];
        let v = audit(&obligations, 3);
        if let TraceVerdict::Ok {
            traceable_count, ..
        } = v
        {
            assert_eq!(traceable_count, 1);
        }
    }

    #[test]
    fn one_below_min_under_tested() {
        let obligations = [("o", 2)];
        let v = audit(&obligations, 3);
        if let TraceVerdict::Ok { under_tested, .. } = v {
            assert_eq!(under_tested, vec!["o".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let obligations = [("o", 3)];
        let r1 = audit(&obligations, 3);
        let r2 = audit(&obligations, 3);
        assert_eq!(r1, r2);
    }

    #[test]
    fn no_tests_sorted() {
        let obligations = [("zeta", 0), ("alpha", 0)];
        let v = audit(&obligations, 1);
        if let TraceVerdict::Ok { no_tests, .. } = v {
            assert_eq!(no_tests, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn under_tested_sorted() {
        let obligations = [("zeta", 1), ("alpha", 1)];
        let v = audit(&obligations, 5);
        if let TraceVerdict::Ok { under_tested, .. } = v {
            assert_eq!(under_tested, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn three_categories_present() {
        let obligations = [("good", 5), ("zero", 0), ("low", 1)];
        let v = audit(&obligations, 3);
        if let TraceVerdict::Ok {
            no_tests,
            under_tested,
            traceable_count,
        } = v
        {
            assert!(no_tests.contains(&"zero".to_string()));
            assert!(under_tested.contains(&"low".to_string()));
            assert_eq!(traceable_count, 1);
        }
    }
}
