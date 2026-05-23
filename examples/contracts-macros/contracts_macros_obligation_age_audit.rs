//! # Contracts-Macros Obligation Age Audit
//!
//! Audit obligation freshness — flag any obligation older than the
//! given staleness threshold (in days). Returns sorted stale IDs and
//! the maximum age observed.
//!
//! Demonstrates the **CMM.136** recipe for PMAT-203 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ITIL release management — staleness gates; OWASP CRS rule
//!  age tracking.
//!
//! Run with: cargo run --example contracts_macros_obligation_age_audit
//!
//! Added by PMAT-203 (catalog 1450→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AgeAuditVerdict {
    Ok {
        stale_ids: Vec<String>,
        max_age_days: u32,
    },
    InvalidConfig,
}

pub fn audit(obligations: &[(&str, u32)], staleness_days: u32) -> AgeAuditVerdict {
    if obligations.is_empty() || staleness_days == 0 {
        return AgeAuditVerdict::InvalidConfig;
    }
    let mut stale: Vec<String> = obligations
        .iter()
        .filter(|(_, age)| *age >= staleness_days)
        .map(|(id, _)| (*id).to_string())
        .collect();
    stale.sort();
    let max_age = obligations.iter().map(|(_, a)| *a).max().unwrap_or(0);
    AgeAuditVerdict::Ok {
        stale_ids: stale,
        max_age_days: max_age,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_age_audit")?;

    let obligations = [("o1", 30), ("o2", 90), ("o3", 200)];
    println!("threshold 60: {:?}", audit(&obligations, 60));
    println!("invalid: {:?}", audit(&[], 60));
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
    fn fresh_obligation_not_flagged() {
        let v = audit(&[("o", 10)], 60);
        if let AgeAuditVerdict::Ok { stale_ids, .. } = v {
            assert!(stale_ids.is_empty());
        }
    }

    #[test]
    fn stale_obligation_flagged() {
        let v = audit(&[("o", 100)], 60);
        if let AgeAuditVerdict::Ok { stale_ids, .. } = v {
            assert_eq!(stale_ids, vec!["o".to_string()]);
        }
    }

    #[test]
    fn max_age_correct() {
        let v = audit(&[("a", 30), ("b", 90), ("c", 200)], 60);
        if let AgeAuditVerdict::Ok { max_age_days, .. } = v {
            assert_eq!(max_age_days, 200);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[], 60), AgeAuditVerdict::InvalidConfig);
    }

    #[test]
    fn zero_threshold_rejected() {
        assert_eq!(audit(&[("a", 10)], 0), AgeAuditVerdict::InvalidConfig);
    }

    #[test]
    fn boundary_age_inclusive() {
        let v = audit(&[("o", 60)], 60);
        if let AgeAuditVerdict::Ok { stale_ids, .. } = v {
            assert_eq!(stale_ids, vec!["o".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&[("o", 50)], 30);
        let r2 = audit(&[("o", 50)], 30);
        assert_eq!(r1, r2);
    }

    #[test]
    fn stale_ids_sorted() {
        let v = audit(&[("zeta", 100), ("alpha", 100)], 50);
        if let AgeAuditVerdict::Ok { stale_ids, .. } = v {
            assert_eq!(stale_ids, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn many_obligations_handled() {
        let obligations: Vec<(&str, u32)> = (0..30).map(|_| ("o", 100)).collect();
        let v = audit(&obligations, 50);
        if let AgeAuditVerdict::Ok { stale_ids, .. } = v {
            assert_eq!(stale_ids.len(), 30);
        }
    }

    #[test]
    fn no_stale_returns_empty() {
        let v = audit(&[("a", 10), ("b", 20)], 100);
        if let AgeAuditVerdict::Ok { stale_ids, .. } = v {
            assert!(stale_ids.is_empty());
        }
    }

    #[test]
    fn unicode_id_supported() {
        let v = audit(&[("café", 100)], 50);
        if let AgeAuditVerdict::Ok { stale_ids, .. } = v {
            assert_eq!(stale_ids, vec!["café".to_string()]);
        }
    }
}
