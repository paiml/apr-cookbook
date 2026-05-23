//! # Contracts-Macros Obligation Breaking-Change Audit
//!
//! Flag obligation changes that break consumer contracts (semver-style):
//! tightening preconditions or weakening postconditions are breaking;
//! the inverse is safe.
//!
//! Demonstrates the **CMM.91** recipe for PMAT-188 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: semver.org §8 (breaking changes); Liskov substitution
//!  principle (Liskov & Wing, TOPLAS 1994).
//!
//! Run with: cargo run --example contracts_macros_obligation_breaking_change_audit
//!
//! Added by PMAT-188 (catalog 1315→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ChangeKind {
    PreconditionTightened,
    PreconditionWeakened,
    PostconditionWeakened,
    PostconditionStrengthened,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Severity {
    Breaking,
    Safe,
}

#[derive(Debug, PartialEq)]
pub enum AuditVerdict {
    Ok {
        per_change: Vec<(String, Severity)>,
        breaking_count: u32,
    },
    InvalidConfig,
}

pub fn classify(changes: &[(&str, ChangeKind)]) -> AuditVerdict {
    if changes.is_empty() {
        return AuditVerdict::InvalidConfig;
    }
    let mut per_change: Vec<(String, Severity)> = Vec::with_capacity(changes.len());
    let mut breaking_count = 0u32;
    for (id, kind) in changes {
        let sev = match kind {
            ChangeKind::PreconditionTightened | ChangeKind::PostconditionWeakened => {
                breaking_count += 1;
                Severity::Breaking
            }
            ChangeKind::PreconditionWeakened | ChangeKind::PostconditionStrengthened => {
                Severity::Safe
            }
        };
        per_change.push(((*id).to_string(), sev));
    }
    AuditVerdict::Ok {
        per_change,
        breaking_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_breaking_change_audit")?;

    let changes = [
        ("o1", ChangeKind::PreconditionTightened),
        ("o2", ChangeKind::PostconditionStrengthened),
        ("o3", ChangeKind::PostconditionWeakened),
        ("o4", ChangeKind::PreconditionWeakened),
    ];
    println!("audit: {:?}", classify(&changes));
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
    fn pre_tightened_is_breaking() {
        let changes = [("o", ChangeKind::PreconditionTightened)];
        let v = classify(&changes);
        if let AuditVerdict::Ok { breaking_count, .. } = v {
            assert_eq!(breaking_count, 1);
        }
    }

    #[test]
    fn pre_weakened_is_safe() {
        let changes = [("o", ChangeKind::PreconditionWeakened)];
        let v = classify(&changes);
        if let AuditVerdict::Ok { per_change, .. } = v {
            assert_eq!(per_change[0].1, Severity::Safe);
        }
    }

    #[test]
    fn post_weakened_is_breaking() {
        let changes = [("o", ChangeKind::PostconditionWeakened)];
        let v = classify(&changes);
        if let AuditVerdict::Ok { breaking_count, .. } = v {
            assert_eq!(breaking_count, 1);
        }
    }

    #[test]
    fn post_strengthened_is_safe() {
        let changes = [("o", ChangeKind::PostconditionStrengthened)];
        let v = classify(&changes);
        if let AuditVerdict::Ok { per_change, .. } = v {
            assert_eq!(per_change[0].1, Severity::Safe);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(classify(&[]), AuditVerdict::InvalidConfig);
    }

    #[test]
    fn mixed_changes_counted_correctly() {
        let changes = [
            ("a", ChangeKind::PreconditionTightened),
            ("b", ChangeKind::PreconditionWeakened),
            ("c", ChangeKind::PostconditionWeakened),
            ("d", ChangeKind::PostconditionStrengthened),
        ];
        let v = classify(&changes);
        if let AuditVerdict::Ok { breaking_count, .. } = v {
            assert_eq!(breaking_count, 2);
        }
    }

    #[test]
    fn count_matches_change_count() {
        let changes = [
            ("a", ChangeKind::PreconditionWeakened),
            ("b", ChangeKind::PreconditionTightened),
        ];
        let v = classify(&changes);
        if let AuditVerdict::Ok { per_change, .. } = v {
            assert_eq!(per_change.len(), 2);
        }
    }

    #[test]
    fn deterministic() {
        let changes = [("o", ChangeKind::PreconditionTightened)];
        let r1 = classify(&changes);
        let r2 = classify(&changes);
        assert_eq!(r1, r2);
    }

    #[test]
    fn all_safe_zero_breaking() {
        let changes = [
            ("a", ChangeKind::PreconditionWeakened),
            ("b", ChangeKind::PostconditionStrengthened),
        ];
        let v = classify(&changes);
        if let AuditVerdict::Ok { breaking_count, .. } = v {
            assert_eq!(breaking_count, 0);
        }
    }

    #[test]
    fn all_breaking_count_matches() {
        let changes = [
            ("a", ChangeKind::PreconditionTightened),
            ("b", ChangeKind::PostconditionWeakened),
        ];
        let v = classify(&changes);
        if let AuditVerdict::Ok { breaking_count, .. } = v {
            assert_eq!(breaking_count, 2);
        }
    }

    #[test]
    fn order_preserved() {
        let changes = [
            ("first", ChangeKind::PreconditionTightened),
            ("second", ChangeKind::PreconditionWeakened),
        ];
        let v = classify(&changes);
        if let AuditVerdict::Ok { per_change, .. } = v {
            assert_eq!(per_change[0].0, "first");
            assert_eq!(per_change[1].0, "second");
        }
    }
}
