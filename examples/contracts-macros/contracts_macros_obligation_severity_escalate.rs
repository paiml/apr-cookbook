//! # Contracts-Macros Obligation Severity Escalate
//!
//! Bump severity when obligation unresolved beyond `escalate_after_days`.
//! Returns escalated obligations + new severity per id.
//!
//! Demonstrates the **CMM.127** recipe for PMAT-200 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PagerDuty escalation policy; ITIL incident escalation
//!  matrix.
//!
//! Run with: cargo run --example contracts_macros_obligation_severity_escalate
//!
//! Added by PMAT-200 (catalog 1423→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy, PartialOrd, Ord, Eq)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, PartialEq)]
pub enum EscalateVerdict {
    Ok {
        escalated: Vec<String>,
        per_obligation: Vec<(String, Severity)>,
    },
    InvalidConfig,
}

pub fn escalate(
    obligations: &[(&str, Severity, u32)], // (id, severity, age_days)
    escalate_after_days: u32,
) -> EscalateVerdict {
    if obligations.is_empty() || escalate_after_days == 0 {
        return EscalateVerdict::InvalidConfig;
    }
    let mut escalated: Vec<String> = Vec::new();
    let mut per_obligation: Vec<(String, Severity)> = Vec::with_capacity(obligations.len());
    for (id, sev, age) in obligations {
        let new_sev = if *age > escalate_after_days {
            let bumped = match sev {
                Severity::Low => Severity::Medium,
                Severity::Medium => Severity::High,
                Severity::High => Severity::Critical,
                Severity::Critical => Severity::Critical,
            };
            if bumped != *sev {
                escalated.push((*id).to_string());
            }
            bumped
        } else {
            *sev
        };
        per_obligation.push(((*id).to_string(), new_sev));
    }
    escalated.sort();
    EscalateVerdict::Ok {
        escalated,
        per_obligation,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_severity_escalate")?;

    let obligations = [
        ("o1", Severity::Low, 30),
        ("o2", Severity::Medium, 5),
        ("o3", Severity::High, 60),
    ];
    println!("audit: {:?}", escalate(&obligations, 14));
    println!("invalid: {:?}", escalate(&[], 14));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escalator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fresh_no_escalation() {
        let obs = [("o", Severity::Low, 5)];
        let v = escalate(&obs, 14);
        if let EscalateVerdict::Ok { escalated, .. } = v {
            assert!(escalated.is_empty());
        }
    }

    #[test]
    fn old_low_to_medium() {
        let obs = [("o", Severity::Low, 30)];
        let v = escalate(&obs, 14);
        if let EscalateVerdict::Ok {
            escalated,
            per_obligation,
        } = v
        {
            assert_eq!(escalated, vec!["o".to_string()]);
            assert_eq!(per_obligation[0].1, Severity::Medium);
        }
    }

    #[test]
    fn critical_does_not_bump() {
        let obs = [("o", Severity::Critical, 100)];
        let v = escalate(&obs, 14);
        if let EscalateVerdict::Ok { escalated, .. } = v {
            assert!(escalated.is_empty());
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(escalate(&[], 14), EscalateVerdict::InvalidConfig);
    }

    #[test]
    fn zero_threshold_rejected() {
        let obs = [("o", Severity::Low, 5)];
        assert_eq!(escalate(&obs, 0), EscalateVerdict::InvalidConfig);
    }

    #[test]
    fn boundary_at_threshold_no_escalate() {
        let obs = [("o", Severity::Low, 14)];
        let v = escalate(&obs, 14);
        if let EscalateVerdict::Ok { escalated, .. } = v {
            assert!(escalated.is_empty());
        }
    }

    #[test]
    fn one_over_threshold_escalates() {
        let obs = [("o", Severity::Low, 15)];
        let v = escalate(&obs, 14);
        if let EscalateVerdict::Ok { escalated, .. } = v {
            assert_eq!(escalated, vec!["o".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let obs = [("o", Severity::Low, 30)];
        let r1 = escalate(&obs, 14);
        let r2 = escalate(&obs, 14);
        assert_eq!(r1, r2);
    }

    #[test]
    fn escalated_sorted() {
        let obs = [("zeta", Severity::Low, 30), ("alpha", Severity::Low, 30)];
        let v = escalate(&obs, 14);
        if let EscalateVerdict::Ok { escalated, .. } = v {
            assert_eq!(escalated, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn severity_ordering() {
        assert!(Severity::Critical > Severity::High);
        assert!(Severity::High > Severity::Medium);
        assert!(Severity::Medium > Severity::Low);
    }

    #[test]
    fn high_to_critical() {
        let obs = [("o", Severity::High, 30)];
        let v = escalate(&obs, 14);
        if let EscalateVerdict::Ok { per_obligation, .. } = v {
            assert_eq!(per_obligation[0].1, Severity::Critical);
        }
    }
}
