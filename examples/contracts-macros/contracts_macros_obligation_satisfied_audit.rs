//! # Contracts-Macros Obligation Satisfaction Audit
//!
//! Track obligation satisfaction states (satisfied / violated /
//! unknown). Returns counts per state plus list of violated and
//! unknown obligation ids.
//!
//! Demonstrates the **CMM.85** recipe for PMAT-186 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Floyd-Hoare assertion semantics; design-by-contract
//!  state classification (Meyer, Eiffel 1986).
//!
//! Run with: cargo run --example contracts_macros_obligation_satisfied_audit
//!
//! Added by PMAT-186 (catalog 1297→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum SatState {
    Satisfied,
    Violated,
    Unknown,
}

#[derive(Debug, PartialEq)]
pub enum AuditVerdict {
    Ok {
        satisfied: u32,
        violated: u32,
        unknown: u32,
        violated_ids: Vec<String>,
        unknown_ids: Vec<String>,
    },
    InvalidConfig,
}

pub fn audit(obligations: &[(&str, SatState)]) -> AuditVerdict {
    if obligations.is_empty() {
        return AuditVerdict::InvalidConfig;
    }
    let mut sat = 0u32;
    let mut vio = 0u32;
    let mut unk = 0u32;
    let mut violated_ids: Vec<String> = Vec::new();
    let mut unknown_ids: Vec<String> = Vec::new();
    for (id, state) in obligations {
        match state {
            SatState::Satisfied => sat += 1,
            SatState::Violated => {
                vio += 1;
                violated_ids.push((*id).to_string());
            }
            SatState::Unknown => {
                unk += 1;
                unknown_ids.push((*id).to_string());
            }
        }
    }
    violated_ids.sort();
    unknown_ids.sort();
    AuditVerdict::Ok {
        satisfied: sat,
        violated: vio,
        unknown: unk,
        violated_ids,
        unknown_ids,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_satisfied_audit")?;

    let obligations = [
        ("o1", SatState::Satisfied),
        ("o2", SatState::Violated),
        ("o3", SatState::Unknown),
        ("o4", SatState::Satisfied),
    ];
    println!("audit: {:?}", audit(&obligations));
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
    fn all_satisfied_zero_violated() {
        let obligations = [("o1", SatState::Satisfied), ("o2", SatState::Satisfied)];
        let v = audit(&obligations);
        if let AuditVerdict::Ok {
            satisfied,
            violated,
            ..
        } = v
        {
            assert_eq!(satisfied, 2);
            assert_eq!(violated, 0);
        }
    }

    #[test]
    fn violated_ids_collected() {
        let obligations = [("bad", SatState::Violated)];
        let v = audit(&obligations);
        if let AuditVerdict::Ok { violated_ids, .. } = v {
            assert_eq!(violated_ids, vec!["bad".to_string()]);
        }
    }

    #[test]
    fn unknown_ids_collected() {
        let obligations = [("?", SatState::Unknown)];
        let v = audit(&obligations);
        if let AuditVerdict::Ok { unknown_ids, .. } = v {
            assert_eq!(unknown_ids, vec!["?".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), AuditVerdict::InvalidConfig);
    }

    #[test]
    fn counts_sum_to_total() {
        let obligations = [
            ("a", SatState::Satisfied),
            ("b", SatState::Violated),
            ("c", SatState::Unknown),
        ];
        let v = audit(&obligations);
        if let AuditVerdict::Ok {
            satisfied,
            violated,
            unknown,
            ..
        } = v
        {
            assert_eq!(satisfied + violated + unknown, 3);
        }
    }

    #[test]
    fn deterministic() {
        let obligations = [("a", SatState::Satisfied)];
        let r1 = audit(&obligations);
        let r2 = audit(&obligations);
        assert_eq!(r1, r2);
    }

    #[test]
    fn violated_ids_sorted() {
        let obligations = [("zeta", SatState::Violated), ("alpha", SatState::Violated)];
        let v = audit(&obligations);
        if let AuditVerdict::Ok { violated_ids, .. } = v {
            assert_eq!(violated_ids, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn unknown_ids_sorted() {
        let obligations = [("zeta", SatState::Unknown), ("alpha", SatState::Unknown)];
        let v = audit(&obligations);
        if let AuditVerdict::Ok { unknown_ids, .. } = v {
            assert_eq!(unknown_ids, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn satisfied_not_in_violated_list() {
        let obligations = [("ok", SatState::Satisfied)];
        let v = audit(&obligations);
        if let AuditVerdict::Ok { violated_ids, .. } = v {
            assert!(violated_ids.is_empty());
        }
    }

    #[test]
    fn duplicate_id_kept_separate() {
        let obligations = [("a", SatState::Violated), ("a", SatState::Violated)];
        let v = audit(&obligations);
        if let AuditVerdict::Ok {
            violated,
            violated_ids,
            ..
        } = v
        {
            assert_eq!(violated, 2);
            assert_eq!(violated_ids.len(), 2);
        }
    }

    #[test]
    fn many_obligations_handled() {
        let obligations: Vec<(&str, SatState)> =
            (0..20).map(|_| ("o", SatState::Satisfied)).collect();
        let v = audit(&obligations);
        if let AuditVerdict::Ok { satisfied, .. } = v {
            assert_eq!(satisfied, 20);
        }
    }
}
