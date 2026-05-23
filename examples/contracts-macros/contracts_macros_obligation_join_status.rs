//! # Contracts-Macros Obligation Join Status
//!
//! Combine status from two sources (contract spec + impl status) into
//! a unified verdict. Both must agree to mark `Ready`; conflicts are
//! flagged with diagnostic.
//!
//! Demonstrates the **CMM.118** recipe for PMAT-197 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SQL FULL OUTER JOIN semantics; CRDT eventual-consistency
//!  reconciliation.
//!
//! Run with: cargo run --example contracts_macros_obligation_join_status
//!
//! Added by PMAT-197 (catalog 1396→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum SourceStatus {
    Pending,
    Done,
}

#[derive(Debug, PartialEq, Clone)]
pub enum JoinedStatus {
    Ready,
    SpecPending,
    ImplPending,
    BothPending,
}

#[derive(Debug, PartialEq)]
pub enum JoinVerdict {
    Ok {
        per_obligation: Vec<(String, JoinedStatus)>,
        ready_count: u32,
    },
    InvalidConfig,
}

pub fn join(sources: &[(&str, SourceStatus, SourceStatus)]) -> JoinVerdict {
    if sources.is_empty() {
        return JoinVerdict::InvalidConfig;
    }
    let mut per_obligation: Vec<(String, JoinedStatus)> = Vec::with_capacity(sources.len());
    let mut ready_count = 0u32;
    for (id, spec, imp) in sources {
        let joined = match (spec, imp) {
            (SourceStatus::Done, SourceStatus::Done) => {
                ready_count += 1;
                JoinedStatus::Ready
            }
            (SourceStatus::Pending, SourceStatus::Done) => JoinedStatus::SpecPending,
            (SourceStatus::Done, SourceStatus::Pending) => JoinedStatus::ImplPending,
            (SourceStatus::Pending, SourceStatus::Pending) => JoinedStatus::BothPending,
        };
        per_obligation.push(((*id).to_string(), joined));
    }
    JoinVerdict::Ok {
        per_obligation,
        ready_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_join_status")?;

    let sources = [
        ("o1", SourceStatus::Done, SourceStatus::Done),
        ("o2", SourceStatus::Done, SourceStatus::Pending),
        ("o3", SourceStatus::Pending, SourceStatus::Pending),
    ];
    println!("audit: {:?}", join(&sources));
    println!("invalid: {:?}", join(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn both_done_ready() {
        let s = [("o", SourceStatus::Done, SourceStatus::Done)];
        let v = join(&s);
        if let JoinVerdict::Ok { per_obligation, .. } = v {
            assert_eq!(per_obligation[0].1, JoinedStatus::Ready);
        }
    }

    #[test]
    fn spec_pending_flagged() {
        let s = [("o", SourceStatus::Pending, SourceStatus::Done)];
        let v = join(&s);
        if let JoinVerdict::Ok { per_obligation, .. } = v {
            assert_eq!(per_obligation[0].1, JoinedStatus::SpecPending);
        }
    }

    #[test]
    fn impl_pending_flagged() {
        let s = [("o", SourceStatus::Done, SourceStatus::Pending)];
        let v = join(&s);
        if let JoinVerdict::Ok { per_obligation, .. } = v {
            assert_eq!(per_obligation[0].1, JoinedStatus::ImplPending);
        }
    }

    #[test]
    fn both_pending_flagged() {
        let s = [("o", SourceStatus::Pending, SourceStatus::Pending)];
        let v = join(&s);
        if let JoinVerdict::Ok { per_obligation, .. } = v {
            assert_eq!(per_obligation[0].1, JoinedStatus::BothPending);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(join(&[]), JoinVerdict::InvalidConfig);
    }

    #[test]
    fn ready_count_accurate() {
        let s = [
            ("a", SourceStatus::Done, SourceStatus::Done),
            ("b", SourceStatus::Done, SourceStatus::Done),
            ("c", SourceStatus::Pending, SourceStatus::Done),
        ];
        let v = join(&s);
        if let JoinVerdict::Ok { ready_count, .. } = v {
            assert_eq!(ready_count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let s = [("o", SourceStatus::Done, SourceStatus::Done)];
        let r1 = join(&s);
        let r2 = join(&s);
        assert_eq!(r1, r2);
    }

    #[test]
    fn order_preserved() {
        let s = [
            ("first", SourceStatus::Done, SourceStatus::Done),
            ("second", SourceStatus::Done, SourceStatus::Done),
        ];
        let v = join(&s);
        if let JoinVerdict::Ok { per_obligation, .. } = v {
            assert_eq!(per_obligation[0].0, "first");
            assert_eq!(per_obligation[1].0, "second");
        }
    }

    #[test]
    fn count_matches_input_length() {
        let s = [
            ("a", SourceStatus::Done, SourceStatus::Done),
            ("b", SourceStatus::Pending, SourceStatus::Done),
        ];
        let v = join(&s);
        if let JoinVerdict::Ok { per_obligation, .. } = v {
            assert_eq!(per_obligation.len(), 2);
        }
    }

    #[test]
    fn all_ready_count_equals_total() {
        let s: Vec<(&str, SourceStatus, SourceStatus)> = (0..5)
            .map(|_| ("o", SourceStatus::Done, SourceStatus::Done))
            .collect();
        let v = join(&s);
        if let JoinVerdict::Ok { ready_count, .. } = v {
            assert_eq!(ready_count, 5);
        }
    }

    #[test]
    fn no_ready_when_all_pending() {
        let s = [
            ("a", SourceStatus::Pending, SourceStatus::Pending),
            ("b", SourceStatus::Pending, SourceStatus::Done),
        ];
        let v = join(&s);
        if let JoinVerdict::Ok { ready_count, .. } = v {
            assert_eq!(ready_count, 0);
        }
    }
}
