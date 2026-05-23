//! # Distributed Byzantine Quorum Sizer
//!
//! Quorum requirements:
//!   Crash failures: tolerate f failures with N = 2f + 1 nodes
//!   Byzantine failures: tolerate f failures with N = 3f + 1 nodes
//!
//! This recipe checks: given (failure_model, total_nodes, max_failures),
//! is the cluster correctly sized? Returns required_quorum_for_decision.
//!
//! Demonstrates the **DIST.11** recipe for PMAT-142 (distributed coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Castro & Liskov (1999). Practical Byzantine Fault Tolerance.
//!
//! Run with: cargo run --example distributed_byzantine_quorum
//!
//! Added by PMAT-142 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureModel {
    Crash,
    Byzantine,
}

#[derive(Debug, PartialEq)]
pub enum QuorumVerdict {
    Ok {
        required_quorum: u32,
        ok_to_decide: bool,
    },
    UnderProvisioned {
        required_nodes: u32,
        actual: u32,
    },
    InvalidNodeCount,
}

pub fn check(
    model: FailureModel,
    total_nodes: u32,
    expected_failures: u32,
    available_nodes: u32,
) -> QuorumVerdict {
    if total_nodes == 0 {
        return QuorumVerdict::InvalidNodeCount;
    }
    let required_total = match model {
        FailureModel::Crash => 2 * expected_failures + 1,
        FailureModel::Byzantine => 3 * expected_failures + 1,
    };
    if total_nodes < required_total {
        return QuorumVerdict::UnderProvisioned {
            required_nodes: required_total,
            actual: total_nodes,
        };
    }
    let required_quorum = match model {
        FailureModel::Crash => total_nodes / 2 + 1,
        FailureModel::Byzantine => 2 * total_nodes / 3 + 1,
    };
    QuorumVerdict::Ok {
        required_quorum,
        ok_to_decide: available_nodes >= required_quorum,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distributed_byzantine_quorum")?;

    println!(
        "Crash 5/1, all up: {:?}",
        check(FailureModel::Crash, 5, 1, 5)
    );
    println!(
        "Crash 5/1, 2 down: {:?}",
        check(FailureModel::Crash, 5, 1, 3)
    );
    println!(
        "Byzantine 7/2: {:?}",
        check(FailureModel::Byzantine, 7, 2, 7)
    );
    println!(
        "Byzantine 4/2 under-provisioned: {:?}",
        check(FailureModel::Byzantine, 4, 2, 4)
    );
    println!("Invalid: {:?}", check(FailureModel::Crash, 0, 0, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quorum_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn crash_5_nodes_quorum_3() {
        // N = 5, simple majority quorum = 3.
        let v = check(FailureModel::Crash, 5, 1, 5);
        if let QuorumVerdict::Ok {
            required_quorum, ..
        } = v
        {
            assert_eq!(required_quorum, 3);
        }
    }

    #[test]
    fn byzantine_7_nodes_quorum_5() {
        // N = 7, 2f+1 = 5 quorum.
        let v = check(FailureModel::Byzantine, 7, 2, 7);
        if let QuorumVerdict::Ok {
            required_quorum, ..
        } = v
        {
            assert_eq!(required_quorum, 5);
        }
    }

    #[test]
    fn crash_under_provisioned_for_2_failures() {
        // Need 2×2+1 = 5 nodes for 2 failures; have 3.
        let v = check(FailureModel::Crash, 3, 2, 3);
        assert!(matches!(v, QuorumVerdict::UnderProvisioned { .. }));
    }

    #[test]
    fn byzantine_under_provisioned_for_2_failures() {
        // Need 3×2+1 = 7 nodes for 2 failures; have 4.
        let v = check(FailureModel::Byzantine, 4, 2, 4);
        assert!(matches!(v, QuorumVerdict::UnderProvisioned { .. }));
    }

    #[test]
    fn ok_to_decide_when_enough_alive() {
        let v = check(FailureModel::Crash, 5, 1, 4);
        if let QuorumVerdict::Ok { ok_to_decide, .. } = v {
            assert!(ok_to_decide);
        }
    }

    #[test]
    fn not_ok_to_decide_when_too_few_alive() {
        let v = check(FailureModel::Crash, 5, 1, 2);
        if let QuorumVerdict::Ok { ok_to_decide, .. } = v {
            assert!(!ok_to_decide);
        }
    }

    #[test]
    fn invalid_zero_nodes() {
        assert_eq!(
            check(FailureModel::Crash, 0, 0, 0),
            QuorumVerdict::InvalidNodeCount
        );
    }

    #[test]
    fn byzantine_requires_more_nodes_than_crash() {
        // For f=2: crash needs 5, byzantine needs 7.
        let crash = check(FailureModel::Crash, 5, 2, 5);
        let byzantine = check(FailureModel::Byzantine, 5, 2, 5);
        assert!(matches!(crash, QuorumVerdict::Ok { .. }));
        assert!(matches!(byzantine, QuorumVerdict::UnderProvisioned { .. }));
    }

    #[test]
    fn byzantine_quorum_larger_than_crash() {
        let crash = check(FailureModel::Crash, 9, 2, 9);
        let byzantine = check(FailureModel::Byzantine, 9, 2, 9);
        if let (
            QuorumVerdict::Ok {
                required_quorum: cq,
                ..
            },
            QuorumVerdict::Ok {
                required_quorum: bq,
                ..
            },
        ) = (crash, byzantine)
        {
            assert!(bq > cq);
        }
    }

    #[test]
    fn zero_failures_quorum_simple_majority() {
        let v = check(FailureModel::Crash, 5, 0, 5);
        if let QuorumVerdict::Ok {
            required_quorum, ..
        } = v
        {
            assert_eq!(required_quorum, 3);
        }
    }
}
