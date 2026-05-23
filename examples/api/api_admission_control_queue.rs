//! # API Admission Control Queue
//!
//! When the inference queue is full, admit/reject decisions:
//!   queue_depth < soft_limit → AdmitImmediate
//!   soft_limit ≤ depth < hard_limit → AdmitWithBackoff
//!   depth ≥ hard_limit → Reject (HTTP 503)
//!
//! Plus: priority-class override (Premium always admitted unless above
//! hard_limit × 2).
//!
//! Demonstrates the **API.11** recipe for PMAT-138 (api coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Cherkasova-Phaal admission control for overload protection.
//!
//! Run with: cargo run --example api_admission_control_queue
//!
//! Added by PMAT-138 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    Free,
    Pro,
    Premium,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackoffHint {
    None,
    Modest,
    Long,
}

#[derive(Debug, PartialEq)]
pub enum AdmissionVerdict {
    AdmitImmediate,
    AdmitWithBackoff { hint: BackoffHint },
    Reject { reason: &'static str },
    InvalidLimits,
}

pub fn decide(
    queue_depth: u32,
    soft_limit: u32,
    hard_limit: u32,
    priority: Priority,
) -> AdmissionVerdict {
    if soft_limit == 0 || hard_limit <= soft_limit {
        return AdmissionVerdict::InvalidLimits;
    }
    if priority == Priority::Premium && queue_depth < hard_limit.saturating_mul(2) {
        if queue_depth < soft_limit {
            return AdmissionVerdict::AdmitImmediate;
        }
        return AdmissionVerdict::AdmitWithBackoff {
            hint: BackoffHint::Modest,
        };
    }
    if queue_depth < soft_limit {
        return AdmissionVerdict::AdmitImmediate;
    }
    if queue_depth < hard_limit {
        let hint = if priority == Priority::Pro {
            BackoffHint::Modest
        } else {
            BackoffHint::Long
        };
        return AdmissionVerdict::AdmitWithBackoff { hint };
    }
    AdmissionVerdict::Reject {
        reason: "queue at hard limit",
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("api_admission_control_queue")?;

    let soft = 50;
    let hard = 100;
    for (depth, prio) in [
        (10u32, Priority::Free),
        (60, Priority::Free),
        (110, Priority::Free),
        (110, Priority::Premium),
        (210, Priority::Premium),
    ] {
        println!(
            "depth={depth} prio={prio:?} → {:?}",
            decide(depth, soft, hard, prio)
        );
    }
    println!("invalid: {:?}", decide(0, 0, 100, Priority::Free));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn under_soft_limit_immediate() {
        let v = decide(10, 50, 100, Priority::Free);
        assert_eq!(v, AdmissionVerdict::AdmitImmediate);
    }

    #[test]
    fn between_limits_backoff() {
        let v = decide(60, 50, 100, Priority::Free);
        assert!(matches!(v, AdmissionVerdict::AdmitWithBackoff { .. }));
    }

    #[test]
    fn at_hard_limit_rejected() {
        let v = decide(100, 50, 100, Priority::Free);
        assert!(matches!(v, AdmissionVerdict::Reject { .. }));
    }

    #[test]
    fn premium_admitted_above_hard_limit() {
        // Premium gets a 2× hard limit.
        let v = decide(110, 50, 100, Priority::Premium);
        assert!(matches!(v, AdmissionVerdict::AdmitWithBackoff { .. }));
    }

    #[test]
    fn premium_rejected_above_2x_hard() {
        let v = decide(210, 50, 100, Priority::Premium);
        assert!(matches!(v, AdmissionVerdict::Reject { .. }));
    }

    #[test]
    fn pro_gets_modest_backoff() {
        let v = decide(60, 50, 100, Priority::Pro);
        assert_eq!(
            v,
            AdmissionVerdict::AdmitWithBackoff {
                hint: BackoffHint::Modest
            }
        );
    }

    #[test]
    fn free_gets_long_backoff() {
        let v = decide(60, 50, 100, Priority::Free);
        assert_eq!(
            v,
            AdmissionVerdict::AdmitWithBackoff {
                hint: BackoffHint::Long
            }
        );
    }

    #[test]
    fn invalid_limits_zero_soft_rejected() {
        assert_eq!(
            decide(10, 0, 100, Priority::Free),
            AdmissionVerdict::InvalidLimits
        );
    }

    #[test]
    fn invalid_limits_hard_below_soft_rejected() {
        assert_eq!(
            decide(10, 100, 50, Priority::Free),
            AdmissionVerdict::InvalidLimits
        );
    }

    #[test]
    fn premium_under_soft_immediate() {
        let v = decide(10, 50, 100, Priority::Premium);
        assert_eq!(v, AdmissionVerdict::AdmitImmediate);
    }

    #[test]
    fn at_soft_limit_starts_backoff() {
        // Exactly soft_limit → backoff.
        let v = decide(50, 50, 100, Priority::Free);
        assert!(matches!(v, AdmissionVerdict::AdmitWithBackoff { .. }));
    }
}
