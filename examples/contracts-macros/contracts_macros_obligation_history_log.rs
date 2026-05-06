//! # Contracts-Macros Obligation History Log
//!
//! Verify the obligation history log is append-only and timestamps
//! are non-decreasing. Returns first violation index (if any) plus
//! sorted-status flag.
//!
//! Demonstrates the **CMM.103** recipe for PMAT-192 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: append-only log invariant in event-sourcing systems
//!  (Fowler, Event Sourcing pattern, 2005).
//!
//! Run with: cargo run --example contracts_macros_obligation_history_log
//!
//! Added by PMAT-192 (catalog 1351→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LogVerdict {
    Ok {
        sorted: bool,
        first_violation: Option<u32>,
        entries: u32,
    },
    InvalidConfig,
}

pub fn verify(timestamps: &[u64]) -> LogVerdict {
    if timestamps.is_empty() {
        return LogVerdict::InvalidConfig;
    }
    let mut first_violation: Option<u32> = None;
    for i in 1..timestamps.len() {
        if timestamps[i] < timestamps[i - 1] {
            first_violation = Some(i as u32);
            break;
        }
    }
    LogVerdict::Ok {
        sorted: first_violation.is_none(),
        first_violation,
        entries: timestamps.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_history_log")?;

    let good = [10u64, 20, 30, 40];
    println!("ordered: {:?}", verify(&good));
    let bad = [10u64, 20, 15, 30];
    println!("violation: {:?}", verify(&bad));
    println!("invalid: {:?}", verify(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn ordered_log_passes() {
        let v = verify(&[10, 20, 30]);
        if let LogVerdict::Ok {
            sorted,
            first_violation,
            ..
        } = v
        {
            assert!(sorted);
            assert!(first_violation.is_none());
        }
    }

    #[test]
    fn violation_at_index_2() {
        let v = verify(&[10, 20, 15]);
        if let LogVerdict::Ok {
            sorted,
            first_violation,
            ..
        } = v
        {
            assert!(!sorted);
            assert_eq!(first_violation, Some(2));
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(verify(&[]), LogVerdict::InvalidConfig);
    }

    #[test]
    fn single_entry_sorted() {
        let v = verify(&[42]);
        if let LogVerdict::Ok { sorted, .. } = v {
            assert!(sorted);
        }
    }

    #[test]
    fn equal_timestamps_allowed() {
        let v = verify(&[10, 10, 10]);
        if let LogVerdict::Ok { sorted, .. } = v {
            assert!(sorted);
        }
    }

    #[test]
    fn entry_count_correct() {
        let v = verify(&[10, 20, 30]);
        if let LogVerdict::Ok { entries, .. } = v {
            assert_eq!(entries, 3);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = verify(&[10, 20]);
        let r2 = verify(&[10, 20]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn first_violation_only() {
        let v = verify(&[10, 5, 1]);
        if let LogVerdict::Ok {
            first_violation, ..
        } = v
        {
            // Should report index 1 (5 < 10), not 2.
            assert_eq!(first_violation, Some(1));
        }
    }

    #[test]
    fn very_long_log() {
        let log: Vec<u64> = (0..1000).collect();
        let v = verify(&log);
        if let LogVerdict::Ok { sorted, .. } = v {
            assert!(sorted);
        }
    }

    #[test]
    fn max_value_handled() {
        let v = verify(&[u64::MAX, u64::MAX]);
        if let LogVerdict::Ok { sorted, .. } = v {
            assert!(sorted);
        }
    }

    #[test]
    fn last_entry_violation() {
        let v = verify(&[10, 20, 5]);
        if let LogVerdict::Ok {
            first_violation, ..
        } = v
        {
            assert_eq!(first_violation, Some(2));
        }
    }
}
