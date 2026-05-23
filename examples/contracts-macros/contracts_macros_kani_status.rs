//! # Contracts-Macros Kani Status Tracker
//!
//! Track Kani (symbolic execution) status per contract obligation:
//! Verified, Timeout, Failed, NotAttempted. Roll up to a single
//! contract-level Kani verdict.
//!
//! Demonstrates the **CMM.22** recipe for PMAT-165 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Kani Rust Verifier (model-checker for Rust).
//!
//! Run with: cargo run --example contracts_macros_kani_status
//!
//! Added by PMAT-165 (catalog 1108→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KaniStatus {
    Verified,
    Timeout,
    Failed,
    NotAttempted,
}

#[derive(Debug, PartialEq)]
pub enum KaniRollup {
    AllVerified {
        count: u32,
    },
    Mixed {
        verified: u32,
        timeout: u32,
        failed: u32,
        not_attempted: u32,
    },
    HasFailures {
        failed: u32,
    },
    EmptyContract,
}

pub fn rollup(statuses: &[KaniStatus]) -> KaniRollup {
    if statuses.is_empty() {
        return KaniRollup::EmptyContract;
    }
    let mut verified = 0u32;
    let mut timeout = 0u32;
    let mut failed = 0u32;
    let mut not_attempted = 0u32;
    for s in statuses {
        match s {
            KaniStatus::Verified => verified += 1,
            KaniStatus::Timeout => timeout += 1,
            KaniStatus::Failed => failed += 1,
            KaniStatus::NotAttempted => not_attempted += 1,
        }
    }
    if failed > 0 {
        return KaniRollup::HasFailures { failed };
    }
    if verified == statuses.len() as u32 {
        return KaniRollup::AllVerified { count: verified };
    }
    KaniRollup::Mixed {
        verified,
        timeout,
        failed,
        not_attempted,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_kani_status")?;

    println!(
        "all verified: {:?}",
        rollup(&[KaniStatus::Verified, KaniStatus::Verified])
    );
    println!(
        "has failure: {:?}",
        rollup(&[KaniStatus::Verified, KaniStatus::Failed])
    );
    println!(
        "mixed: {:?}",
        rollup(&[
            KaniStatus::Verified,
            KaniStatus::Timeout,
            KaniStatus::NotAttempted
        ])
    );
    println!("empty: {:?}", rollup(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollup_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn all_verified_recognized() {
        let v = rollup(&[KaniStatus::Verified, KaniStatus::Verified]);
        if let KaniRollup::AllVerified { count } = v {
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn failed_takes_precedence() {
        let v = rollup(&[KaniStatus::Verified, KaniStatus::Failed]);
        assert!(matches!(v, KaniRollup::HasFailures { .. }));
    }

    #[test]
    fn mixed_returns_counts() {
        let v = rollup(&[
            KaniStatus::Verified,
            KaniStatus::Timeout,
            KaniStatus::NotAttempted,
        ]);
        if let KaniRollup::Mixed {
            verified,
            timeout,
            failed,
            not_attempted,
        } = v
        {
            assert_eq!(verified, 1);
            assert_eq!(timeout, 1);
            assert_eq!(failed, 0);
            assert_eq!(not_attempted, 1);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(rollup(&[]), KaniRollup::EmptyContract);
    }

    #[test]
    fn single_verified_all_verified() {
        let v = rollup(&[KaniStatus::Verified]);
        assert!(matches!(v, KaniRollup::AllVerified { count: 1 }));
    }

    #[test]
    fn single_timeout_mixed() {
        let v = rollup(&[KaniStatus::Timeout]);
        assert!(matches!(v, KaniRollup::Mixed { .. }));
    }

    #[test]
    fn single_failed_has_failures() {
        let v = rollup(&[KaniStatus::Failed]);
        if let KaniRollup::HasFailures { failed } = v {
            assert_eq!(failed, 1);
        }
    }

    #[test]
    fn multiple_failures_count() {
        let v = rollup(&[KaniStatus::Failed, KaniStatus::Failed, KaniStatus::Verified]);
        if let KaniRollup::HasFailures { failed } = v {
            assert_eq!(failed, 2);
        }
    }

    #[test]
    fn all_not_attempted_mixed() {
        let v = rollup(&[KaniStatus::NotAttempted, KaniStatus::NotAttempted]);
        assert!(matches!(v, KaniRollup::Mixed { .. }));
    }

    #[test]
    fn deterministic() {
        let s = vec![KaniStatus::Verified, KaniStatus::Timeout];
        let a = rollup(&s);
        let b = rollup(&s);
        assert_eq!(a, b);
    }
}
