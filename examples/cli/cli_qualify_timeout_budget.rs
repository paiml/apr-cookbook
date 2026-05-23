//! # apr qualify — Per-Gate Timeout Budget
//!
//! `apr qualify <FILE> --timeout <S>` caps each gate's wall-clock time at
//! S seconds. The total budget is `S × len(gates)`. This recipe builds
//! the budget calculator and asserts the boundary contract: timeout=0
//! is rejected (no progress possible), timeout=u32::MAX is allowed but
//! warned about (CI runs that effectively never time out).
//!
//! Demonstrates the **QUALIFY.5** recipe for PMAT-094 (apr qualify coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender QUALIFY-003
//!
//! Run with: cargo run --example cli_qualify_timeout_budget
//!
//! Added by PMAT-094 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BudgetVerdict {
    Ok {
        total_seconds: u64,
        warn_unbounded: bool,
    },
    InvalidTimeout,
    EmptyGateList,
}

const UNBOUNDED_WARN_FLOOR: u32 = 3600 * 24; // 24 h ≈ unbounded

pub fn compute_budget(per_gate_seconds: u32, gate_count: usize) -> BudgetVerdict {
    if per_gate_seconds == 0 {
        return BudgetVerdict::InvalidTimeout;
    }
    if gate_count == 0 {
        return BudgetVerdict::EmptyGateList;
    }
    BudgetVerdict::Ok {
        total_seconds: u64::from(per_gate_seconds) * gate_count as u64,
        warn_unbounded: per_gate_seconds >= UNBOUNDED_WARN_FLOOR,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_qualify_timeout_budget")?;

    let cases = [
        ("default smoke", 120, 4),
        ("standard run", 120, 7),
        ("full strict", 300, 11),
        ("zero timeout", 0, 7),
        ("empty gates", 120, 0),
        ("unbounded", 100_000, 7),
    ];
    for (label, t, n) in cases {
        println!(
            "{label:>15}  t={t:>6}s n={n:>2}  →  {:?}",
            compute_budget(t, n)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_budget_multiplies() {
        let v = compute_budget(120, 7);
        assert_eq!(
            v,
            BudgetVerdict::Ok {
                total_seconds: 840,
                warn_unbounded: false
            }
        );
    }

    #[test]
    fn zero_timeout_rejected() {
        // Timeout 0 means every gate hits its limit on entry — no progress.
        assert_eq!(compute_budget(0, 7), BudgetVerdict::InvalidTimeout);
    }

    #[test]
    fn empty_gate_list_rejected() {
        assert_eq!(compute_budget(120, 0), BudgetVerdict::EmptyGateList);
    }

    #[test]
    fn unbounded_timeout_warned() {
        // ≥24h per gate effectively disables the timeout — surface the warning
        // so CI doesn't accidentally hang for days.
        let v = compute_budget(100_000, 4);
        if let BudgetVerdict::Ok { warn_unbounded, .. } = v {
            assert!(warn_unbounded);
        } else {
            panic!("expected Ok, got {v:?}");
        }
    }

    #[test]
    fn bounded_timeout_not_warned() {
        let v = compute_budget(120, 4);
        if let BudgetVerdict::Ok { warn_unbounded, .. } = v {
            assert!(!warn_unbounded);
        }
    }

    #[test]
    fn large_budget_does_not_overflow() {
        // u32::MAX timeout × usize gate count must compute via u64 widening.
        let v = compute_budget(u32::MAX, 1000);
        assert!(matches!(v, BudgetVerdict::Ok { .. }));
    }
}
