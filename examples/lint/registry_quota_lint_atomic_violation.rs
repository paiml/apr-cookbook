//! # Recipe: Registry Quota Lint — Atomic Write Violation
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr registry-quota-lint --observation-file observation.json` (atomic fail)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the atomic-write parity rule. The registry uses
//! write-then-rename for blob commits; if `atomic_writes != atomic_commits`
//! it means a process started a write but never renamed (process crash, OOM,
//! disk full). The garbage collector cannot reclaim those orphans without
//! risking an in-flight commit, so the lint flags this as an immediate
//! ship-blocker requiring operator attention.
//!
//! ## Run Command
//! ```bash
//! cargo run --example registry_quota_lint_atomic_violation
//! ```
//!
//! ## References
//! - aprender CRUX-A-22 (atomic-commit invariant).
//! - POSIX rename() atomicity guarantee (IEEE 1003.1-2017 §3.408).
//!
//! Added by PMAT-090 (expand-cookbooks followup — registry/cache lint coverage).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicFinding {
    pub orphans: u64,
    pub recommendation: &'static str,
}

pub fn check_atomic_parity(obs: &Value) -> Option<AtomicFinding> {
    let writes = obs.get("atomic_writes").and_then(Value::as_u64)?;
    let commits = obs.get("atomic_commits").and_then(Value::as_u64)?;
    if writes <= commits {
        return None;
    }
    let orphans = writes - commits;
    let recommendation = if orphans < 10 {
        "manual rm of .tmp-* under registry root"
    } else {
        "halt new writes; run apr registry repair --orphans"
    };
    Some(AtomicFinding {
        orphans,
        recommendation,
    })
}

fn build_torn_observation() -> Value {
    json!({
        "schema_version": 1,
        "ceiling_bytes": 50_000_000_000u64,
        "current_bytes": 12_400_000_000u64,
        "atomic_writes": 8425,
        "atomic_commits": 8421
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("registry_quota_lint_atomic_violation")?;
    let obs = build_torn_observation();
    let finding = check_atomic_parity(&obs);

    println!("=== Recipe: {} ===", ctx.name());
    match &finding {
        Some(f) => {
            println!("orphans: {} blob(s)", f.orphans);
            println!("recommendation: {}", f.recommendation);
        }
        None => println!("clean: writes == commits"),
    }
    ctx.record_string_metric("verdict", if finding.is_none() { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_violation_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn parity_emits_no_finding() {
        let obs = json!({ "atomic_writes": 100, "atomic_commits": 100 });
        assert!(check_atomic_parity(&obs).is_none());
    }

    #[test]
    fn small_orphan_count_recommends_manual_cleanup() {
        let obs = json!({ "atomic_writes": 105, "atomic_commits": 100 });
        let f = check_atomic_parity(&obs).unwrap();
        assert_eq!(f.orphans, 5);
        assert!(f.recommendation.contains("manual rm"));
    }

    #[test]
    fn large_orphan_count_recommends_repair_tool() {
        let obs = json!({ "atomic_writes": 200, "atomic_commits": 100 });
        let f = check_atomic_parity(&obs).unwrap();
        assert_eq!(f.orphans, 100);
        assert!(f.recommendation.contains("registry repair"));
    }

    #[test]
    fn commits_exceeding_writes_is_silently_clean() {
        // commits > writes means the GC already reclaimed orphans from a
        // previous run — no lint signal needed (this is normal post-recovery).
        let obs = json!({ "atomic_writes": 100, "atomic_commits": 105 });
        assert!(check_atomic_parity(&obs).is_none());
    }
}
