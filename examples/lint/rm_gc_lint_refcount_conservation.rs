//! # Recipe: rm/gc Lint — Refcount Conservation
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr rm-gc-lint --observation-file observation.json` (refcount path)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the refcount conservation law: for each blob touched by an
//! `apr rm` operation, `refcount_before == refcount_after + aliases_removed`.
//! Violations indicate either lost decrements (refcount stuck high — leak)
//! or extra decrements (refcount too low — silent corruption that the
//! orphan detector cannot reach).
//!
//! ## Run Command
//! ```bash
//! cargo run --example rm_gc_lint_refcount_conservation
//! ```
//!
//! ## References
//! - aprender CRUX-A-25 (refcount conservation invariant).
//! - rkyv/zerocopy (refcount accounting in zero-copy registries).
//!
//! Added by PMAT-090 (expand-cookbooks followup — registry/cache lint coverage).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConservationFinding {
    pub cid: String,
    pub before: i64,
    pub removed: i64,
    pub after: i64,
    pub kind: ConservationKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConservationKind {
    LeakedRef,   // after > before - removed (decrement lost)
    DoubleFreed, // after < before - removed (extra decrement)
}

pub fn check_conservation(obs: &Value) -> Vec<ConservationFinding> {
    let mut out = Vec::new();
    let Some(blobs) = obs.get("blobs").and_then(Value::as_array) else {
        return out;
    };
    for b in blobs {
        let cid = b.get("cid").and_then(Value::as_str).unwrap_or("?").into();
        let before = b
            .get("refcount_before")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let after = b.get("refcount_after").and_then(Value::as_i64).unwrap_or(0);
        let removed = b
            .get("aliases_removed")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let expected = before - removed;
        if after > expected {
            out.push(ConservationFinding {
                cid,
                before,
                removed,
                after,
                kind: ConservationKind::LeakedRef,
            });
        } else if after < expected {
            out.push(ConservationFinding {
                cid,
                before,
                removed,
                after,
                kind: ConservationKind::DoubleFreed,
            });
        }
    }
    out
}

fn build_conserved_observation() -> Value {
    json!({
        "blobs": [
            { "cid": "b3:aaa", "refcount_before": 3, "aliases_removed": 1, "refcount_after": 2 },
            { "cid": "b3:bbb", "refcount_before": 1, "aliases_removed": 1, "refcount_after": 0 }
        ]
    })
}

fn build_violating_observation() -> Value {
    json!({
        "blobs": [
            { "cid": "b3:aaa", "refcount_before": 3, "aliases_removed": 1, "refcount_after": 3 }, // leaked
            { "cid": "b3:bbb", "refcount_before": 1, "aliases_removed": 1, "refcount_after": -1 } // double-free
        ]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("rm_gc_lint_refcount_conservation")?;

    let clean = check_conservation(&build_conserved_observation());
    let dirty = check_conservation(&build_violating_observation());

    println!("=== Recipe: {} ===", ctx.name());
    println!("conserved obs:     {} findings", clean.len());
    println!("violating obs:     {} findings", dirty.len());
    for f in &dirty {
        println!(
            "  {} {:?}: before={}, removed={}, after={} (expected={})",
            f.cid,
            f.kind,
            f.before,
            f.removed,
            f.after,
            f.before - f.removed
        );
    }

    ctx.record_metric("violations", dirty.len() as i64);
    ctx.record_string_metric("verdict", if clean.is_empty() { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conservation_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn conserved_observation_has_no_findings() {
        let f = check_conservation(&build_conserved_observation());
        assert!(f.is_empty(), "expected clean: {f:?}");
    }

    #[test]
    fn detects_leaked_reference() {
        let obs = json!({
            "blobs": [{ "cid": "x", "refcount_before": 5, "aliases_removed": 2, "refcount_after": 5 }]
        });
        let f = check_conservation(&obs);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].kind, ConservationKind::LeakedRef);
    }

    #[test]
    fn detects_double_free() {
        let obs = json!({
            "blobs": [{ "cid": "x", "refcount_before": 1, "aliases_removed": 1, "refcount_after": -1 }]
        });
        let f = check_conservation(&obs);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].kind, ConservationKind::DoubleFreed);
    }

    #[test]
    fn separate_kinds_in_same_observation_both_reported() {
        let f = check_conservation(&build_violating_observation());
        assert_eq!(f.len(), 2);
        let kinds: Vec<ConservationKind> = f.iter().map(|x| x.kind).collect();
        assert!(kinds.contains(&ConservationKind::LeakedRef));
        assert!(kinds.contains(&ConservationKind::DoubleFreed));
    }
}
