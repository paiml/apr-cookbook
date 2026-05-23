//! # Recipe: rm/gc Lint — Orphan Detection
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr rm-gc-lint --observation-file observation.json` (orphan path)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates orphan detection: blobs that are present in the registry
//! but referenced by no alias. Orphans are normal *during* GC (`reclaimed`
//! is true and `refcount_after` is 0) but a non-zero count *after* GC means
//! the GC pass missed them. The lint distinguishes "still-orphaned-after-gc"
//! from "expected mid-GC orphan" by gating on the `gc_completed` flag.
//!
//! ## Run Command
//! ```bash
//! cargo run --example rm_gc_lint_orphan_detection
//! ```
//!
//! ## References
//! - aprender CRUX-A-25 (orphan-after-gc invariant).
//!
//! Added by PMAT-090 (expand-cookbooks followup — registry/cache lint coverage).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrphanReport {
    pub gc_completed: bool,
    pub still_orphaned: Vec<String>, // CIDs
}

pub fn detect_orphans(obs: &Value) -> OrphanReport {
    let gc_completed = obs
        .get("gc_completed")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut still_orphaned = Vec::new();
    if let Some(blobs) = obs.get("blobs").and_then(Value::as_array) {
        for b in blobs {
            let rc = b.get("refcount_after").and_then(Value::as_i64).unwrap_or(0);
            let reclaimed = b.get("reclaimed").and_then(Value::as_bool) == Some(true);
            if rc == 0 && !reclaimed {
                if let Some(cid) = b.get("cid").and_then(Value::as_str) {
                    still_orphaned.push(cid.to_string());
                }
            }
        }
    }
    OrphanReport {
        gc_completed,
        still_orphaned,
    }
}

pub fn report_severity(r: &OrphanReport) -> &'static str {
    match (r.gc_completed, r.still_orphaned.is_empty()) {
        (_, true) => "pass",
        (false, false) => "info", // mid-GC, expected
        (true, false) => "error", // post-GC, leak
    }
}

fn build_post_gc_orphans() -> Value {
    json!({
        "gc_completed": true,
        "blobs": [
            { "cid": "b3:111", "refcount_after": 2, "reclaimed": false },
            { "cid": "b3:222", "refcount_after": 0, "reclaimed": false }, // ⚠ leak
            { "cid": "b3:333", "refcount_after": 0, "reclaimed": true  }, // ok, freed
            { "cid": "b3:444", "refcount_after": 0, "reclaimed": false }  // ⚠ leak
        ]
    })
}

fn build_mid_gc_orphans() -> Value {
    json!({
        "gc_completed": false,
        "blobs": [
            { "cid": "b3:111", "refcount_after": 0, "reclaimed": false }
        ]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("rm_gc_lint_orphan_detection")?;

    for (label, obs) in [
        ("post-gc", build_post_gc_orphans()),
        ("mid-gc", build_mid_gc_orphans()),
    ] {
        let r = detect_orphans(&obs);
        let sev = report_severity(&r);
        println!(
            "{label:>8}  gc_completed={} orphans={:?}  severity={sev}",
            r.gc_completed, r.still_orphaned
        );
    }

    ctx.record_string_metric("verdict", "matrix_printed");
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orphan_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn post_gc_orphans_are_error() {
        let r = detect_orphans(&build_post_gc_orphans());
        assert_eq!(r.still_orphaned.len(), 2);
        assert_eq!(report_severity(&r), "error");
    }

    #[test]
    fn mid_gc_orphans_are_info_not_error() {
        // Important: don't page the operator while GC is mid-run.
        let r = detect_orphans(&build_mid_gc_orphans());
        assert_eq!(r.still_orphaned.len(), 1);
        assert_eq!(report_severity(&r), "info");
    }

    #[test]
    fn no_orphans_passes_regardless_of_gc_state() {
        let obs = json!({
            "gc_completed": true,
            "blobs": [{ "cid": "x", "refcount_after": 1, "reclaimed": false }]
        });
        let r = detect_orphans(&obs);
        assert_eq!(report_severity(&r), "pass");
    }

    #[test]
    fn reclaimed_blob_does_not_count_as_orphan() {
        // refcount_after=0 with reclaimed=true is the GC's *output*, not a leak.
        let obs = json!({
            "gc_completed": true,
            "blobs": [{ "cid": "x", "refcount_after": 0, "reclaimed": true }]
        });
        let r = detect_orphans(&obs);
        assert!(r.still_orphaned.is_empty());
    }
}
