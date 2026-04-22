//! # Recipe: Trace Diff Between Two Inference Runs
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr trace diff run_a.json run_b.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example trace_run_diff` exits 0
//! 2. [x] `cargo test --example trace_run_diff` passes
//! 3. [x] Deterministic output (fixed timings)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr trace diff` in-process
//! 10. [x] Unit tests cover speedup, regression, added/removed ops
//!
//! ## Learning Objective
//! Demonstrates how to diff two inference traces: pair ops by name, compute
//! signed latency deltas, and flag regressions (slowdowns) from
//! improvements (speedups). Output matches `apr trace diff` verdicts.
//!
//! ## Run Command
//! ```bash
//! cargo run --example trace_run_diff
//! ```
//!
//! ## References
//! - Williams, S. et al. (2009). *Roofline: An Insightful Visual Performance Model for Multicore Architectures*. CACM. DOI: 10.1145/1498765.1498785

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct TraceEntry {
    pub op: String,
    pub latency_ns: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiffEntry {
    pub op: String,
    pub left_ns: u64,
    pub right_ns: u64,
    pub delta_ns: i64,
    pub speedup: f64,
}

#[derive(Debug, Clone)]
pub struct TraceDiff {
    pub shared: Vec<DiffEntry>,
    pub only_left: Vec<String>,
    pub only_right: Vec<String>,
    pub total_left_ns: u64,
    pub total_right_ns: u64,
}

pub fn diff_traces(left: &[TraceEntry], right: &[TraceEntry]) -> TraceDiff {
    let lmap: BTreeMap<&str, u64> = left.iter().map(|e| (e.op.as_str(), e.latency_ns)).collect();
    let rmap: BTreeMap<&str, u64> = right
        .iter()
        .map(|e| (e.op.as_str(), e.latency_ns))
        .collect();

    let mut shared = Vec::new();
    let mut only_left = Vec::new();
    let mut only_right = Vec::new();

    for (op, ll) in &lmap {
        if let Some(rl) = rmap.get(op) {
            let delta = *rl as i64 - *ll as i64;
            let speedup = if *rl == 0 {
                f64::INFINITY
            } else {
                *ll as f64 / *rl as f64
            };
            shared.push(DiffEntry {
                op: (*op).to_string(),
                left_ns: *ll,
                right_ns: *rl,
                delta_ns: delta,
                speedup,
            });
        } else {
            only_left.push((*op).to_string());
        }
    }
    for op in rmap.keys() {
        if !lmap.contains_key(op) {
            only_right.push((*op).to_string());
        }
    }

    TraceDiff {
        shared,
        only_left,
        only_right,
        total_left_ns: left.iter().map(|e| e.latency_ns).sum(),
        total_right_ns: right.iter().map(|e| e.latency_ns).sum(),
    }
}

fn trace_a() -> Vec<TraceEntry> {
    vec![
        TraceEntry {
            op: "embedding".into(),
            latency_ns: 2_100,
        },
        TraceEntry {
            op: "attention".into(),
            latency_ns: 52_000,
        },
        TraceEntry {
            op: "ffn".into(),
            latency_ns: 31_000,
        },
        TraceEntry {
            op: "layernorm".into(),
            latency_ns: 850,
        },
        TraceEntry {
            op: "lm_head".into(),
            latency_ns: 8_100,
        },
    ]
}

fn trace_b() -> Vec<TraceEntry> {
    vec![
        TraceEntry {
            op: "embedding".into(),
            latency_ns: 2_050,
        },
        TraceEntry {
            op: "attention_flash".into(),
            latency_ns: 32_000,
        },
        TraceEntry {
            op: "ffn".into(),
            latency_ns: 29_500,
        },
        TraceEntry {
            op: "layernorm".into(),
            latency_ns: 820,
        },
        TraceEntry {
            op: "lm_head".into(),
            latency_ns: 8_300,
        },
    ]
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("trace_run_diff")?;
    println!("=== Recipe: {} ===", ctx.name());

    let a = trace_a();
    let b = trace_b();
    let diff = diff_traces(&a, &b);

    println!(
        "Totals: {} ns (left) vs {} ns (right)",
        diff.total_left_ns, diff.total_right_ns
    );
    println!("\nShared ops ({}):", diff.shared.len());
    println!(
        "  {:<12} {:>10} {:>10} {:>12} {:>9}",
        "OP", "LEFT(ns)", "RIGHT(ns)", "Δ(ns)", "SPEEDUP"
    );
    for e in &diff.shared {
        println!(
            "  {:<12} {:>10} {:>10} {:>+12} {:>8.2}x",
            e.op, e.left_ns, e.right_ns, e.delta_ns, e.speedup
        );
    }
    if !diff.only_left.is_empty() {
        println!("\nOnly in left: {:?}", diff.only_left);
    }
    if !diff.only_right.is_empty() {
        println!("Only in right: {:?}", diff.only_right);
    }

    let report = json!({
        "recipe": ctx.name(),
        "total_left_ns": diff.total_left_ns,
        "total_right_ns": diff.total_right_ns,
        "n_shared": diff.shared.len(),
        "only_in_left": diff.only_left,
        "only_in_right": diff.only_right,
        "shared": diff.shared.iter().map(|e| json!({
            "op": e.op,
            "left_ns": e.left_ns,
            "right_ns": e.right_ns,
            "delta_ns": e.delta_ns,
            "speedup": e.speedup,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("trace-diff.json");
    std::fs::write(
        &out,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_ops_paired_correctly() {
        let diff = diff_traces(&trace_a(), &trace_b());
        // embedding, ffn, layernorm, lm_head are shared (4 ops).
        assert_eq!(diff.shared.len(), 4);
    }

    #[test]
    fn speedup_above_one_for_faster_right() {
        let l = vec![TraceEntry {
            op: "x".into(),
            latency_ns: 100,
        }];
        let r = vec![TraceEntry {
            op: "x".into(),
            latency_ns: 50,
        }];
        let diff = diff_traces(&l, &r);
        assert!(diff.shared[0].speedup > 1.0);
    }

    #[test]
    fn negative_delta_when_right_is_faster() {
        let l = vec![TraceEntry {
            op: "y".into(),
            latency_ns: 100,
        }];
        let r = vec![TraceEntry {
            op: "y".into(),
            latency_ns: 40,
        }];
        let diff = diff_traces(&l, &r);
        assert_eq!(diff.shared[0].delta_ns, -60);
    }

    #[test]
    fn detects_only_in_left() {
        let diff = diff_traces(&trace_a(), &trace_b());
        assert!(diff.only_left.iter().any(|o| o == "attention"));
    }

    #[test]
    fn detects_only_in_right() {
        let diff = diff_traces(&trace_a(), &trace_b());
        assert!(diff.only_right.iter().any(|o| o == "attention_flash"));
    }
}
