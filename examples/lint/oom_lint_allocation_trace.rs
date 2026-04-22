//! # Recipe: OOM-Lint — Allocation Trace Composition
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr oom-lint postmortem.json --with-trace trace.jsonl`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example oom_lint_allocation_trace` exits 0
//! 2. [x] `cargo test --example oom_lint_allocation_trace` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Combines `oom-lint` with an allocation trace (NDJSON of per-tensor
//! allocations ordered by time). Reconstructs the memory-at-fault timeline,
//! identifies the top-N largest live tensors at the moment of failure, and
//! emits a diagnosis suitable for a post-incident review.
//!
//! ## Run Command
//! ```bash
//! cargo run --example oom_lint_allocation_trace
//! ```
//!
//! ## References
//! - Ren, J. et al. (2021). *ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning*. arXiv:2104.07857

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Alloc {
    pub name: String,
    pub bytes: u64,
    pub op: String, // "alloc" | "free"
}

pub fn parse_trace(ndjson: &str) -> Vec<Alloc> {
    ndjson
        .lines()
        .filter(|l| !l.is_empty())
        .filter_map(|l| {
            let v: Value = serde_json::from_str(l).ok()?;
            Some(Alloc {
                name: v.get("name")?.as_str()?.to_string(),
                bytes: v.get("bytes")?.as_u64()?,
                op: v.get("op")?.as_str()?.to_string(),
            })
        })
        .collect()
}

pub fn live_tensors(trace: &[Alloc]) -> HashMap<String, u64> {
    let mut live = HashMap::new();
    for a in trace {
        match a.op.as_str() {
            "alloc" => {
                *live.entry(a.name.clone()).or_insert(0) += a.bytes;
            }
            "free" => {
                if let Some(v) = live.get_mut(&a.name) {
                    *v = v.saturating_sub(a.bytes);
                    if *v == 0 {
                        live.remove(&a.name);
                    }
                }
            }
            _ => {}
        }
    }
    live
}

pub fn top_n<S: std::hash::BuildHasher>(
    live: &HashMap<String, u64, S>,
    n: usize,
) -> Vec<(String, u64)> {
    let mut v: Vec<(String, u64)> = live.iter().map(|(k, v)| (k.clone(), *v)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v.into_iter().take(n).collect()
}

fn build_trace() -> String {
    let events = [
        ("alloc", "embed.weight", 2_000_000_000u64),
        ("alloc", "decoder.q", 12_000_000_000u64),
        ("alloc", "decoder.k", 12_000_000_000u64),
        ("alloc", "decoder.v", 12_000_000_000u64),
        ("alloc", "decoder.o", 12_000_000_000u64),
        ("free", "embed.weight", 2_000_000_000u64),
        ("alloc", "activations", 16_000_000_000u64),
    ];
    let mut out = String::new();
    for (op, name, bytes) in events {
        out.push_str(
            &serde_json::to_string(&json!({
                "op": op, "name": name, "bytes": bytes
            }))
            .unwrap_or_default(),
        );
        out.push('\n');
    }
    out
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("oom_lint_allocation_trace")?;
    let trace_src = build_trace();
    let p = ctx.path("trace.ndjson");
    std::fs::write(&p, &trace_src)?;

    let trace = parse_trace(&trace_src);
    let live = live_tensors(&trace);
    let top3 = top_n(&live, 3);
    let live_total: u64 = live.values().sum();

    println!("=== Recipe: {} ===", ctx.name());
    println!("Trace: {} events", trace.len());
    println!(
        "Live tensors at fault: {} (total={} bytes)",
        live.len(),
        live_total
    );
    println!("Top 3:");
    for (name, bytes) in &top3 {
        println!("  {:<20}  {:>14}", name, bytes);
    }
    // Typical heuristic: attention KV + activations dominate.
    let attn_live: u64 = live
        .iter()
        .filter(|(k, _)| k.starts_with("decoder."))
        .map(|(_, v)| *v)
        .sum();
    let attn_pct = (100 * attn_live).checked_div(live_total).unwrap_or(0);
    println!(
        "\nAttention KV live = {} bytes ({} % of live)",
        attn_live, attn_pct
    );

    ctx.record_metric("live_tensors", live.len() as i64);
    ctx.record_metric("live_bytes", live_total as i64);
    ctx.record_metric("attn_kv_live_bytes", attn_live as i64);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_parses() {
        let t = parse_trace(&build_trace());
        assert_eq!(t.len(), 7);
    }

    #[test]
    fn freed_tensor_not_in_live() {
        let t = parse_trace(&build_trace());
        let live = live_tensors(&t);
        assert!(!live.contains_key("embed.weight"));
    }

    #[test]
    fn top_n_sorted_descending() {
        let t = parse_trace(&build_trace());
        let live = live_tensors(&t);
        let top = top_n(&live, 3);
        assert!(top[0].1 >= top[1].1);
        assert!(top[1].1 >= top[2].1);
    }
}
