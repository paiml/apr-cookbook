//! # apr ptx-map — Reverse Lookup (kernel → layers)
//!
//! `apr ptx-map <FILE> --reverse <KERNEL>` inverts the layer→kernel
//! dispatch map: given a kernel name, list every layer/step that
//! actually invokes it. This recipe builds the reverse-index pure
//! function and asserts the contract: an empty result means the kernel
//! is dead code (compiled but never dispatched), which is itself a
//! useful CI signal.
//!
//! Demonstrates the **PTXMAP.4** recipe for PMAT-096 (apr ptx-map coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PTX-MAP-002 + dead-kernel detection
//!
//! Run with: cargo run --example cli_ptx_map_reverse_lookup
//!
//! Added by PMAT-096 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchEntry {
    pub layer: &'static str,
    pub step: &'static str, // "prefill" | "decode"
    pub kernel: &'static str,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ReverseIndex {
    pub kernel: String,
    pub callers: Vec<(String, String)>, // (layer, step)
    pub is_dead: bool,
}

pub fn build_reverse_index(map: &[DispatchEntry], kernel: &str) -> ReverseIndex {
    let callers: Vec<(String, String)> = map
        .iter()
        .filter(|d| d.kernel == kernel)
        .map(|d| (d.layer.to_string(), d.step.to_string()))
        .collect();
    let is_dead = callers.is_empty();
    ReverseIndex {
        kernel: kernel.to_string(),
        callers,
        is_dead,
    }
}

pub fn group_by_kernel(map: &[DispatchEntry]) -> BTreeMap<&'static str, usize> {
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for d in map {
        *counts.entry(d.kernel).or_insert(0) += 1;
    }
    counts
}

fn sample_dispatch_map() -> Vec<DispatchEntry> {
    vec![
        DispatchEntry {
            layer: "model.layers.0.self_attn",
            step: "decode",
            kernel: "Q4KGemv",
        },
        DispatchEntry {
            layer: "model.layers.0.self_attn",
            step: "prefill",
            kernel: "Q4KGemm",
        },
        DispatchEntry {
            layer: "model.layers.1.self_attn",
            step: "decode",
            kernel: "Q4KGemv",
        },
        DispatchEntry {
            layer: "model.layers.0.mlp",
            step: "decode",
            kernel: "Q4KGemv",
        },
        DispatchEntry {
            layer: "lm_head",
            step: "decode",
            kernel: "FP16Gemm",
        },
    ]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_ptx_map_reverse_lookup")?;
    let map = sample_dispatch_map();

    for k in ["Q4KGemv", "Q4KGemm", "FP16Gemm", "FA3HmaB128xS128"] {
        let idx = build_reverse_index(&map, k);
        if idx.is_dead {
            println!("--reverse {k:>20}  →  DEAD KERNEL (no callers)");
        } else {
            println!("--reverse {k:>20}  →  {} callers", idx.callers.len());
            for (layer, step) in idx.callers {
                println!("    {layer}  [{step}]");
            }
        }
    }

    println!("\nKernel-frequency report:");
    for (k, n) in group_by_kernel(&map) {
        println!("  {k:>15}  {n} dispatches");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn known_kernel_yields_callers() {
        let map = sample_dispatch_map();
        let idx = build_reverse_index(&map, "Q4KGemv");
        assert!(!idx.is_dead);
        assert_eq!(idx.callers.len(), 3);
    }

    #[test]
    fn dead_kernel_flagged() {
        // Kernel compiled into the binary but never dispatched.
        let map = sample_dispatch_map();
        let idx = build_reverse_index(&map, "FA3HmaB128xS128");
        assert!(idx.is_dead);
        assert!(idx.callers.is_empty());
    }

    #[test]
    fn each_caller_carries_layer_and_step() {
        let map = sample_dispatch_map();
        let idx = build_reverse_index(&map, "Q4KGemv");
        for (layer, step) in &idx.callers {
            assert!(!layer.is_empty());
            assert!(!step.is_empty());
            assert!(matches!(step.as_str(), "prefill" | "decode"));
        }
    }

    #[test]
    fn group_by_kernel_counts_match_total_dispatches() {
        let map = sample_dispatch_map();
        let counts = group_by_kernel(&map);
        let total: usize = counts.values().sum();
        assert_eq!(total, map.len());
    }

    #[test]
    fn group_by_kernel_alphabetical_ordering() {
        let map = sample_dispatch_map();
        let counts = group_by_kernel(&map);
        let keys: Vec<&&str> = counts.keys().collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "BTreeMap must yield sorted keys");
    }
}
