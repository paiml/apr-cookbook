//! # apr gpu --nccl-strategy — Multi-GPU Communication Strategy
//!
//! Multi-GPU training distributes the global batch via one of:
//! Ring (best for ≤ 8 GPUs, good bandwidth efficiency), Tree (better
//! for ≥ 16 GPUs, lower latency), Hierarchical (multi-node fallback).
//! This recipe builds the picker.
//!
//! Demonstrates the **GPU.4** recipe for PMAT-120 (apr gpu coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GPU-001 + NCCL collective algorithms (Patarasuk & Yuan 2009)
//!
//! Run with: cargo run --example cli_gpu_nccl_strategy_picker
//!
//! Added by PMAT-120 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclStrategy {
    Ring,
    Tree,
    Hierarchical,
}

#[derive(Debug, PartialEq)]
pub enum PickerVerdict {
    Ok(NcclStrategy),
    NoGpus,
}

pub fn pick(num_gpus: u32, num_nodes: u32) -> PickerVerdict {
    if num_gpus == 0 {
        return PickerVerdict::NoGpus;
    }
    if num_nodes > 1 {
        return PickerVerdict::Ok(NcclStrategy::Hierarchical);
    }
    if num_gpus <= 8 {
        PickerVerdict::Ok(NcclStrategy::Ring)
    } else {
        PickerVerdict::Ok(NcclStrategy::Tree)
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_gpu_nccl_strategy_picker")?;

    let cases = [(0u32, 1u32), (4, 1), (8, 1), (16, 1), (8, 2)];
    for (g, n) in cases {
        println!("gpus={g} nodes={n}  →  {:?}", pick(g, n));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn zero_gpus_rejected() {
        assert_eq!(pick(0, 1), PickerVerdict::NoGpus);
    }

    #[test]
    fn small_single_node_uses_ring() {
        assert_eq!(pick(4, 1), PickerVerdict::Ok(NcclStrategy::Ring));
    }

    #[test]
    fn at_8_gpus_still_ring() {
        // Boundary: ≤ 8 inclusive.
        assert_eq!(pick(8, 1), PickerVerdict::Ok(NcclStrategy::Ring));
    }

    #[test]
    fn over_8_single_node_uses_tree() {
        assert_eq!(pick(9, 1), PickerVerdict::Ok(NcclStrategy::Tree));
        assert_eq!(pick(16, 1), PickerVerdict::Ok(NcclStrategy::Tree));
    }

    #[test]
    fn multi_node_uses_hierarchical() {
        assert_eq!(pick(8, 2), PickerVerdict::Ok(NcclStrategy::Hierarchical));
    }

    #[test]
    fn multi_node_overrides_gpu_count_choice() {
        // Even with only 4 GPUs total, multi-node forces hierarchical.
        assert_eq!(pick(4, 2), PickerVerdict::Ok(NcclStrategy::Hierarchical));
    }

    #[test]
    fn single_gpu_single_node_uses_ring() {
        // Degenerate but valid.
        assert_eq!(pick(1, 1), PickerVerdict::Ok(NcclStrategy::Ring));
    }

    #[test]
    fn large_single_node_uses_tree() {
        assert_eq!(pick(64, 1), PickerVerdict::Ok(NcclStrategy::Tree));
    }
}
