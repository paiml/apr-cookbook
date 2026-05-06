//! # Distributed All-Reduce Strategy Picker
//!
//! All-reduce aggregates per-worker gradients into a global mean. Three
//! algorithms: Ring (bandwidth-optimal for small W, latency = O(W));
//! Tree (latency-optimal, log W); Recursive Halving-Doubling (best for
//! large W on fat-tree). This recipe picks the strategy based on
//! worker count + payload size + topology.
//!
//! Demonstrates the **DIST.3** recipe for PMAT-124 (distributed coverage —
//! closing F-invariant gap).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Patarasuk & Yuan (2009). Bandwidth optimal all-reduce algorithms for clusters of workstations.
//!
//! Run with: cargo run --example distributed_allreduce_strategy
//!
//! Added by PMAT-124 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllReduceStrategy {
    Ring,
    Tree,
    RecursiveHalvingDoubling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    SingleSwitch,
    FatTree,
    Hierarchical,
}

#[derive(Debug, PartialEq)]
pub enum PickVerdict {
    Ok(AllReduceStrategy),
    NoWorkers,
    InvalidPayload,
}

const SMALL_PAYLOAD_BYTES: u64 = 64 * 1024; // 64 KiB

pub fn pick(num_workers: u32, payload_bytes: u64, topology: Topology) -> PickVerdict {
    if num_workers == 0 {
        return PickVerdict::NoWorkers;
    }
    if payload_bytes == 0 {
        return PickVerdict::InvalidPayload;
    }
    if num_workers == 1 {
        // Trivial — any algorithm degenerates to noop.
        return PickVerdict::Ok(AllReduceStrategy::Ring);
    }
    let strategy = match (num_workers, payload_bytes < SMALL_PAYLOAD_BYTES, topology) {
        // Tiny payloads: latency dominates → Tree.
        (_, true, _) => AllReduceStrategy::Tree,
        // Large payloads on fat-tree: recursive halving.
        (n, false, Topology::FatTree) if n >= 16 => AllReduceStrategy::RecursiveHalvingDoubling,
        // Large payloads on simple switch: Ring (bandwidth optimal).
        (_, false, Topology::SingleSwitch | Topology::FatTree) => AllReduceStrategy::Ring,
        // Hierarchical (multi-rack) → recursive halving.
        (_, false, Topology::Hierarchical) => AllReduceStrategy::RecursiveHalvingDoubling,
    };
    PickVerdict::Ok(strategy)
}

pub fn estimated_latency_factor(strategy: AllReduceStrategy, num_workers: u32) -> f64 {
    let w = f64::from(num_workers.max(1));
    match strategy {
        AllReduceStrategy::Ring => 2.0 * (w - 1.0),
        AllReduceStrategy::Tree => 2.0 * (w.log2().ceil()),
        AllReduceStrategy::RecursiveHalvingDoubling => 2.0 * w.log2().ceil(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distributed_allreduce_strategy")?;

    let cases = [
        (8u32, 1024u64, Topology::SingleSwitch),
        (8, 1_000_000, Topology::SingleSwitch),
        (32, 10_000_000, Topology::FatTree),
        (32, 10_000_000, Topology::Hierarchical),
        (1, 1024, Topology::SingleSwitch),
    ];
    for (w, p, t) in cases {
        println!("W={w} payload={p} topo={t:?}  →  {:?}", pick(w, p, t));
    }
    for w in [4u32, 16, 64, 256] {
        for s in [
            AllReduceStrategy::Ring,
            AllReduceStrategy::Tree,
            AllReduceStrategy::RecursiveHalvingDoubling,
        ] {
            println!(
                "W={w} {s:?}  →  latency factor = {}",
                estimated_latency_factor(s, w)
            );
        }
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
    fn small_payload_picks_tree() {
        // Tiny payload → latency dominates → Tree.
        assert_eq!(
            pick(8, 1024, Topology::SingleSwitch),
            PickVerdict::Ok(AllReduceStrategy::Tree)
        );
    }

    #[test]
    fn large_payload_single_switch_picks_ring() {
        assert_eq!(
            pick(8, 10_000_000, Topology::SingleSwitch),
            PickVerdict::Ok(AllReduceStrategy::Ring)
        );
    }

    #[test]
    fn large_workers_fat_tree_picks_halving_doubling() {
        assert_eq!(
            pick(32, 10_000_000, Topology::FatTree),
            PickVerdict::Ok(AllReduceStrategy::RecursiveHalvingDoubling)
        );
    }

    #[test]
    fn hierarchical_topology_uses_halving_doubling() {
        assert_eq!(
            pick(32, 10_000_000, Topology::Hierarchical),
            PickVerdict::Ok(AllReduceStrategy::RecursiveHalvingDoubling)
        );
    }

    #[test]
    fn single_worker_degenerates() {
        assert_eq!(
            pick(1, 1_000_000, Topology::SingleSwitch),
            PickVerdict::Ok(AllReduceStrategy::Ring)
        );
    }

    #[test]
    fn zero_workers_rejected() {
        assert_eq!(
            pick(0, 1024, Topology::SingleSwitch),
            PickVerdict::NoWorkers
        );
    }

    #[test]
    fn zero_payload_rejected() {
        assert_eq!(
            pick(8, 0, Topology::SingleSwitch),
            PickVerdict::InvalidPayload
        );
    }

    #[test]
    fn ring_latency_linear_in_workers() {
        let l4 = estimated_latency_factor(AllReduceStrategy::Ring, 4);
        let l8 = estimated_latency_factor(AllReduceStrategy::Ring, 8);
        // Ring: 2(w-1); l8 / l4 should be (2*7) / (2*3) = 14/6 ≈ 2.33.
        assert!((l8 / l4 - 14.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn tree_latency_logarithmic() {
        // Tree: 2 ⌈log₂ w⌉; l16 / l4 = (2*4) / (2*2) = 2.
        let l4 = estimated_latency_factor(AllReduceStrategy::Tree, 4);
        let l16 = estimated_latency_factor(AllReduceStrategy::Tree, 16);
        assert!((l16 / l4 - 2.0).abs() < 1e-6);
    }
}
