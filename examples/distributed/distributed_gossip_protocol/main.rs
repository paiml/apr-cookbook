#![allow(unused_imports)]
//! Gossip Protocol for Decentralized Model Parameter Averaging
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Demonstrates a gossip-based protocol where nodes exchange and average model
//! parameters without a central coordinator. Each round, every node picks a
//! random peer, and the pair averages their parameters. Over time, all nodes
//! converge to the same parameter vector (the global average).
//!
//! # Algorithm
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                   Gossip Protocol Rounds                      │
//! ├──────────────────────────────────────────────────────────────┤
//! │  Round 1:  N0↔N3  N1↔N5  N2↔N7  N4↔N6                      │
//! │  Round 2:  N0↔N1  N2↔N4  N3↔N6  N5↔N7                      │
//! │  ...                                                         │
//! │  Round K:  All nodes converge → global average               │
//! │                                                              │
//! │  Convergence: divergence halves roughly every 2 rounds       │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example distributed_gossip_protocol
//! ```
//!
//! # Recipe Metadata
//!
//! - **Category**: Distributed Computing
//! - **Complexity**: Advanced
//! - **Dependencies**: rand, apr_cookbook
//! - **IIUR**: Isolated, Idempotent, Useful, Reproducible
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Dean, J. et al. (2012). *Large Scale Distributed Deep Networks*. NeurIPS. arXiv:1206.5533

use apr_cookbook::prelude::*;
use rand::Rng;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    println!("=== Gossip Protocol: Decentralized Parameter Averaging ===\n");

    let mut ctx = RecipeContext::new("distributed_gossip_protocol")?;

    // Section 1: Initialize nodes
    let mut nodes = init_nodes(ctx.rng(), NUM_NODES, NUM_PARAMS);
    let (init_max, _init_avg) = print_initialization(&nodes);

    // Section 2: Run gossip rounds
    println!("2. Gossip Rounds");
    println!("   ─────────────────────────────────────────");
    println!("   ┌───────┬────────────────┬────────────────┬──────────┐");
    println!("   │ Round │ Max Divergence │ Avg Divergence │ Messages │");
    println!("   ├───────┼────────────────┼────────────────┼──────────┤");

    let result = run_gossip(ctx.rng(), &mut nodes, NUM_ROUNDS);

    for round in &result.rounds {
        println!(
            "   │ {:>5} │ {:>14.8} │ {:>14.8} │ {:>8} │",
            round.round, round.max_divergence, round.avg_divergence, round.messages,
        );
    }
    println!("   └───────┴────────────────┴────────────────┴──────────┘");
    println!();

    // Section 3: Convergence analysis
    print_convergence_analysis(&result);

    // Section 4: Final verification
    print_final_verification(&nodes);

    // Record metrics
    let total_messages: usize = result.rounds.iter().map(|r| r.messages).sum();
    ctx.record_float_metric("initial_max_divergence", init_max);
    ctx.record_float_metric("final_max_divergence", result.final_divergence);
    ctx.record_metric("total_messages", total_messages as i64);
    ctx.record_metric("total_rounds", NUM_ROUNDS as i64);
    ctx.record_string_metric("converged", if result.converged { "yes" } else { "no" });

    ctx.report()?;

    println!("\n=== Example Complete ===");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn test_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_init_nodes_count() {
        let mut rng = test_rng();
        let nodes = init_nodes(&mut rng, 8, 100);
        assert_eq!(nodes.len(), 8);
        for (i, node) in nodes.iter().enumerate() {
            assert_eq!(node.id, i);
            assert_eq!(node.params.len(), 100);
            assert_eq!(node.version, 0);
        }
    }

    #[test]
    fn test_init_nodes_different_params() {
        let mut rng = test_rng();
        let nodes = init_nodes(&mut rng, 4, 50);
        // Nodes should have slightly different parameters (base + noise)
        assert_ne!(nodes[0].params, nodes[1].params);
        assert_ne!(nodes[1].params, nodes[2].params);
    }

    #[test]
    fn test_l2_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!((l2_distance(&a, &b)).abs() < 1e-15);
    }

    #[test]
    fn test_l2_distance_known_value() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((l2_distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_measure_divergence_identical_nodes() {
        let nodes = vec![
            GossipNode {
                id: 0,
                params: vec![1.0, 2.0],
                version: 0,
            },
            GossipNode {
                id: 1,
                params: vec![1.0, 2.0],
                version: 0,
            },
        ];
        let (max_div, avg_div) = measure_divergence(&nodes);
        assert!(max_div.abs() < 1e-15);
        assert!(avg_div.abs() < 1e-15);
    }

    #[test]
    fn test_select_pairs_valid() {
        let mut rng = test_rng();
        let pairs = select_pairs(&mut rng, 8);
        // With 8 nodes, we should get exactly 4 pairs
        assert_eq!(pairs.len(), 4);
        // Each pair should have a < b
        for &(a, b) in &pairs {
            assert!(a < b);
            assert!(a < 8);
            assert!(b < 8);
        }
        // No node should appear in more than one pair
        let mut seen = vec![false; 8];
        for &(a, b) in &pairs {
            assert!(!seen[a], "Node {a} appears in multiple pairs");
            assert!(!seen[b], "Node {b} appears in multiple pairs");
            seen[a] = true;
            seen[b] = true;
        }
    }

    #[test]
    fn test_exchange_and_average() {
        let mut nodes = vec![
            GossipNode {
                id: 0,
                params: vec![2.0, 4.0, 6.0],
                version: 0,
            },
            GossipNode {
                id: 1,
                params: vec![8.0, 10.0, 12.0],
                version: 0,
            },
        ];
        exchange_and_average(&mut nodes, 0, 1);
        // After averaging: (2+8)/2=5, (4+10)/2=7, (6+12)/2=9
        assert!((nodes[0].params[0] - 5.0).abs() < 1e-10);
        assert!((nodes[0].params[1] - 7.0).abs() < 1e-10);
        assert!((nodes[0].params[2] - 9.0).abs() < 1e-10);
        // Both nodes should have identical params
        assert_eq!(nodes[0].params, nodes[1].params);
        // Versions should increment
        assert_eq!(nodes[0].version, 1);
        assert_eq!(nodes[1].version, 1);
    }

    #[test]
    fn test_gossip_convergence() {
        let mut rng = test_rng();
        let mut nodes = init_nodes(&mut rng, NUM_NODES, NUM_PARAMS);
        let result = run_gossip(&mut rng, &mut nodes, NUM_ROUNDS);
        // After 20 rounds with 8 nodes, divergence should be very small
        assert!(
            result.final_divergence < 1e-3,
            "Expected convergence, got divergence = {}",
            result.final_divergence
        );
        assert!(!result.rounds.is_empty());
    }

    #[test]
    fn test_gossip_divergence_decreases() {
        let mut rng = test_rng();
        let mut nodes = init_nodes(&mut rng, 8, 50);
        let result = run_gossip(&mut rng, &mut nodes, 10);
        // Divergence should generally decrease (check first vs last)
        let first_div = result.rounds[0].max_divergence;
        let last_div = result.rounds[result.rounds.len() - 1].max_divergence;
        assert!(
            last_div < first_div,
            "Divergence should decrease: first={first_div}, last={last_div}"
        );
    }

    #[test]
    fn test_gossip_deterministic() {
        let mut rng1 = StdRng::seed_from_u64(99);
        let mut nodes1 = init_nodes(&mut rng1, 4, 20);
        let result1 = run_gossip(&mut rng1, &mut nodes1, 5);

        let mut rng2 = StdRng::seed_from_u64(99);
        let mut nodes2 = init_nodes(&mut rng2, 4, 20);
        let result2 = run_gossip(&mut rng2, &mut nodes2, 5);

        for (r1, r2) in result1.rounds.iter().zip(result2.rounds.iter()) {
            assert!(
                (r1.max_divergence - r2.max_divergence).abs() < 1e-15,
                "Gossip should be deterministic for the same seed"
            );
        }
    }
}
