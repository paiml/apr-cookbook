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

/// Number of nodes participating in the gossip protocol.
const NUM_NODES: usize = 8;

/// Number of model parameters per node.
const NUM_PARAMS: usize = 100;

/// Number of gossip rounds to execute.
const NUM_ROUNDS: usize = 20;

/// Convergence threshold: nodes are considered converged when max divergence
/// falls below this value.
const CONVERGENCE_EPSILON: f64 = 1e-6;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// A single node in the gossip network, holding local model parameters.
#[derive(Debug, Clone)]
struct GossipNode {
    /// Unique node identifier.
    id: usize,
    /// Local model parameters.
    params: Vec<f64>,
    /// Monotonically increasing version counter, bumped on each parameter update.
    version: u64,
}

/// Metrics captured during a single gossip round.
#[derive(Debug, Clone)]
struct GossipRound {
    /// Round number (0-indexed).
    round: usize,
    /// Pairs of node ids that exchanged parameters this round.
    #[allow(dead_code)]
    pairs: Vec<(usize, usize)>,
    /// Maximum L2 divergence across all node pairs after this round.
    max_divergence: f64,
    /// Average L2 divergence across all node pairs after this round.
    avg_divergence: f64,
    /// Total number of messages sent this round (2 per exchange pair).
    messages: usize,
}

/// Final result of running the gossip protocol to completion.
#[derive(Debug)]
struct ConvergenceResult {
    /// Per-round metrics.
    rounds: Vec<GossipRound>,
    /// Whether all nodes converged within epsilon.
    converged: bool,
    /// Max divergence at the end of the final round.
    final_divergence: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Core logic
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize nodes with base weights plus small per-node noise.
fn init_nodes(rng: &mut impl Rng, num_nodes: usize, num_params: usize) -> Vec<GossipNode> {
    // Generate shared base weights
    let base: Vec<f64> = (0..num_params).map(|_| rng.gen_range(-1.0..1.0)).collect();

    (0..num_nodes)
        .map(|id| {
            let params: Vec<f64> = base.iter().map(|&b| b + rng.gen_range(-0.1..0.1)).collect();
            GossipNode {
                id,
                params,
                version: 0,
            }
        })
        .collect()
}

/// Compute the L2 (Euclidean) distance between two parameter vectors.
fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Measure divergence across all node pairs.
///
/// Returns `(max_divergence, avg_divergence)`.
fn measure_divergence(nodes: &[GossipNode]) -> (f64, f64) {
    let n = nodes.len();
    let mut max_div: f64 = 0.0;
    let mut sum_div: f64 = 0.0;
    let mut count: usize = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let d = l2_distance(&nodes[i].params, &nodes[j].params);
            if d > max_div {
                max_div = d;
            }
            sum_div += d;
            count += 1;
        }
    }

    let avg_div = if count > 0 {
        sum_div / count as f64
    } else {
        0.0
    };
    (max_div, avg_div)
}

/// Select gossip pairs for one round. Each node picks a random peer.
///
/// Returns deduplicated pairs `(a, b)` where `a < b`.
fn select_pairs(rng: &mut impl Rng, num_nodes: usize) -> Vec<(usize, usize)> {
    let mut paired = vec![false; num_nodes];
    let mut pairs = Vec::new();

    // Iterate nodes in random order
    let mut order: Vec<usize> = (0..num_nodes).collect();
    for i in (1..order.len()).rev() {
        let j = rng.gen_range(0..=i);
        order.swap(i, j);
    }

    for &node in &order {
        if paired[node] {
            continue;
        }
        // Pick a random unpaired peer
        let unpaired: Vec<usize> = (0..num_nodes)
            .filter(|&p| p != node && !paired[p])
            .collect();
        if unpaired.is_empty() {
            continue;
        }
        let peer = unpaired[rng.gen_range(0..unpaired.len())];

        let (a, b) = if node < peer {
            (node, peer)
        } else {
            (peer, node)
        };
        pairs.push((a, b));
        paired[a] = true;
        paired[b] = true;
    }

    pairs
}

/// Execute one gossip exchange: both nodes average their parameters.
fn exchange_and_average(nodes: &mut [GossipNode], a: usize, b: usize) {
    let num_params = nodes[a].params.len();
    for i in 0..num_params {
        let avg = (nodes[a].params[i] + nodes[b].params[i]) * 0.5;
        nodes[a].params[i] = avg;
        nodes[b].params[i] = avg;
    }
    nodes[a].version += 1;
    nodes[b].version += 1;
}

/// Run the full gossip protocol for the specified number of rounds.
fn run_gossip(
    rng: &mut impl Rng,
    nodes: &mut [GossipNode],
    num_rounds: usize,
) -> ConvergenceResult {
    let mut rounds = Vec::with_capacity(num_rounds);
    let mut converged = false;
    let mut final_divergence = f64::MAX;

    for round_idx in 0..num_rounds {
        let pairs = select_pairs(rng, nodes.len());
        let messages = pairs.len() * 2; // each pair exchanges 2 messages

        for &(a, b) in &pairs {
            exchange_and_average(nodes, a, b);
        }

        let (max_div, avg_div) = measure_divergence(nodes);
        final_divergence = max_div;

        rounds.push(GossipRound {
            round: round_idx,
            pairs,
            max_divergence: max_div,
            avg_divergence: avg_div,
            messages,
        });

        if max_div < CONVERGENCE_EPSILON {
            converged = true;
        }
    }

    ConvergenceResult {
        rounds,
        converged,
        final_divergence,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Print node initialization summary and return initial divergence metrics.
fn print_initialization(nodes: &[GossipNode]) -> (f64, f64) {
    println!("1. Node Initialization");
    println!("   ─────────────────────────────────────────");
    println!("   Nodes:       {NUM_NODES}");
    println!("   Parameters:  {NUM_PARAMS}");
    println!("   Rounds:      {NUM_ROUNDS}");
    println!("   Epsilon:     {CONVERGENCE_EPSILON}");
    println!();

    let (init_max, init_avg) = measure_divergence(nodes);
    println!("   Initial divergence:");
    println!("     Max L2: {init_max:.6}");
    println!("     Avg L2: {init_avg:.6}");
    println!();

    for node in nodes {
        println!(
            "   Node {} params[0..3]: [{:.4}, {:.4}, {:.4}]",
            node.id, node.params[0], node.params[1], node.params[2]
        );
    }
    println!();

    (init_max, init_avg)
}

/// Print convergence analysis: halving rate, total messages, and final status.
fn print_convergence_analysis(result: &ConvergenceResult) {
    println!("3. Convergence Analysis");
    println!("   ─────────────────────────────────────────");

    if result.rounds.len() >= 4 {
        println!("   Halving rate (max divergence ratio between consecutive rounds):");
        for i in 1..result.rounds.len() {
            let prev = result.rounds[i - 1].max_divergence;
            let curr = result.rounds[i].max_divergence;
            let ratio = if prev > 0.0 { curr / prev } else { 0.0 };
            if i <= 10 || i == result.rounds.len() - 1 {
                println!("     Round {} -> {}: ratio = {:.4}", i - 1, i, ratio);
            }
        }
    }
    println!();

    let total_messages: usize = result.rounds.iter().map(|r| r.messages).sum();
    println!("   Total messages sent: {total_messages}");
    println!("   Final max divergence: {:.10}", result.final_divergence);
    println!(
        "   Converged (< {CONVERGENCE_EPSILON}): {}",
        result.converged
    );
    println!();
}

/// Verify all nodes are within epsilon of each other, print results.
fn print_final_verification(nodes: &[GossipNode]) {
    println!("4. Final Verification");
    println!("   ─────────────────────────────────────────");

    let verification_epsilon = 1e-4;
    let all_close = nodes.iter().enumerate().all(|(i, _)| {
        nodes[i + 1..]
            .iter()
            .all(|other| l2_distance(&nodes[i].params, &other.params) <= verification_epsilon)
    });

    println!("   Verification epsilon: {verification_epsilon}");
    println!("   All nodes within epsilon: {all_close}");
    println!();

    for node in nodes {
        println!(
            "   Node {} (v{}): params[0..3] = [{:.6}, {:.6}, {:.6}]",
            node.id, node.version, node.params[0], node.params[1], node.params[2]
        );
    }
    println!();
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

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
