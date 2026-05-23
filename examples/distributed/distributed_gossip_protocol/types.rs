//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// Number of nodes participating in the gossip protocol.
pub const NUM_NODES: usize = 8;

/// Number of model parameters per node.
pub const NUM_PARAMS: usize = 100;

/// Number of gossip rounds to execute.
pub const NUM_ROUNDS: usize = 20;

// Convergence threshold: nodes are considered converged when max divergence
/// falls below this value.
pub const CONVERGENCE_EPSILON: f64 = 1e-6;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// A single node in the gossip network, holding local model parameters.
#[derive(Debug, Clone)]
pub struct GossipNode {
    // Unique node identifier.
    pub id: usize,
    // Local model parameters.
    pub params: Vec<f64>,
    // Monotonically increasing version counter, bumped on each parameter update.
    pub version: u64,
}

/// Metrics captured during a single gossip round.
#[derive(Debug, Clone)]
pub struct GossipRound {
    // Round number (0-indexed).
    pub round: usize,
    /// Pairs of node ids that exchanged parameters this round.
    #[allow(dead_code)]
    pub pairs: Vec<(usize, usize)>,
    // Maximum L2 divergence across all node pairs after this round.
    pub max_divergence: f64,
    // Average L2 divergence across all node pairs after this round.
    pub avg_divergence: f64,
    // Total number of messages sent this round (2 per exchange pair).
    pub messages: usize,
}

/// Final result of running the gossip protocol to completion.
#[derive(Debug)]
pub struct ConvergenceResult {
    // Per-round metrics.
    pub rounds: Vec<GossipRound>,
    // Whether all nodes converged within epsilon.
    pub converged: bool,
    // Max divergence at the end of the final round.
    pub final_divergence: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Core logic
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize nodes with base weights plus small per-node noise.
pub fn init_nodes(rng: &mut impl Rng, num_nodes: usize, num_params: usize) -> Vec<GossipNode> {
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
pub fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// Measure divergence across all node pairs.
//
/// Returns `(max_divergence, avg_divergence)`.
pub fn measure_divergence(nodes: &[GossipNode]) -> (f64, f64) {
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

// Select gossip pairs for one round. Each node picks a random peer.
//
/// Returns deduplicated pairs `(a, b)` where `a < b`.
pub fn select_pairs(rng: &mut impl Rng, num_nodes: usize) -> Vec<(usize, usize)> {
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
pub fn exchange_and_average(nodes: &mut [GossipNode], a: usize, b: usize) {
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
pub fn run_gossip(
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
pub fn print_initialization(nodes: &[GossipNode]) -> (f64, f64) {
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
pub fn print_convergence_analysis(result: &ConvergenceResult) {
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
pub fn print_final_verification(nodes: &[GossipNode]) {
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
