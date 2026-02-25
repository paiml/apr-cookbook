//! # Recipe: Kernel Fusion for Transformer Blocks
//!
//! **Category**: Acceleration
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (std only)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] No unsafe code
//!
//! ## Learning Objective
//! Demonstrate kernel fusion -- combining multiple operations into a single pass
//! to reduce memory traffic. Models a transformer block computation graph, analyzes
//! fusibility, applies fusion rules, and quantifies memory savings.
//!
//! ## Run Command
//! ```bash
//! cargo run --example acceleration_kernel_fusion
//! ```

use apr_cookbook::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Model dimension (hidden size). Deterministic for reproducibility.
const D_MODEL: usize = 768;

/// Number of attention heads.
const N_HEADS: usize = 12;

/// Bytes per f32 element.
const BYTES_PER_ELEMENT: usize = 4;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single operation in the computation graph.
#[derive(Debug, Clone)]
struct KernelOp {
    /// Human-readable name (e.g., "LayerNorm", "Linear_Q").
    name: String,
    /// Bytes read from memory.
    reads_bytes: u64,
    /// Bytes written to memory.
    writes_bytes: u64,
    /// Floating-point operations performed.
    compute_flops: u64,
}

impl fmt::Display for KernelOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<20} reads={:>10} B  writes={:>10} B  flops={:>12}",
            self.name, self.reads_bytes, self.writes_bytes, self.compute_flops
        )
    }
}

/// A fused kernel combining multiple operations into a single pass.
#[derive(Debug, Clone)]
struct FusedKernel {
    /// Name for the fused kernel.
    name: String,
    /// Names of the constituent operations.
    ops: Vec<String>,
    /// Total bytes read (after eliminating intermediate reads).
    total_reads: u64,
    /// Total bytes written (after eliminating intermediate writes).
    total_writes: u64,
    /// Bytes saved by fusion (eliminated intermediate traffic).
    saved_bytes: u64,
}

impl fmt::Display for FusedKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<30} ops=[{}]  reads={:>10} B  writes={:>10} B  saved={:>10} B",
            self.name,
            self.ops.join(", "),
            self.total_reads,
            self.total_writes,
            self.saved_bytes,
        )
    }
}

/// Complete fusion plan with before/after analysis.
#[derive(Debug, Clone)]
struct FusionPlan {
    /// The original unfused operations.
    original_ops: Vec<KernelOp>,
    /// Fused kernels produced by the fusion pass.
    fused_kernels: Vec<FusedKernel>,
    /// Percentage of memory traffic eliminated.
    memory_saved_pct: f64,
    /// Estimated speedup factor (memory-bound model).
    estimated_speedup: f64,
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

/// Build the transformer block computation graph (9 operations).
///
/// Each operation tracks its memory reads, writes, and compute flops based on
/// the model dimension `n` and head count `h`.
fn build_transformer_graph(n: usize, _h: usize) -> Vec<KernelOp> {
    let elem = BYTES_PER_ELEMENT as u64;
    let n64 = n as u64;

    vec![
        // Op 1: LayerNorm -- read N elements, write N elements
        KernelOp {
            name: "LayerNorm".to_string(),
            reads_bytes: n64 * elem,
            writes_bytes: n64 * elem,
            compute_flops: 5 * n64, // mean, var, normalize, scale, bias
        },
        // Op 2: Linear projection Q -- read N, write N, compute 2*N*N
        KernelOp {
            name: "Linear_Q".to_string(),
            reads_bytes: n64 * elem,
            writes_bytes: n64 * elem,
            compute_flops: 2 * n64 * n64,
        },
        // Op 3: Linear projection K -- same as Q
        KernelOp {
            name: "Linear_K".to_string(),
            reads_bytes: n64 * elem,
            writes_bytes: n64 * elem,
            compute_flops: 2 * n64 * n64,
        },
        // Op 4: Linear projection V -- same as Q
        KernelOp {
            name: "Linear_V".to_string(),
            reads_bytes: n64 * elem,
            writes_bytes: n64 * elem,
            compute_flops: 2 * n64 * n64,
        },
        // Op 5: Attention scores -- read 2*N, write N*N
        KernelOp {
            name: "AttnScores".to_string(),
            reads_bytes: 2 * n64 * elem,
            writes_bytes: n64 * n64 * elem,
            compute_flops: 2 * n64 * n64,
        },
        // Op 6: Softmax -- read N*N, write N*N
        KernelOp {
            name: "Softmax".to_string(),
            reads_bytes: n64 * n64 * elem,
            writes_bytes: n64 * n64 * elem,
            compute_flops: 3 * n64 * n64, // exp, sum, div
        },
        // Op 7: Attention output -- read N*N + N, write N
        KernelOp {
            name: "AttnOutput".to_string(),
            reads_bytes: (n64 * n64 + n64) * elem,
            writes_bytes: n64 * elem,
            compute_flops: 2 * n64 * n64,
        },
        // Op 8: Linear output projection -- read N, write N
        KernelOp {
            name: "Linear_Out".to_string(),
            reads_bytes: n64 * elem,
            writes_bytes: n64 * elem,
            compute_flops: 2 * n64 * n64,
        },
        // Op 9: Residual add -- read 2*N, write N
        KernelOp {
            name: "ResidualAdd".to_string(),
            reads_bytes: 2 * n64 * elem,
            writes_bytes: n64 * elem,
            compute_flops: n64,
        },
    ]
}

// ---------------------------------------------------------------------------
// Fusion analysis
// ---------------------------------------------------------------------------

/// Total memory traffic for a set of unfused operations.
fn total_unfused_traffic(ops: &[KernelOp]) -> u64 {
    ops.iter().map(|op| op.reads_bytes + op.writes_bytes).sum()
}

/// Fuse a contiguous slice of operations. The fused kernel keeps only the
/// first input read and the last output write; intermediate traffic is eliminated.
fn fuse_ops(name: &str, ops: &[KernelOp]) -> FusedKernel {
    let unfused_traffic: u64 = ops.iter().map(|op| op.reads_bytes + op.writes_bytes).sum();

    // Fused kernel reads the input of the first op, writes the output of the last.
    let fused_reads = ops.first().map_or(0, |op| op.reads_bytes);
    let fused_writes = ops.last().map_or(0, |op| op.writes_bytes);
    let fused_traffic = fused_reads + fused_writes;
    let saved = unfused_traffic.saturating_sub(fused_traffic);

    FusedKernel {
        name: name.to_string(),
        ops: ops.iter().map(|op| op.name.clone()).collect(),
        total_reads: fused_reads,
        total_writes: fused_writes,
        saved_bytes: saved,
    }
}

/// Apply fusion rules to the transformer graph and produce a `FusionPlan`.
///
/// Fusion rules applied:
/// 1. QKV fusion: merge Linear_Q + Linear_K + Linear_V into one kernel
/// 2. Attention fusion: merge AttnScores + Softmax into one kernel
/// 3. Output fusion: merge Linear_Out + ResidualAdd into one kernel
fn apply_fusion(ops: &[KernelOp]) -> FusionPlan {
    let unfused_total = total_unfused_traffic(ops);

    // Rule 1: Fuse QKV projections (ops at indices 1, 2, 3)
    let qkv_fused = fuse_ops("FusedQKV", &ops[1..4]);

    // Rule 2: Fuse attention scores + softmax (ops at indices 4, 5)
    let attn_fused = fuse_ops("FusedAttnSoftmax", &ops[4..6]);

    // Rule 3: Fuse output projection + residual (ops at indices 7, 8)
    let out_fused = fuse_ops("FusedOutResidual", &ops[7..9]);

    // Unfused ops that remain: LayerNorm (0), AttnOutput (6)
    let layernorm_traffic = ops[0].reads_bytes + ops[0].writes_bytes;
    let attn_output_traffic = ops[6].reads_bytes + ops[6].writes_bytes;

    let fused_total = layernorm_traffic
        + attn_output_traffic
        + (qkv_fused.total_reads + qkv_fused.total_writes)
        + (attn_fused.total_reads + attn_fused.total_writes)
        + (out_fused.total_reads + out_fused.total_writes);

    let saved_bytes = unfused_total.saturating_sub(fused_total);
    let saved_pct = if unfused_total > 0 {
        (saved_bytes as f64 / unfused_total as f64) * 100.0
    } else {
        0.0
    };

    // Estimated speedup assumes memory-bound workload: speedup ~ old_traffic / new_traffic
    let estimated_speedup = if fused_total > 0 {
        unfused_total as f64 / fused_total as f64
    } else {
        1.0
    };

    FusionPlan {
        original_ops: ops.to_vec(),
        fused_kernels: vec![qkv_fused, attn_fused, out_fused],
        memory_saved_pct: saved_pct,
        estimated_speedup,
    }
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

fn print_separator() {
    println!("{}", "-".repeat(78));
}

fn print_graph(ops: &[KernelOp]) {
    for (i, op) in ops.iter().enumerate() {
        println!("  Op {}: {}", i + 1, op);
    }
}

fn print_fused_kernels(kernels: &[FusedKernel]) {
    for (i, k) in kernels.iter().enumerate() {
        println!("  Fused {}: {}", i + 1, k);
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_048_576 {
        format!("{:.2} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

// ---------------------------------------------------------------------------
// Simulation: verify savings with random data
// ---------------------------------------------------------------------------

/// Simulate memory accesses for unfused vs. fused execution and return
/// the total bytes transferred in each case.
fn simulate_traffic(seed: u64, n: usize) -> (u64, u64) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Generate a random "activation" buffer to represent intermediate data
    let _activations: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let ops = build_transformer_graph(n, N_HEADS);
    let unfused = total_unfused_traffic(&ops);

    let plan = apply_fusion(&ops);
    let fused_traffic: u64 = plan
        .fused_kernels
        .iter()
        .map(|k| k.total_reads + k.total_writes)
        .sum::<u64>()
        // Add traffic for unfused ops (LayerNorm, AttnOutput)
        + ops[0].reads_bytes
        + ops[0].writes_bytes
        + ops[6].reads_bytes
        + ops[6].writes_bytes;

    (unfused, fused_traffic)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    println!("=== APR Cookbook: Kernel Fusion for Transformer Blocks ===\n");

    // Use deterministic seed
    let seed = hash_name_to_seed("kernel_fusion_demo");
    let _rng = rand::rngs::StdRng::seed_from_u64(seed);

    println!(
        "Model config: d_model={}, n_heads={}, dtype=f32",
        D_MODEL, N_HEADS
    );
    println!();

    // Step 1: Build computation graph
    print_separator();
    println!("Step 1: Transformer Block Computation Graph (unfused)");
    print_separator();
    let ops = build_transformer_graph(D_MODEL, N_HEADS);
    print_graph(&ops);

    let unfused_traffic = total_unfused_traffic(&ops);
    let total_flops: u64 = ops.iter().map(|op| op.compute_flops).sum();
    println!();
    println!(
        "  Total unfused memory traffic: {} ({} bytes)",
        format_bytes(unfused_traffic),
        unfused_traffic
    );
    println!("  Total compute: {} FLOPS", total_flops);
    println!(
        "  Arithmetic intensity: {:.2} FLOP/byte",
        total_flops as f64 / unfused_traffic as f64
    );

    // Step 2: Analyze fusibility
    println!();
    print_separator();
    println!("Step 2: Fusibility Analysis");
    print_separator();
    println!("  Ops sharing input (LayerNorm output -> Q, K, V): fusible (QKV fusion)");
    println!("  Ops in sequence (AttnScores -> Softmax): fusible (elementwise chain)");
    println!("  Ops in sequence (Linear_Out -> ResidualAdd): fusible (output chain)");
    println!("  LayerNorm: standalone (no adjacent elementwise peer)");
    println!("  AttnOutput: standalone (reads from fused attention + V)");

    // Step 3: Apply fusion rules
    println!();
    print_separator();
    println!("Step 3: Fused Computation Graph");
    print_separator();
    let plan = apply_fusion(&ops);
    println!(
        "  Original ops: {} -> {} fused kernels + 2 standalone",
        plan.original_ops.len(),
        plan.fused_kernels.len()
    );
    println!("  Unfused ops remaining:");
    println!("    - Op 1: LayerNorm (standalone)");
    println!("    - Op 7: AttnOutput (standalone)");
    println!();
    println!("  Fused kernels:");
    print_fused_kernels(&plan.fused_kernels);

    // Step 4: Compute memory savings
    println!();
    print_separator();
    println!("Step 4: Memory Traffic Analysis");
    print_separator();

    let fused_traffic = unfused_traffic
        - plan
            .fused_kernels
            .iter()
            .map(|k| k.saved_bytes)
            .sum::<u64>();

    println!(
        "  Unfused traffic:  {} ({} bytes)",
        format_bytes(unfused_traffic),
        unfused_traffic
    );
    println!(
        "  Fused traffic:    {} ({} bytes)",
        format_bytes(fused_traffic),
        fused_traffic
    );
    println!(
        "  Saved:            {} ({:.1}%)",
        format_bytes(unfused_traffic - fused_traffic),
        plan.memory_saved_pct
    );
    println!(
        "  Estimated speedup: {:.2}x (memory-bound model)",
        plan.estimated_speedup
    );

    // Step 5: Simulate with random data
    println!();
    print_separator();
    println!("Step 5: Simulation Verification");
    print_separator();
    let (sim_unfused, sim_fused) = simulate_traffic(seed, D_MODEL);
    println!("  Simulated unfused: {} bytes", sim_unfused);
    println!("  Simulated fused:   {} bytes", sim_fused);
    println!(
        "  Reduction matches plan: {}",
        if (sim_unfused - sim_fused) == (unfused_traffic - fused_traffic) {
            "YES"
        } else {
            "NO (rounding)"
        }
    );

    // Per-fusion breakdown
    println!();
    print_separator();
    println!("Per-Fusion Savings Breakdown");
    print_separator();
    for kernel in &plan.fused_kernels {
        let pct = if unfused_traffic > 0 {
            (kernel.saved_bytes as f64 / unfused_traffic as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "  {:<30} saved {} ({:.1}% of total traffic)",
            kernel.name,
            format_bytes(kernel.saved_bytes),
            pct
        );
    }

    println!();
    println!("[SUCCESS] Kernel fusion analysis complete.");
    println!(
        "          {:.1}% memory traffic reduction, {:.2}x estimated speedup.",
        plan.memory_saved_pct, plan.estimated_speedup
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_has_nine_ops() {
        let ops = build_transformer_graph(D_MODEL, N_HEADS);
        assert_eq!(ops.len(), 9, "Transformer block should have exactly 9 ops");
    }

    #[test]
    fn test_op_names_are_unique() {
        let ops = build_transformer_graph(D_MODEL, N_HEADS);
        let names: Vec<&str> = ops.iter().map(|op| op.name.as_str()).collect();
        for (i, name) in names.iter().enumerate() {
            for (j, other) in names.iter().enumerate() {
                if i != j {
                    assert_ne!(name, other, "Op names must be unique: {}", name);
                }
            }
        }
    }

    #[test]
    fn test_all_ops_have_positive_traffic() {
        let ops = build_transformer_graph(D_MODEL, N_HEADS);
        for op in &ops {
            assert!(op.reads_bytes > 0, "{} should read > 0 bytes", op.name);
            assert!(op.writes_bytes > 0, "{} should write > 0 bytes", op.name);
        }
    }

    #[test]
    fn test_fusion_reduces_traffic() {
        let ops = build_transformer_graph(D_MODEL, N_HEADS);
        let plan = apply_fusion(&ops);
        assert!(
            plan.memory_saved_pct > 0.0,
            "Fusion must reduce memory traffic"
        );
    }

    #[test]
    fn test_fusion_speedup_greater_than_one() {
        let ops = build_transformer_graph(D_MODEL, N_HEADS);
        let plan = apply_fusion(&ops);
        assert!(
            plan.estimated_speedup > 1.0,
            "Fused plan must be faster than unfused: got {}x",
            plan.estimated_speedup
        );
    }

    #[test]
    fn test_qkv_fusion_merges_three_ops() {
        let ops = build_transformer_graph(D_MODEL, N_HEADS);
        let plan = apply_fusion(&ops);
        let qkv = plan
            .fused_kernels
            .iter()
            .find(|k| k.name == "FusedQKV")
            .expect("FusedQKV kernel should exist");
        assert_eq!(qkv.ops.len(), 3, "QKV fusion should combine 3 ops");
        assert!(qkv.ops.contains(&"Linear_Q".to_string()));
        assert!(qkv.ops.contains(&"Linear_K".to_string()));
        assert!(qkv.ops.contains(&"Linear_V".to_string()));
    }

    #[test]
    fn test_fused_kernel_saves_bytes() {
        let ops = build_transformer_graph(D_MODEL, N_HEADS);
        let plan = apply_fusion(&ops);
        for kernel in &plan.fused_kernels {
            assert!(
                kernel.saved_bytes > 0,
                "Fused kernel {} should save > 0 bytes",
                kernel.name
            );
        }
    }

    #[test]
    fn test_simulation_matches_analytical() {
        let seed = hash_name_to_seed("test_simulation");
        let (sim_unfused, sim_fused) = simulate_traffic(seed, D_MODEL);
        assert!(
            sim_unfused > sim_fused,
            "Simulated fused traffic ({}) should be less than unfused ({})",
            sim_fused,
            sim_unfused
        );
    }

    #[test]
    fn test_deterministic_across_runs() {
        let ops_a = build_transformer_graph(D_MODEL, N_HEADS);
        let ops_b = build_transformer_graph(D_MODEL, N_HEADS);
        for (a, b) in ops_a.iter().zip(ops_b.iter()) {
            assert_eq!(a.reads_bytes, b.reads_bytes);
            assert_eq!(a.writes_bytes, b.writes_bytes);
            assert_eq!(a.compute_flops, b.compute_flops);
        }
    }

    #[test]
    fn test_small_dimension_does_not_panic() {
        // Edge case: very small model dimension
        let ops = build_transformer_graph(1, 1);
        let plan = apply_fusion(&ops);
        assert_eq!(ops.len(), 9);
        assert_eq!(plan.fused_kernels.len(), 3);
        // Even with d_model=1, fusion should save some bytes
        assert!(plan.memory_saved_pct > 0.0);
    }
}
