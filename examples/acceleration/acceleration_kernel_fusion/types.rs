#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Model dimension (hidden size). Deterministic for reproducibility.
pub const D_MODEL: usize = 768;

/// Number of attention heads.
pub const N_HEADS: usize = 12;

/// Bytes per f32 element.
pub const BYTES_PER_ELEMENT: usize = 4;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single operation in the computation graph.
#[derive(Debug, Clone)]
pub struct KernelOp {
    // Human-readable name (e.g., "LayerNorm", "Linear_Q").
    pub name: String,
    // Bytes read from memory.
    pub reads_bytes: u64,
    // Bytes written to memory.
    pub writes_bytes: u64,
    // Floating-point operations performed.
    pub compute_flops: u64,
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
pub struct FusedKernel {
    // Name for the fused kernel.
    pub name: String,
    // Names of the constituent operations.
    pub ops: Vec<String>,
    // Total bytes read (after eliminating intermediate reads).
    pub total_reads: u64,
    // Total bytes written (after eliminating intermediate writes).
    pub total_writes: u64,
    // Bytes saved by fusion (eliminated intermediate traffic).
    pub saved_bytes: u64,
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
pub struct FusionPlan {
    // The original unfused operations.
    pub original_ops: Vec<KernelOp>,
    // Fused kernels produced by the fusion pass.
    pub fused_kernels: Vec<FusedKernel>,
    // Percentage of memory traffic eliminated.
    pub memory_saved_pct: f64,
    // Estimated speedup factor (memory-bound model).
    pub estimated_speedup: f64,
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

// Build the transformer block computation graph (9 operations).
//
// Each operation tracks its memory reads, writes, and compute flops based on
/// the model dimension `n` and head count `h`.
pub fn build_transformer_graph(n: usize, _h: usize) -> Vec<KernelOp> {
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
pub fn total_unfused_traffic(ops: &[KernelOp]) -> u64 {
    ops.iter().map(|op| op.reads_bytes + op.writes_bytes).sum()
}

// Fuse a contiguous slice of operations. The fused kernel keeps only the
/// first input read and the last output write; intermediate traffic is eliminated.
pub fn fuse_ops(name: &str, ops: &[KernelOp]) -> FusedKernel {
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

// Apply fusion rules to the transformer graph and produce a `FusionPlan`.
//
// Fusion rules applied:
// 1. QKV fusion: merge Linear_Q + Linear_K + Linear_V into one kernel
// 2. Attention fusion: merge AttnScores + Softmax into one kernel
/// 3. Output fusion: merge Linear_Out + ResidualAdd into one kernel
pub fn apply_fusion(ops: &[KernelOp]) -> FusionPlan {
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

pub fn print_separator() {
    println!("{}", "-".repeat(78));
}

pub fn print_graph(ops: &[KernelOp]) {
    for (i, op) in ops.iter().enumerate() {
        println!("  Op {}: {}", i + 1, op);
    }
}

pub fn print_fused_kernels(kernels: &[FusedKernel]) {
    for (i, k) in kernels.iter().enumerate() {
        println!("  Fused {}: {}", i + 1, k);
    }
}

pub fn format_bytes(bytes: u64) -> String {
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

// Simulate memory accesses for unfused vs. fused execution and return
/// the total bytes transferred in each case.
pub fn simulate_traffic(seed: u64, n: usize) -> (u64, u64) {
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
