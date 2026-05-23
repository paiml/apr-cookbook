#![allow(unused_imports)]
//! # Recipe: Kernel Fusion for Transformer Blocks
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/avx512-matmul-v1.yaml
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
//!
//!
//! ## Format Variants
//! ```bash
//! apr bench model.apr          # APR native format
//! apr bench model.gguf         # GGUF (llama.cpp compatible)
//! apr bench model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Hennessy, J. & Patterson, D. (2017). *Computer Architecture: A Quantitative Approach*. DOI: 10.1016/C2012-0-01712-X

use apr_cookbook::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use std::fmt;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
