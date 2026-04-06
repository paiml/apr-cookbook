#![allow(unused_imports)]
//! # Recipe: Pipeline Parallelism for Distributed Inference
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! **Category**: Distributed Computing
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## Learning Objective
//! Demonstrate pipeline parallelism: splitting model layers across devices and
//! processing micro-batches through a staged pipeline. Compares pipelined vs
//! sequential execution and visualises the schedule as an ASCII Gantt chart.
//!
//! ## Run Command
//! ```bash
//! cargo run --example distributed_pipeline_parallel
//! ```
//!
//! ## Architecture
//!
//! ```text
//! Micro-batch flow through 4 pipeline stages (one per "GPU"):
//!
//!  Stage 0        Stage 1        Stage 2        Stage 3
//! [Embedding]  [Xformer 0-3]  [Xformer 4-7]  [Output Head]
//!   GPU 0          GPU 1          GPU 2          GPU 3
//!
//! Cycle   0: |  mb0  |      |      |      |
//! Cycle   1: |  mb1  |  mb0 |      |      |
//! Cycle   2: |  mb2  |  mb1 |  mb0 |      |
//! Cycle   3: |  mb3  |  mb2 |  mb1 |  mb0 |   <-- pipeline full
//! Cycle   4: |      |  mb3  |  mb2 |  mb1 |
//! Cycle   5: |      |      |  mb3  |  mb2 |
//! Cycle   6: |      |      |      |  mb3  |   <-- drain
//! ```
//!
//! ## Toyota Way Principles
//! - **Heijunka** (Level scheduling): Pipeline keeps all stages busy
//! - **Muda** (Waste elimination): Minimise idle bubble cycles
//! - **Jidoka** (Quality built-in): Deterministic schedule, verifiable metrics
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
    println!("=== Pipeline Parallelism Example ===\n");

    let mut ctx = RecipeContext::new("distributed_pipeline_parallel")?;

    // =========================================================================
    // Section 1: Define the pipeline stages
    // =========================================================================
    println!("1. Pipeline Stages (4 GPUs)");
    println!("   ─────────────────────────────────────────");

    let stages = build_pipeline();
    for stage in &stages {
        println!(
            "   Stage {}: {} (compute_cycles={})",
            stage.id, stage.name, stage.compute_cycles,
        );
    }
    println!();

    // =========================================================================
    // Section 2: Generate a batch of 8 samples and split into micro-batches
    // =========================================================================
    println!("2. Micro-Batch Splitting");
    println!("   ─────────────────────────────────────────");

    let batch_size = 8;
    let micro_batch_size = 2;
    let input_dim = 64;

    let samples: Vec<Vec<f64>> = (0..batch_size)
        .map(|_| {
            (0..input_dim)
                .map(|_| ctx.rng().gen_range(-1.0..1.0))
                .collect()
        })
        .collect();

    let micro_batches = split_micro_batches(&samples, micro_batch_size)?;
    println!("   Total samples:      {batch_size}");
    println!("   Micro-batch size:   {micro_batch_size}");
    println!("   Micro-batches:      {}", micro_batches.len());
    for mb in &micro_batches {
        println!(
            "   mb{}: {} samples, first dim={:.4}",
            mb.id,
            mb.samples.len(),
            mb.samples[0][0],
        );
    }
    println!();

    // =========================================================================
    // Section 3: Build the pipeline schedule
    // =========================================================================
    println!("3. Pipeline Schedule (Clock-Cycle Scheduler)");
    println!("   ─────────────────────────────────────────");

    let schedule = build_schedule(&stages, &micro_batches)?;
    println!("   Schedule entries: {}", schedule.len());
    for entry in &schedule {
        println!(
            "   Cycle {:2}: Stage {} processes mb{}",
            entry.cycle, entry.stage_id, entry.micro_batch_id,
        );
    }
    println!();

    // =========================================================================
    // Section 4: Gantt Chart
    // =========================================================================
    println!("4. Pipeline Gantt Chart");
    println!("   ─────────────────────────────────────────");

    let chart = render_gantt_chart(&schedule, stages.len(), micro_batches.len());
    print!("{chart}");
    println!();

    // =========================================================================
    // Section 5: Pipeline Bubble Analysis
    // =========================================================================
    println!("5. Pipeline Bubble Analysis");
    println!("   ─────────────────────────────────────────");

    let metrics = compute_metrics(&schedule, &stages, batch_size, micro_batches.len());
    println!("   Total cycles:    {}", metrics.total_cycles);
    println!("   Bubble cycles:   {}", metrics.bubble_cycles);
    println!("   Bubble ratio:    {:.1}%", metrics.bubble_ratio * 100.0);
    println!(
        "   Throughput:      {:.2} samples/cycle",
        metrics.throughput
    );
    println!("   Speedup vs seq:  {:.2}x", metrics.speedup);
    println!();

    // =========================================================================
    // Section 6: Per-Stage Utilisation
    // =========================================================================
    println!("6. Per-Stage Utilisation");
    println!("   ─────────────────────────────────────────");

    let util = stage_utilisation(&schedule, stages.len(), metrics.total_cycles);
    println!("   ┌──────────────────────┬──────────────┐");
    println!("   │ Stage                │ Utilisation  │");
    println!("   ├──────────────────────┼──────────────┤");
    for (i, &u) in util.iter().enumerate() {
        println!("   │ {:20} │ {:>10.1}%  │", stages[i].name, u * 100.0,);
    }
    println!("   └──────────────────────┴──────────────┘");
    println!();

    // =========================================================================
    // Section 7: Sequential vs Pipelined Comparison
    // =========================================================================
    println!("7. Sequential vs Pipelined Comparison");
    println!("   ─────────────────────────────────────────");

    let seq_cycles = micro_batches.len() * stages.len();
    println!("   Sequential cycles: {seq_cycles}");
    println!("   Pipelined cycles:  {}", metrics.total_cycles);
    println!("   Speedup factor:    {:.2}x", metrics.speedup);
    println!();

    // =========================================================================
    // Section 8: Simulate Forward Pass
    // =========================================================================
    println!("8. Simulated Forward Pass");
    println!("   ─────────────────────────────────────────");

    let mut current_data: Vec<Vec<Vec<f64>>> =
        micro_batches.iter().map(|mb| mb.samples.clone()).collect();

    for stage in &stages {
        for mb_data in &mut current_data {
            *mb_data = simulate_stage_forward(stage, mb_data, ctx.rng());
        }
    }

    for (i, mb_output) in current_data.iter().enumerate() {
        let first_logit = mb_output
            .first()
            .and_then(|s| s.first())
            .copied()
            .unwrap_or(0.0);
        println!(
            "   mb{i}: output_dim={}, first_logit={:.6}",
            mb_output.first().map_or(0, Vec::len),
            first_logit,
        );
    }
    println!();

    // =========================================================================
    // Section 9: Record Metrics
    // =========================================================================
    ctx.record_metric("total_cycles", metrics.total_cycles as i64);
    ctx.record_metric("bubble_cycles", metrics.bubble_cycles as i64);
    ctx.record_float_metric("bubble_ratio", metrics.bubble_ratio);
    ctx.record_float_metric("throughput", metrics.throughput);
    ctx.record_float_metric("speedup", metrics.speedup);
    ctx.report()?;

    println!("\n=== Example Complete ===");
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_pipeline_has_four_stages() {
        let stages = build_pipeline();
        assert_eq!(stages.len(), 4);
        assert_eq!(stages[0].name, "Embedding");
        assert_eq!(stages[3].name, "Output Head");
    }

    #[test]
    fn test_split_micro_batches_correct_count() {
        let samples: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64]).collect();
        let batches = split_micro_batches(&samples, 2).expect("split should succeed");
        assert_eq!(batches.len(), 4);
        for mb in &batches {
            assert_eq!(mb.samples.len(), 2);
        }
    }

    #[test]
    fn test_split_micro_batches_zero_size_errors() {
        let samples: Vec<Vec<f64>> = vec![vec![1.0]];
        let result = split_micro_batches(&samples, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_schedule_entry_count() {
        let stages = build_pipeline();
        let samples: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64]).collect();
        let mbs = split_micro_batches(&samples, 2).expect("split");
        let schedule = build_schedule(&stages, &mbs).expect("schedule");
        // 4 micro-batches * 4 stages = 16 active entries
        assert_eq!(schedule.len(), 16);
    }

    #[test]
    fn test_schedule_total_cycles() {
        let stages = build_pipeline();
        let samples: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64]).collect();
        let mbs = split_micro_batches(&samples, 2).expect("split");
        let schedule = build_schedule(&stages, &mbs).expect("schedule");
        let max_cycle = schedule.iter().map(|e| e.cycle).max().unwrap_or(0);
        // total_cycles = 4 micro-batches + 4 stages - 1 = 7, last cycle index = 6
        assert_eq!(max_cycle, 6);
    }

    #[test]
    fn test_schedule_empty_stages_errors() {
        let mbs = vec![MicroBatch {
            id: 0,
            samples: vec![vec![1.0]],
        }];
        let result = build_schedule(&[], &mbs);
        assert!(result.is_err());
    }

    #[test]
    fn test_schedule_empty_micro_batches_errors() {
        let stages = build_pipeline();
        let result = build_schedule(&stages, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_bubble_cycles_correct() {
        let stages = build_pipeline();
        let samples: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64]).collect();
        let mbs = split_micro_batches(&samples, 2).expect("split");
        let schedule = build_schedule(&stages, &mbs).expect("schedule");
        let total_cycles = mbs.len() + stages.len() - 1; // 7
        let bubbles = count_bubble_cycles(&schedule, stages.len(), total_cycles);
        // total_slots = 4 * 7 = 28, active = 16, bubbles = 12
        assert_eq!(bubbles, 12);
    }

    #[test]
    fn test_metrics_speedup() {
        let stages = build_pipeline();
        let samples: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64]).collect();
        let mbs = split_micro_batches(&samples, 2).expect("split");
        let schedule = build_schedule(&stages, &mbs).expect("schedule");
        let metrics = compute_metrics(&schedule, &stages, 8, mbs.len());
        // sequential = 4 * 4 = 16, pipelined = 7, speedup = 16/7 ~ 2.29
        assert!(metrics.speedup > 2.0);
        assert!(metrics.speedup < 3.0);
    }

    #[test]
    fn test_stage_utilisation_symmetry() {
        let stages = build_pipeline();
        let samples: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64]).collect();
        let mbs = split_micro_batches(&samples, 2).expect("split");
        let schedule = build_schedule(&stages, &mbs).expect("schedule");
        let total_cycles = mbs.len() + stages.len() - 1;
        let util = stage_utilisation(&schedule, stages.len(), total_cycles);
        // Each stage processes exactly 4 micro-batches over 7 cycles = 4/7
        for &u in &util {
            assert!((u - 4.0 / 7.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_gantt_chart_non_empty() {
        let stages = build_pipeline();
        let samples: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64]).collect();
        let mbs = split_micro_batches(&samples, 2).expect("split");
        let schedule = build_schedule(&stages, &mbs).expect("schedule");
        let chart = render_gantt_chart(&schedule, stages.len(), mbs.len());
        assert!(!chart.is_empty());
        assert!(chart.contains("mb0"));
        assert!(chart.contains("Stage 0"));
    }
}
