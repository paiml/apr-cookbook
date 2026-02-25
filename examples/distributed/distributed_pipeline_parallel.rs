//! # Recipe: Pipeline Parallelism for Distributed Inference
//!
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

use apr_cookbook::prelude::*;
use rand::Rng;

// ============================================================================
// Data Structures
// ============================================================================

/// One stage of the pipeline (mapped to a single device / GPU).
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Stage index (0-based).
    pub id: usize,
    /// Human-readable name (e.g. "Embedding").
    pub name: String,
    /// Number of clock cycles this stage needs to process one micro-batch.
    pub compute_cycles: usize,
}

/// A micro-batch: a small slice of the full batch that flows through the
/// pipeline independently.
#[derive(Debug, Clone)]
pub struct MicroBatch {
    /// Micro-batch index (0-based).
    pub id: usize,
    /// Sample data — one inner Vec per sample.
    pub samples: Vec<Vec<f64>>,
}

/// A single entry in the pipeline schedule: "at this cycle, this stage
/// processes this micro-batch".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleEntry {
    /// Clock cycle (0-based).
    pub cycle: usize,
    /// Pipeline stage that is active.
    pub stage_id: usize,
    /// Micro-batch being processed.
    pub micro_batch_id: usize,
}

/// Aggregate metrics produced after simulating the pipeline.
#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    /// Total clock cycles to process all micro-batches.
    pub total_cycles: usize,
    /// Idle (bubble) cycles across all stages.
    pub bubble_cycles: usize,
    /// Bubble ratio: bubble_cycles / (total_cycles * num_stages).
    pub bubble_ratio: f64,
    /// Throughput: samples processed per cycle.
    pub throughput: f64,
    /// Speed-up vs. naive sequential execution.
    pub speedup: f64,
}

// ============================================================================
// Pipeline construction helpers
// ============================================================================

/// Build the default 4-stage pipeline.
///
/// Stage 0: Embedding  (input -> 512-dim)
/// Stage 1: Transformer layers 0-3  (512 -> 512)
/// Stage 2: Transformer layers 4-7  (512 -> 512)
/// Stage 3: Output head  (512 -> vocab logits)
pub fn build_pipeline() -> Vec<PipelineStage> {
    vec![
        PipelineStage {
            id: 0,
            name: "Embedding".to_string(),
            compute_cycles: 1,
        },
        PipelineStage {
            id: 1,
            name: "Transformer L0-3".to_string(),
            compute_cycles: 1,
        },
        PipelineStage {
            id: 2,
            name: "Transformer L4-7".to_string(),
            compute_cycles: 1,
        },
        PipelineStage {
            id: 3,
            name: "Output Head".to_string(),
            compute_cycles: 1,
        },
    ]
}

/// Split a flat sample list into micro-batches of the given size.
///
/// # Errors
///
/// Returns `CookbookError::Validation` if `micro_batch_size` is zero.
pub fn split_micro_batches(
    samples: &[Vec<f64>],
    micro_batch_size: usize,
) -> Result<Vec<MicroBatch>> {
    if micro_batch_size == 0 {
        return Err(CookbookError::invalid_format(
            "micro_batch_size must be > 0",
        ));
    }

    let batches = samples
        .chunks(micro_batch_size)
        .enumerate()
        .map(|(id, chunk)| MicroBatch {
            id,
            samples: chunk.to_vec(),
        })
        .collect();

    Ok(batches)
}

// ============================================================================
// Pipeline scheduler
// ============================================================================

/// Build the clock-cycle schedule for pipeline-parallel execution.
///
/// The schedule follows the classic GPipe pattern: the pipeline fills
/// (ramp-up), runs at full throughput, then drains (ramp-down).
///
/// # Errors
///
/// Returns `CookbookError::Validation` when `stages` or `micro_batches` are empty.
pub fn build_schedule(
    stages: &[PipelineStage],
    micro_batches: &[MicroBatch],
) -> Result<Vec<ScheduleEntry>> {
    if stages.is_empty() {
        return Err(CookbookError::invalid_format(
            "pipeline must have at least one stage",
        ));
    }
    if micro_batches.is_empty() {
        return Err(CookbookError::invalid_format(
            "need at least one micro-batch",
        ));
    }

    let num_stages = stages.len();
    let num_mb = micro_batches.len();
    let total_cycles = num_mb + num_stages - 1;

    let mut schedule = Vec::with_capacity(total_cycles * num_stages);

    for cycle in 0..total_cycles {
        for (stage_idx, _stage) in stages.iter().enumerate() {
            // A stage processes micro-batch (cycle - stage_idx) if that index is valid.
            if cycle >= stage_idx {
                let mb_id = cycle - stage_idx;
                if mb_id < num_mb {
                    schedule.push(ScheduleEntry {
                        cycle,
                        stage_id: stage_idx,
                        micro_batch_id: mb_id,
                    });
                }
            }
        }
    }

    Ok(schedule)
}

/// Count idle (bubble) slots in the schedule.
///
/// A bubble is a (cycle, stage) pair where the stage is idle.
pub fn count_bubble_cycles(
    schedule: &[ScheduleEntry],
    num_stages: usize,
    total_cycles: usize,
) -> usize {
    let active_slots = schedule.len();
    let total_slots = num_stages * total_cycles;
    total_slots.saturating_sub(active_slots)
}

/// Compute pipeline metrics from the schedule.
pub fn compute_metrics(
    schedule: &[ScheduleEntry],
    stages: &[PipelineStage],
    total_samples: usize,
    num_micro_batches: usize,
) -> PipelineMetrics {
    let num_stages = stages.len();
    let total_cycles = num_micro_batches + num_stages - 1;
    let bubble_cycles = count_bubble_cycles(schedule, num_stages, total_cycles);
    let total_slots = num_stages * total_cycles;

    let bubble_ratio = if total_slots > 0 {
        bubble_cycles as f64 / total_slots as f64
    } else {
        0.0
    };

    let throughput = if total_cycles > 0 {
        total_samples as f64 / total_cycles as f64
    } else {
        0.0
    };

    // Sequential baseline: each micro-batch must pass through every stage
    // sequentially (no overlap), so total = num_micro_batches * num_stages.
    let sequential_cycles = num_micro_batches * num_stages;
    let speedup = if total_cycles > 0 {
        sequential_cycles as f64 / total_cycles as f64
    } else {
        1.0
    };

    PipelineMetrics {
        total_cycles,
        bubble_cycles,
        bubble_ratio,
        throughput,
        speedup,
    }
}

/// Compute per-stage utilisation fractions.
pub fn stage_utilisation(
    schedule: &[ScheduleEntry],
    num_stages: usize,
    total_cycles: usize,
) -> Vec<f64> {
    let mut active_per_stage = vec![0usize; num_stages];
    for entry in schedule {
        if entry.stage_id < num_stages {
            active_per_stage[entry.stage_id] += 1;
        }
    }
    active_per_stage
        .iter()
        .map(|&a| {
            if total_cycles > 0 {
                a as f64 / total_cycles as f64
            } else {
                0.0
            }
        })
        .collect()
}

// ============================================================================
// Simulation: process data through the pipeline stages
// ============================================================================

/// Simulate processing a micro-batch through one pipeline stage.
///
/// For the Embedding stage (id 0) the input tokens are projected to 512-dim.
/// For transformer stages (id 1, 2) a simple linear transform is applied.
/// For the output head (id 3) a projection to vocab_size logits is produced.
pub fn simulate_stage_forward(
    stage: &PipelineStage,
    input: &[Vec<f64>],
    rng: &mut impl rand::RngCore,
) -> Vec<Vec<f64>> {
    let dim = 512;
    let vocab_size = 128;

    match stage.id {
        // Embedding: project to 512-dim
        0 => input
            .iter()
            .map(|sample| {
                (0..dim)
                    .map(|i| sample.get(i).copied().unwrap_or(0.0) + 0.01)
                    .collect()
            })
            .collect(),
        // Transformer layers: scale + bias (simulated)
        1 | 2 => input
            .iter()
            .map(|sample| {
                let scale: f64 = 0.95 + rng.gen::<f64>() * 0.1;
                sample.iter().map(|v| v * scale).collect()
            })
            .collect(),
        // Output head: project to vocab logits
        _ => input
            .iter()
            .map(|sample| {
                (0..vocab_size)
                    .map(|i| {
                        sample
                            .iter()
                            .enumerate()
                            .map(|(j, &v)| v * ((i + j) as f64 * 0.001).sin())
                            .sum()
                    })
                    .collect()
            })
            .collect(),
    }
}

// ============================================================================
// Gantt chart rendering
// ============================================================================

/// Render an ASCII Gantt chart of the pipeline schedule.
pub fn render_gantt_chart(
    schedule: &[ScheduleEntry],
    num_stages: usize,
    num_micro_batches: usize,
) -> String {
    let total_cycles = num_micro_batches + num_stages - 1;

    // Build a lookup: (cycle, stage) -> micro_batch_id
    let mut grid: Vec<Vec<Option<usize>>> = vec![vec![None; num_stages]; total_cycles];
    for entry in schedule {
        if entry.cycle < total_cycles && entry.stage_id < num_stages {
            grid[entry.cycle][entry.stage_id] = Some(entry.micro_batch_id);
        }
    }

    let mut out = String::new();

    // Header
    out.push_str("   Cycle  ");
    for s in 0..num_stages {
        out.push_str(&format!("| Stage {s} "));
    }
    out.push_str("|\n");

    out.push_str("   -------");
    for _ in 0..num_stages {
        out.push_str("+--------");
    }
    out.push_str("+\n");

    // Rows
    for (cycle, row) in grid.iter().enumerate().take(total_cycles) {
        out.push_str(&format!("   {:5}  ", cycle));
        for cell in row.iter().take(num_stages) {
            match cell {
                Some(mb) => out.push_str(&format!("|  mb{mb:<3} ")),
                None => out.push_str("|  .... "),
            }
        }
        out.push_str("|\n");
    }

    // Footer
    out.push_str("   -------");
    for _ in 0..num_stages {
        out.push_str("+--------");
    }
    out.push_str("+\n");

    out
}

// ============================================================================
// Main
// ============================================================================

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
