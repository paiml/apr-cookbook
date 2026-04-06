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

// ============================================================================
// Data Structures
// ============================================================================

/// One stage of the pipeline (mapped to a single device / GPU).
#[derive(Debug, Clone)]
pub struct PipelineStage {
    // Stage index (0-based).
    pub id: usize,
    // Human-readable name (e.g. "Embedding").
    pub name: String,
    // Number of clock cycles this stage needs to process one micro-batch.
    pub compute_cycles: usize,
}

// A micro-batch: a small slice of the full batch that flows through the
/// pipeline independently.
#[derive(Debug, Clone)]
pub struct MicroBatch {
    // Micro-batch index (0-based).
    pub id: usize,
    // Sample data — one inner Vec per sample.
    pub samples: Vec<Vec<f64>>,
}

// A single entry in the pipeline schedule: "at this cycle, this stage
/// processes this micro-batch".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleEntry {
    // Clock cycle (0-based).
    pub cycle: usize,
    // Pipeline stage that is active.
    pub stage_id: usize,
    // Micro-batch being processed.
    pub micro_batch_id: usize,
}

/// Aggregate metrics produced after simulating the pipeline.
#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    // Total clock cycles to process all micro-batches.
    pub total_cycles: usize,
    // Idle (bubble) cycles across all stages.
    pub bubble_cycles: usize,
    // Bubble ratio: bubble_cycles / (total_cycles * num_stages).
    pub bubble_ratio: f64,
    // Throughput: samples processed per cycle.
    pub throughput: f64,
    // Speed-up vs. naive sequential execution.
    pub speedup: f64,
}

// ============================================================================
// Pipeline construction helpers
// ============================================================================

// Build the default 4-stage pipeline.
//
// Stage 0: Embedding  (input -> 512-dim)
// Stage 1: Transformer layers 0-3  (512 -> 512)
// Stage 2: Transformer layers 4-7  (512 -> 512)
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

// Split a flat sample list into micro-batches of the given size.
//
// # Errors
//
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

// Build the clock-cycle schedule for pipeline-parallel execution.
//
// The schedule follows the classic GPipe pattern: the pipeline fills
// (ramp-up), runs at full throughput, then drains (ramp-down).
//
// # Errors
//
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

// Count idle (bubble) slots in the schedule.
//
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

// Simulate processing a micro-batch through one pipeline stage.
//
// For the Embedding stage (id 0) the input tokens are projected to 512-dim.
// For transformer stages (id 1, 2) a simple linear transform is applied.
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
