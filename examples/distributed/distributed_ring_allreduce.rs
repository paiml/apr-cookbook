//! Ring-Allreduce for Distributed Gradient Aggregation
//!
//! Demonstrates the ring-allreduce algorithm used in distributed training to
//! aggregate gradients across worker nodes with optimal bandwidth utilization.
//!
//! # Algorithm
//!
//! Ring-allreduce proceeds in two phases over a logical ring of N workers:
//!
//! ```text
//! Phase 1 - Scatter-Reduce (N-1 steps):
//!   Each worker sends one chunk to its right neighbour and receives from
//!   its left neighbour, accumulating (summing) into the received chunk.
//!   After N-1 steps every worker owns the fully-reduced value for exactly
//!   one chunk.
//!
//! Phase 2 - Allgather (N-1 steps):
//!   Each worker sends its complete chunk around the ring.  After N-1 steps
//!   every worker has the full reduced gradient.
//! ```
//!
//! # Communication Cost Comparison
//!
//! | Strategy | Messages           | Total Bytes              |
//! |----------|--------------------|--------------------------|
//! | Ring     | 2*(N-1)            | 2*(N-1)*M/N              |
//! | Naive    | N*(N-1)            | N*(N-1)*M                |
//!
//! Ring-allreduce achieves near-optimal bandwidth utilisation regardless of N.
//!
//! # Running
//!
//! ```bash
//! cargo run --example distributed_ring_allreduce
//! ```
//!
//! # Recipe Metadata
//!
//! - **Category**: Distributed Computing
//! - **Complexity**: Advanced
//! - **Dependencies**: std, rand
//! - **IIUR**: Isolated, Idempotent, Useful, Reproducible
//!
//! ## References
//! - Dean, J. et al. (2012). *Large Scale Distributed Deep Networks*. NeurIPS. arXiv:1206.5533

use apr_cookbook::prelude::*;
use rand::Rng;
use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A simulated worker node in the ring.
#[derive(Debug, Clone)]
struct WorkerNode {
    id: usize,
    gradient: Vec<f64>,
    buffer: Vec<f64>,
}

/// One communication step in the ring-allreduce algorithm.
#[derive(Debug, Clone)]
struct RingStep {
    phase: Phase,
    sender: usize,
    receiver: usize,
    chunk_id: usize,
    bytes_transferred: usize,
}

/// Phase of the ring-allreduce algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    ScatterReduce,
    Allgather,
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScatterReduce => write!(f, "scatter-reduce"),
            Self::Allgather => write!(f, "allgather"),
        }
    }
}

/// Result of a completed allreduce operation.
#[derive(Debug)]
struct AllreduceResult {
    final_gradient: Vec<f64>,
    total_bytes: usize,
    n_steps: usize,
    correct: bool,
}

// ---------------------------------------------------------------------------
// Ring-Allreduce implementation
// ---------------------------------------------------------------------------

/// Compute chunk boundaries for splitting a vector of `total` elements into
/// `n_chunks` roughly-equal pieces.  Returns `(start, len)` pairs.
fn chunk_ranges(total: usize, n_chunks: usize) -> Vec<(usize, usize)> {
    let base = total / n_chunks;
    let remainder = total % n_chunks;
    let mut ranges = Vec::with_capacity(n_chunks);
    let mut offset = 0;
    for i in 0..n_chunks {
        let len = base + usize::from(i < remainder);
        ranges.push((offset, len));
        offset += len;
    }
    ranges
}

/// Execute the ring-allreduce algorithm across the given workers.
///
/// Returns the communication log and the final allreduce result (verified
/// against the expected sum of all original local gradients).
fn ring_allreduce(
    workers: &mut [WorkerNode],
    expected_sum: &[f64],
) -> (Vec<RingStep>, AllreduceResult) {
    let n = workers.len();
    let grad_len = workers[0].gradient.len();
    let ranges = chunk_ranges(grad_len, n);
    let bytes_per_element = size_of::<f64>();
    let mut log: Vec<RingStep> = Vec::new();

    // --- Phase 1: Scatter-Reduce -------------------------------------------
    for step in 0..n - 1 {
        // Each worker i sends chunk (i - step) mod n to worker (i+1) mod n
        // and receives chunk (i - step - 1) mod n from worker (i-1) mod n,
        // then reduces (sums) into its own buffer.
        //
        // We collect the sends first to avoid borrow conflicts.
        let sends: Vec<(usize, usize, Vec<f64>)> = (0..n)
            .map(|i| {
                let chunk_id = (n + i - step) % n;
                let (start, len) = ranges[chunk_id];
                let data = workers[i].gradient[start..start + len].to_vec();
                let receiver = (i + 1) % n;
                (receiver, chunk_id, data)
            })
            .collect();

        for (receiver, chunk_id, data) in &sends {
            let (start, len) = ranges[*chunk_id];
            for (j, &val) in data.iter().enumerate() {
                workers[*receiver].gradient[start + j] += val;
            }
            log.push(RingStep {
                phase: Phase::ScatterReduce,
                sender: (n + *receiver - 1) % n,
                receiver: *receiver,
                chunk_id: *chunk_id,
                bytes_transferred: len * bytes_per_element,
            });
        }
    }

    // --- Phase 2: Allgather ------------------------------------------------
    // After scatter-reduce, worker i owns the fully-reduced chunk
    // (i + 1) mod n.  Copy that into every worker's buffer first.
    for (i, worker) in workers.iter_mut().enumerate() {
        let owning_chunk = (i + 1) % n;
        let (start, len) = ranges[owning_chunk];
        worker.buffer = worker.gradient.clone();
        let _chunk_slice = &worker.buffer[start..start + len];
    }

    for step in 0..n - 1 {
        let sends: Vec<(usize, usize, Vec<f64>)> = (0..n)
            .map(|i| {
                let chunk_id = (n + i + 1 - step) % n;
                let (start, len) = ranges[chunk_id];
                let data = workers[i].gradient[start..start + len].to_vec();
                let receiver = (i + 1) % n;
                (receiver, chunk_id, data)
            })
            .collect();

        for (receiver, chunk_id, data) in &sends {
            let (start, _len) = ranges[*chunk_id];
            workers[*receiver].gradient[start..start + data.len()].copy_from_slice(data);
            log.push(RingStep {
                phase: Phase::Allgather,
                sender: (n + *receiver - 1) % n,
                receiver: *receiver,
                chunk_id: *chunk_id,
                bytes_transferred: data.len() * bytes_per_element,
            });
        }
    }

    // --- Verify correctness ------------------------------------------------
    let total_bytes: usize = log.iter().map(|s| s.bytes_transferred).sum();
    let final_gradient = workers[0].gradient.clone();
    let correct = final_gradient
        .iter()
        .zip(expected_sum.iter())
        .all(|(a, b)| (a - b).abs() < 1e-9);

    let result = AllreduceResult {
        final_gradient,
        total_bytes,
        n_steps: log.len(),
        correct,
    };

    (log, result)
}

// ---------------------------------------------------------------------------
// Naive allreduce (for comparison)
// ---------------------------------------------------------------------------

/// Naive allreduce: every worker broadcasts its entire gradient to every other
/// worker.  Returns total bytes transferred.
fn naive_allreduce_cost(n_workers: usize, grad_len: usize) -> usize {
    let bytes_per_element = size_of::<f64>();
    // Each worker sends to N-1 others, full gradient each time
    n_workers * (n_workers - 1) * grad_len * bytes_per_element
}

/// Ring allreduce theoretical cost.
fn ring_allreduce_cost(n_workers: usize, grad_len: usize) -> usize {
    let bytes_per_element = size_of::<f64>();
    // 2*(N-1) messages, each of size M/N
    2 * (n_workers - 1) * grad_len / n_workers * bytes_per_element
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{bytes} B")
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("distributed_ring_allreduce")?;

    println!("=== Ring-Allreduce for Distributed Gradient Aggregation ===\n");

    // =========================================================================
    // Section 1: Configuration
    // =========================================================================
    let n_workers = 4;
    let grad_len = 1024;

    println!("1. Configuration");
    println!("   ─────────────────────────────────────────");
    println!("   Workers:          {n_workers}");
    println!("   Gradient length:  {grad_len} elements (f64)");
    println!(
        "   Gradient size:    {}",
        format_bytes(grad_len * size_of::<f64>())
    );
    println!();

    // =========================================================================
    // Section 2: Initialize Workers
    // =========================================================================
    println!("2. Worker Initialization (deterministic RNG)");
    println!("   ─────────────────────────────────────────");

    let mut workers: Vec<WorkerNode> = (0..n_workers)
        .map(|id| {
            let gradient: Vec<f64> = (0..grad_len)
                .map(|_| ctx.rng().gen_range(-1.0..1.0))
                .collect();
            let buffer = vec![0.0; grad_len];
            WorkerNode {
                id,
                gradient,
                buffer,
            }
        })
        .collect();

    // Compute expected sum for verification
    let mut expected_sum = vec![0.0f64; grad_len];
    for w in &workers {
        for (acc, val) in expected_sum.iter_mut().zip(w.gradient.iter()) {
            *acc += *val;
        }
    }

    for w in &workers {
        let norm: f64 = w.gradient.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!(
            "   Worker {}: gradient L2-norm = {:.6}, first 3 = [{:.4}, {:.4}, {:.4}]",
            w.id, norm, w.gradient[0], w.gradient[1], w.gradient[2]
        );
    }
    println!();

    // =========================================================================
    // Section 3: Ring-Allreduce Execution
    // =========================================================================
    println!("3. Ring-Allreduce Execution");
    println!("   ─────────────────────────────────────────");

    let (log, result) = ring_allreduce(&mut workers, &expected_sum);

    println!("   ┌──────┬────────────────┬────────┬──────────┬─────────┬───────────┐");
    println!("   │ Step │ Phase          │ Sender │ Receiver │ Chunk   │ Bytes     │");
    println!("   ├──────┼────────────────┼────────┼──────────┼─────────┼───────────┤");
    for (i, step) in log.iter().enumerate() {
        println!(
            "   │ {:>4} │ {:14} │ {:>6} │ {:>8} │ {:>7} │ {:>9} │",
            i,
            step.phase,
            step.sender,
            step.receiver,
            step.chunk_id,
            format_bytes(step.bytes_transferred),
        );
    }
    println!("   └──────┴────────────────┴────────┴──────────┴─────────┴───────────┘");
    println!();

    // =========================================================================
    // Section 4: Verification
    // =========================================================================
    println!("4. Correctness Verification");
    println!("   ─────────────────────────────────────────");
    println!("   Total steps:    {}", result.n_steps);
    println!("   Total bytes:    {}", format_bytes(result.total_bytes));
    println!(
        "   Correct:        {}",
        if result.correct { "YES" } else { "NO" }
    );

    // Show first few elements of the reduced gradient vs expected
    println!(
        "   Reduced[0..3]:  [{:.6}, {:.6}, {:.6}]",
        result.final_gradient[0], result.final_gradient[1], result.final_gradient[2]
    );
    println!(
        "   Expected[0..3]: [{:.6}, {:.6}, {:.6}]",
        expected_sum[0], expected_sum[1], expected_sum[2]
    );

    // Verify all workers converged
    let all_workers_correct = workers.iter().all(|w| {
        w.gradient
            .iter()
            .zip(expected_sum.iter())
            .all(|(a, b)| (a - b).abs() < 1e-9)
    });
    println!(
        "   All workers converged: {}",
        if all_workers_correct { "YES" } else { "NO" }
    );
    println!();

    // =========================================================================
    // Section 5: Communication Cost Comparison
    // =========================================================================
    println!("5. Communication Cost: Ring vs Naive");
    println!("   ─────────────────────────────────────────");

    let ring_cost = ring_allreduce_cost(n_workers, grad_len);
    let naive_cost = naive_allreduce_cost(n_workers, grad_len);
    let ring_msgs = 2 * (n_workers - 1);
    let naive_msgs = n_workers * (n_workers - 1);

    println!("   ┌──────────────┬──────────────┬──────────────────┐");
    println!("   │ Strategy     │ Messages     │ Total Bytes      │");
    println!("   ├──────────────┼──────────────┼──────────────────┤");
    println!(
        "   │ {:12} │ {:>12} │ {:>16} │",
        "Ring",
        ring_msgs,
        format_bytes(ring_cost)
    );
    println!(
        "   │ {:12} │ {:>12} │ {:>16} │",
        "Naive",
        naive_msgs,
        format_bytes(naive_cost)
    );
    println!("   └──────────────┴──────────────┴──────────────────┘");

    let savings = if naive_cost > 0 {
        (1.0 - ring_cost as f64 / naive_cost as f64) * 100.0
    } else {
        0.0
    };
    println!("   Bandwidth savings: {savings:.1}%");
    println!(
        "   Ring efficiency:   {:.2}x fewer bytes than naive",
        naive_cost as f64 / ring_cost.max(1) as f64
    );
    println!();

    // =========================================================================
    // Section 6: Bandwidth Utilization Scaling
    // =========================================================================
    println!("6. Bandwidth Utilization Scaling");
    println!("   ─────────────────────────────────────────");
    println!("   ┌──────────┬─────────────────┬─────────────────┬───────────┐");
    println!("   │ Workers  │ Ring Bytes       │ Naive Bytes     │ Savings   │");
    println!("   ├──────────┼─────────────────┼─────────────────┼───────────┤");

    for nw in [2, 4, 8, 16, 32] {
        let rc = ring_allreduce_cost(nw, grad_len);
        let nc = naive_allreduce_cost(nw, grad_len);
        let pct = if nc > 0 {
            (1.0 - rc as f64 / nc as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "   │ {:>8} │ {:>15} │ {:>15} │ {:>8.1}% │",
            nw,
            format_bytes(rc),
            format_bytes(nc),
            pct
        );
    }
    println!("   └──────────┴─────────────────┴─────────────────┴───────────┘");
    println!();

    // =========================================================================
    // Section 7: Record metrics
    // =========================================================================
    ctx.record_metric("n_workers", n_workers as i64);
    ctx.record_metric("grad_len", grad_len as i64);
    ctx.record_metric("total_steps", result.n_steps as i64);
    ctx.record_metric("total_bytes_ring", result.total_bytes as i64);
    ctx.record_metric("total_bytes_naive", naive_cost as i64);
    ctx.record_float_metric("bandwidth_savings_pct", savings);
    ctx.record_string_metric("correct", if result.correct { "yes" } else { "no" });

    ctx.report()?;

    println!("\n=== Example Complete ===");
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_workers(n: usize, grad_len: usize, ctx: &mut RecipeContext) -> Vec<WorkerNode> {
        (0..n)
            .map(|id| {
                let gradient: Vec<f64> = (0..grad_len)
                    .map(|_| ctx.rng().gen_range(-1.0..1.0))
                    .collect();
                WorkerNode {
                    id,
                    gradient,
                    buffer: vec![0.0; grad_len],
                }
            })
            .collect()
    }

    fn expected_gradient_sum(workers: &[WorkerNode]) -> Vec<f64> {
        let grad_len = workers[0].gradient.len();
        let mut sum = vec![0.0f64; grad_len];
        for w in workers {
            for (acc, val) in sum.iter_mut().zip(w.gradient.iter()) {
                *acc += *val;
            }
        }
        sum
    }

    #[test]
    fn test_chunk_ranges_even_split() {
        let ranges = chunk_ranges(8, 4);
        assert_eq!(ranges.len(), 4);
        assert_eq!(ranges, vec![(0, 2), (2, 2), (4, 2), (6, 2)]);
    }

    #[test]
    fn test_chunk_ranges_uneven_split() {
        let ranges = chunk_ranges(10, 3);
        // 10 / 3 = 3 remainder 1 => first chunk gets 4, rest get 3
        assert_eq!(ranges.len(), 3);
        let total: usize = ranges.iter().map(|(_, len)| len).sum();
        assert_eq!(total, 10);
        assert_eq!(ranges[0], (0, 4));
        assert_eq!(ranges[1], (4, 3));
        assert_eq!(ranges[2], (7, 3));
    }

    #[test]
    fn test_ring_allreduce_correctness_4_workers() {
        let mut ctx = RecipeContext::new("test_ring_4").expect("context");
        let mut workers = make_workers(4, 1024, &mut ctx);
        let expected = expected_gradient_sum(&workers);

        let (_log, result) = ring_allreduce(&mut workers, &expected);

        assert!(result.correct, "Allreduce result must match expected sum");
    }

    #[test]
    fn test_ring_allreduce_correctness_2_workers() {
        let mut ctx = RecipeContext::new("test_ring_2").expect("context");
        let mut workers = make_workers(2, 64, &mut ctx);
        let expected = expected_gradient_sum(&workers);

        let (_log, result) = ring_allreduce(&mut workers, &expected);

        assert!(result.correct, "Allreduce with 2 workers must be correct");
    }

    #[test]
    fn test_ring_allreduce_all_workers_converge() {
        let mut ctx = RecipeContext::new("test_ring_converge").expect("context");
        let mut workers = make_workers(4, 256, &mut ctx);
        let expected = expected_gradient_sum(&workers);

        let _ = ring_allreduce(&mut workers, &expected);

        for w in &workers {
            for (a, b) in w.gradient.iter().zip(expected.iter()) {
                assert!(
                    (a - b).abs() < 1e-9,
                    "Worker {} gradient diverged from expected",
                    w.id
                );
            }
        }
    }

    #[test]
    fn test_ring_allreduce_step_count() {
        let mut ctx = RecipeContext::new("test_ring_steps").expect("context");
        let n = 4;
        let mut workers = make_workers(n, 128, &mut ctx);
        let expected = expected_gradient_sum(&workers);

        let (log, _result) = ring_allreduce(&mut workers, &expected);

        // Each phase has N*(N-1) individual send/receive entries (N workers, N-1 steps)
        let scatter_steps = log
            .iter()
            .filter(|s| s.phase == Phase::ScatterReduce)
            .count();
        let gather_steps = log.iter().filter(|s| s.phase == Phase::Allgather).count();

        assert_eq!(scatter_steps, n * (n - 1));
        assert_eq!(gather_steps, n * (n - 1));
    }

    #[test]
    fn test_naive_allreduce_cost_formula() {
        let cost = naive_allreduce_cost(4, 1024);
        let expected = 4 * 3 * 1024 * size_of::<f64>();
        assert_eq!(cost, expected);
    }

    #[test]
    fn test_ring_allreduce_cost_formula() {
        let cost = ring_allreduce_cost(4, 1024);
        // 2*(4-1) * 1024/4 * 8 = 6 * 256 * 8 = 12288
        let expected = 2 * 3 * (1024 / 4) * size_of::<f64>();
        assert_eq!(cost, expected);
    }

    #[test]
    fn test_ring_cheaper_than_naive() {
        for nw in [2, 4, 8, 16] {
            let ring = ring_allreduce_cost(nw, 1024);
            let naive = naive_allreduce_cost(nw, 1024);
            assert!(
                ring < naive,
                "Ring ({ring}) must be cheaper than naive ({naive}) for {nw} workers"
            );
        }
    }

    #[test]
    fn test_format_bytes_display() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1_500), "1.50 KB");
        assert_eq!(format_bytes(2_500_000), "2.50 MB");
    }

    #[test]
    fn test_deterministic_execution() {
        let mut ctx1 = RecipeContext::new("test_deterministic_ring").expect("context");
        let mut workers1 = make_workers(4, 128, &mut ctx1);
        let expected1 = expected_gradient_sum(&workers1);
        let (_, result1) = ring_allreduce(&mut workers1, &expected1);

        let mut ctx2 = RecipeContext::new("test_deterministic_ring").expect("context");
        let mut workers2 = make_workers(4, 128, &mut ctx2);
        let expected2 = expected_gradient_sum(&workers2);
        let (_, result2) = ring_allreduce(&mut workers2, &expected2);

        assert_eq!(result1.final_gradient, result2.final_gradient);
        assert_eq!(result1.total_bytes, result2.total_bytes);
    }

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", Phase::ScatterReduce), "scatter-reduce");
        assert_eq!(format!("{}", Phase::Allgather), "allgather");
    }
}
