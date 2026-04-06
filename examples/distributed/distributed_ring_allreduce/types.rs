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
use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A simulated worker node in the ring.
#[derive(Debug, Clone)]
pub struct WorkerNode {
    pub id: usize,
    pub gradient: Vec<f64>,
    pub buffer: Vec<f64>,
}

/// One communication step in the ring-allreduce algorithm.
#[derive(Debug, Clone)]
pub struct RingStep {
    pub phase: Phase,
    pub sender: usize,
    pub receiver: usize,
    pub chunk_id: usize,
    pub bytes_transferred: usize,
}

/// Phase of the ring-allreduce algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
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
pub struct AllreduceResult {
    pub final_gradient: Vec<f64>,
    pub total_bytes: usize,
    pub n_steps: usize,
    pub correct: bool,
}

// ---------------------------------------------------------------------------
// Ring-Allreduce implementation
// ---------------------------------------------------------------------------

// Compute chunk boundaries for splitting a vector of `total` elements into
/// `n_chunks` roughly-equal pieces.  Returns `(start, len)` pairs.
pub fn chunk_ranges(total: usize, n_chunks: usize) -> Vec<(usize, usize)> {
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

// Execute the ring-allreduce algorithm across the given workers.
//
// Returns the communication log and the final allreduce result (verified
/// against the expected sum of all original local gradients).
pub fn ring_allreduce(
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

// Naive allreduce: every worker broadcasts its entire gradient to every other
/// worker.  Returns total bytes transferred.
pub fn naive_allreduce_cost(n_workers: usize, grad_len: usize) -> usize {
    let bytes_per_element = size_of::<f64>();
    // Each worker sends to N-1 others, full gradient each time
    n_workers * (n_workers - 1) * grad_len * bytes_per_element
}

/// Ring allreduce theoretical cost.
pub fn ring_allreduce_cost(n_workers: usize, grad_len: usize) -> usize {
    let bytes_per_element = size_of::<f64>();
    // 2*(N-1) messages, each of size M/N
    2 * (n_workers - 1) * grad_len / n_workers * bytes_per_element
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

pub fn format_bytes(bytes: usize) -> String {
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
