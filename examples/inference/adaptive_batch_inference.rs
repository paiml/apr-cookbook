//! Adaptive Batch Inference Example
//!
//! Demonstrates dynamic request batching for model inference: accumulates
//! individual requests until a batch size or timeout threshold, then runs
//! a single vectorized forward pass. Measures throughput vs latency tradeoff.
//!
//! # Batching Strategy
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  Requests arrive individually:  R1, R2, R3, ...                  │
//! │                                                                  │
//! │  Batch accumulator:                                              │
//! │    [R1, R2, R3] ─── batch_size=4? NO, timeout? YES ──► Forward  │
//! │    [R4, R5, R6, R7] ── batch_size=4? YES ──► Forward             │
//! │                                                                  │
//! │  Single batched forward amortizes per-request overhead           │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example adaptive_batch_inference
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Crankshaw, D. et al. (2017). *Clipper: A Low-Latency Online Prediction Serving System*. NSDI. arXiv:1612.03079

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const INPUT_DIM: usize = 64;
const OUTPUT_DIM: usize = 16;

/// A simple model that performs batched matrix multiplication
struct BatchModel {
    weights: Vec<f32>, // OUTPUT_DIM x INPUT_DIM
    bias: Vec<f32>,    // OUTPUT_DIM
}

impl BatchModel {
    fn new(seed: u64) -> Self {
        let weights: Vec<f32> = (0..OUTPUT_DIM * INPUT_DIM)
            .map(|i| {
                let mut hasher = DefaultHasher::new();
                (seed, "weight", i).hash(&mut hasher);
                (hasher.finish() as f32 / u64::MAX as f32 - 0.5) * 0.1
            })
            .collect();

        let bias: Vec<f32> = (0..OUTPUT_DIM)
            .map(|i| {
                let mut hasher = DefaultHasher::new();
                (seed, "bias", i).hash(&mut hasher);
                (hasher.finish() as f32 / u64::MAX as f32 - 0.5) * 0.01
            })
            .collect();

        Self { weights, bias }
    }

    /// Single inference: input[INPUT_DIM] -> output[OUTPUT_DIM]
    fn forward_single(&self, input: &[f32]) -> Vec<f32> {
        let mut output = self.bias.clone();
        for (o, out) in output.iter_mut().enumerate() {
            for (i, &inp) in input.iter().enumerate() {
                *out += self.weights[o * INPUT_DIM + i] * inp;
            }
        }
        output
    }

    /// Batched inference: inputs[batch_size][INPUT_DIM] -> outputs[batch_size][OUTPUT_DIM]
    /// Amortizes overhead: single kernel launch, better cache utilization.
    fn forward_batch(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Simulate batched matmul (in production, this would be a single SIMD/GPU call)
        inputs.iter().map(|inp| self.forward_single(inp)).collect()
    }
}

/// An inference request
#[allow(dead_code)] // id used in tests
struct InferenceRequest {
    id: usize,
    input: Vec<f32>,
    arrival_step: usize,
}

/// Result of processing a request
struct InferenceResult {
    latency_steps: usize, // Steps from arrival to completion
}

/// Batch accumulator with configurable max batch size and timeout
struct BatchAccumulator {
    max_batch_size: usize,
    max_wait_steps: usize,
    pending: Vec<InferenceRequest>,
    oldest_arrival: Option<usize>,
}

impl BatchAccumulator {
    fn new(max_batch_size: usize, max_wait_steps: usize) -> Self {
        Self {
            max_batch_size,
            max_wait_steps,
            pending: Vec::new(),
            oldest_arrival: None,
        }
    }

    fn add_request(&mut self, request: InferenceRequest) {
        if self.oldest_arrival.is_none() {
            self.oldest_arrival = Some(request.arrival_step);
        }
        self.pending.push(request);
    }

    fn should_flush(&self, current_step: usize) -> bool {
        if self.pending.len() >= self.max_batch_size {
            return true;
        }
        if let Some(oldest) = self.oldest_arrival {
            if current_step - oldest >= self.max_wait_steps {
                return true;
            }
        }
        false
    }

    fn flush(&mut self) -> Vec<InferenceRequest> {
        self.oldest_arrival = None;
        std::mem::take(&mut self.pending)
    }

    fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

/// Generate synthetic requests with deterministic arrival pattern
fn generate_requests(n_requests: usize, seed: u64) -> Vec<(usize, InferenceRequest)> {
    let mut requests = Vec::with_capacity(n_requests);
    let mut step = 0;

    for id in 0..n_requests {
        // Deterministic inter-arrival time
        let mut hasher = DefaultHasher::new();
        (seed, "arrival", id).hash(&mut hasher);
        let gap = (hasher.finish() % 3) as usize; // 0-2 steps between arrivals
        step += gap;

        let input: Vec<f32> = (0..INPUT_DIM)
            .map(|i| {
                let mut h = DefaultHasher::new();
                (seed, "input", id, i).hash(&mut h);
                h.finish() as f32 / u64::MAX as f32 - 0.5
            })
            .collect();

        requests.push((
            step,
            InferenceRequest {
                id,
                input,
                arrival_step: step,
            },
        ));
    }

    requests
}

/// Compute latency statistics from results
fn compute_latency_stats(results: &[InferenceResult]) -> (f64, usize) {
    let latencies: Vec<usize> = results.iter().map(|r| r.latency_steps).collect();
    if latencies.is_empty() {
        return (0.0, 0);
    }
    let avg = latencies.iter().sum::<usize>() as f64 / latencies.len() as f64;
    let mut sorted = latencies;
    sorted.sort_unstable();
    let p99 = sorted[sorted.len() * 99 / 100];
    (avg, p99)
}

/// Process a single batch: run forward pass and record results
fn process_batch(
    model: &BatchModel,
    batch: Vec<InferenceRequest>,
    step: usize,
    results: &mut Vec<InferenceResult>,
) {
    let inputs: Vec<Vec<f32>> = batch.iter().map(|r| r.input.clone()).collect();
    let _ = model.forward_batch(&inputs);
    for req in batch {
        results.push(InferenceResult {
            latency_steps: step - req.arrival_step,
        });
    }
}

/// Run simulation with given batch configuration
fn run_simulation(
    model: &BatchModel,
    requests: Vec<(usize, InferenceRequest)>,
    max_batch_size: usize,
    max_wait_steps: usize,
) -> SimulationStats {
    let mut accumulator = BatchAccumulator::new(max_batch_size, max_wait_steps);
    let mut results: Vec<InferenceResult> = Vec::new();
    let mut request_iter = requests.into_iter().peekable();
    let mut step = 0;
    let mut n_batches = 0;
    let mut total_batch_size = 0;

    loop {
        // Add any arriving requests at this step
        while let Some(&(arrival, _)) = request_iter.peek() {
            if arrival <= step {
                let (_, req) = request_iter.next().unwrap();
                accumulator.add_request(req);
            } else {
                break;
            }
        }

        // Check if we should process a batch
        if accumulator.should_flush(step)
            || (request_iter.peek().is_none() && !accumulator.is_empty())
        {
            let batch = accumulator.flush();
            total_batch_size += batch.len();
            n_batches += 1;
            process_batch(model, batch, step, &mut results);
        }

        if request_iter.peek().is_none() && accumulator.is_empty() {
            break;
        }
        step += 1;
    }

    let (avg_latency, p99_latency) = compute_latency_stats(&results);

    SimulationStats {
        n_requests: results.len(),
        n_batches,
        avg_batch_size: if n_batches > 0 {
            total_batch_size as f64 / n_batches as f64
        } else {
            0.0
        },
        forward_calls: n_batches,
        avg_latency,
        p99_latency,
        throughput: if step > 0 {
            results.len() as f64 / step as f64
        } else {
            0.0
        },
    }
}

struct SimulationStats {
    n_requests: usize,
    n_batches: usize,
    avg_batch_size: f64,
    forward_calls: usize,
    avg_latency: f64,
    p99_latency: usize,
    throughput: f64,
}

fn main() {
    println!("=== Adaptive Batch Inference Example ===\n");

    let model = BatchModel::new(42);
    let n_requests = 200;

    // =========================================================================
    // Section 1: No Batching (batch_size=1)
    // =========================================================================
    println!("1. Baseline: No Batching");
    println!("   ─────────────────────────────────────────");

    let requests = generate_requests(n_requests, 42);
    let stats = run_simulation(&model, requests, 1, 0);

    println!("   Requests:     {}", stats.n_requests);
    println!("   Batches:      {}", stats.n_batches);
    println!("   Forward calls: {}", stats.forward_calls);
    println!("   Avg latency:  {:.1} steps", stats.avg_latency);
    println!("   Throughput:   {:.3} req/step", stats.throughput);
    println!();

    // =========================================================================
    // Section 2: Batch Size Sweep
    // =========================================================================
    println!("2. Batch Size Sweep (timeout=3)");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>6} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Batch", "Batches", "Avg Size", "Forwards", "Latency", "Throughput"
    );
    println!("   {}", "─".repeat(60));

    for batch_size in [1, 2, 4, 8, 16, 32] {
        let requests = generate_requests(n_requests, 42);
        let stats = run_simulation(&model, requests, batch_size, 3);
        println!(
            "   {:>6} {:>8} {:>10.1} {:>10} {:>10.1} {:>10.3}",
            batch_size,
            stats.n_batches,
            stats.avg_batch_size,
            stats.forward_calls,
            stats.avg_latency,
            stats.throughput
        );
    }
    println!();

    // =========================================================================
    // Section 3: Timeout Sweep
    // =========================================================================
    println!("3. Timeout Sweep (batch_size=8)");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>8} {:>8} {:>10} {:>10} {:>8} {:>10}",
        "Timeout", "Batches", "Avg Size", "AvgLat", "P99Lat", "Throughput"
    );
    println!("   {}", "─".repeat(58));

    for timeout in [0, 1, 2, 4, 8, 16] {
        let requests = generate_requests(n_requests, 42);
        let stats = run_simulation(&model, requests, 8, timeout);
        println!(
            "   {:>8} {:>8} {:>10.1} {:>10.1} {:>8} {:>10.3}",
            timeout,
            stats.n_batches,
            stats.avg_batch_size,
            stats.avg_latency,
            stats.p99_latency,
            stats.throughput
        );
    }
    println!();

    // =========================================================================
    // Section 4: Load Impact
    // =========================================================================
    println!("4. Load Impact (batch_size=8, timeout=3)");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Requests", "Batches", "AvgBatch", "Latency", "Throughput"
    );
    println!("   {}", "─".repeat(52));

    for n in [50, 100, 200, 500, 1000] {
        let requests = generate_requests(n, 42);
        let stats = run_simulation(&model, requests, 8, 3);
        println!(
            "   {:>10} {:>10} {:>10.1} {:>10.1} {:>10.3}",
            n, stats.n_batches, stats.avg_batch_size, stats.avg_latency, stats.throughput
        );
    }
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_model_single() {
        let model = BatchModel::new(42);
        let input = vec![0.1f32; INPUT_DIM];
        let output = model.forward_single(&input);
        assert_eq!(output.len(), OUTPUT_DIM);
    }

    #[test]
    fn test_batch_model_deterministic() {
        let model = BatchModel::new(42);
        let input = vec![0.5f32; INPUT_DIM];
        let out1 = model.forward_single(&input);
        let out2 = model.forward_single(&input);
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_batch_forward_matches_single() {
        let model = BatchModel::new(42);
        let inputs: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32 * 0.1; INPUT_DIM]).collect();

        let singles: Vec<Vec<f32>> = inputs.iter().map(|inp| model.forward_single(inp)).collect();
        let batch = model.forward_batch(&inputs);

        for (s, b) in singles.iter().zip(batch.iter()) {
            for (a, c) in s.iter().zip(b.iter()) {
                assert!((a - c).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_accumulator_flush_by_size() {
        let mut acc = BatchAccumulator::new(4, 100);
        for i in 0..4 {
            acc.add_request(InferenceRequest {
                id: i,
                input: vec![0.0; INPUT_DIM],
                arrival_step: i,
            });
        }
        assert!(acc.should_flush(0));
        let batch = acc.flush();
        assert_eq!(batch.len(), 4);
        assert!(acc.is_empty());
    }

    #[test]
    fn test_accumulator_flush_by_timeout() {
        let mut acc = BatchAccumulator::new(100, 3);
        acc.add_request(InferenceRequest {
            id: 0,
            input: vec![0.0; INPUT_DIM],
            arrival_step: 0,
        });
        assert!(!acc.should_flush(1));
        assert!(!acc.should_flush(2));
        assert!(acc.should_flush(3));
    }

    #[test]
    fn test_simulation_processes_all_requests() {
        let model = BatchModel::new(42);
        let requests = generate_requests(50, 42);
        let stats = run_simulation(&model, requests, 8, 3);
        assert_eq!(stats.n_requests, 50);
    }

    #[test]
    fn test_larger_batch_fewer_forwards() {
        let model = BatchModel::new(42);

        let requests1 = generate_requests(100, 42);
        let stats1 = run_simulation(&model, requests1, 1, 0);

        let requests8 = generate_requests(100, 42);
        let stats8 = run_simulation(&model, requests8, 8, 3);

        assert!(
            stats8.forward_calls <= stats1.forward_calls,
            "Batch=8 forwards {} should be <= batch=1 forwards {}",
            stats8.forward_calls,
            stats1.forward_calls
        );
    }

    #[test]
    fn test_generate_requests_deterministic() {
        let r1 = generate_requests(20, 42);
        let r2 = generate_requests(20, 42);
        assert_eq!(r1.len(), r2.len());
        for ((s1, req1), (s2, req2)) in r1.iter().zip(r2.iter()) {
            assert_eq!(s1, s2);
            assert_eq!(req1.id, req2.id);
            assert_eq!(req1.input, req2.input);
        }
    }
}
