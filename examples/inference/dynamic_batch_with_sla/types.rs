#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// SLA Tier
// ---------------------------------------------------------------------------

/// SLA tier defining the latency budget for a request
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlaTier {
    // Hard real-time: 10 ms deadline
    RealTime,
    // Interactive: 50 ms deadline
    Interactive,
    // Batch / offline: 500 ms deadline
    Batch,
}

impl SlaTier {
    pub fn deadline_us(self) -> u64 {
        match self {
            Self::RealTime => 10_000,
            Self::Interactive => 50_000,
            Self::Batch => 500_000,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::RealTime => "RealTime",
            Self::Interactive => "Interactive",
            Self::Batch => "Batch",
        }
    }

    pub fn all() -> &'static [SlaTier] {
        &[SlaTier::RealTime, SlaTier::Interactive, SlaTier::Batch]
    }
}

// ---------------------------------------------------------------------------
// Inference Request
// ---------------------------------------------------------------------------

/// A single inference request carrying an SLA contract
#[allow(dead_code)] // payload_size used in tests and display
#[derive(Clone)]
pub struct InferenceRequest {
    pub id: usize,
    pub sla_tier: SlaTier,
    pub arrival_time_us: u64,
    pub payload_size: usize,
    pub deadline_us: u64,
}

impl InferenceRequest {
    pub fn new(id: usize, sla_tier: SlaTier, arrival_time_us: u64, payload_size: usize) -> Self {
        Self {
            id,
            sla_tier,
            arrival_time_us,
            payload_size,
            deadline_us: arrival_time_us + sla_tier.deadline_us(),
        }
    }

    /// Microseconds remaining before the deadline at the given timestamp.
    pub fn slack_us(&self, now_us: u64) -> u64 {
        self.deadline_us.saturating_sub(now_us)
    }

    /// Whether the deadline has already passed.
    pub fn is_expired(&self, now_us: u64) -> bool {
        now_us >= self.deadline_us
    }
}

// ---------------------------------------------------------------------------
// Batch Decision
// ---------------------------------------------------------------------------

/// Outcome of the batch-size selection algorithm
pub struct BatchDecision {
    pub batch_size: usize,
    pub requests: Vec<InferenceRequest>,
    pub estimated_latency_us: u64,
}

// ---------------------------------------------------------------------------
// Completed Request (for metrics)
// ---------------------------------------------------------------------------

pub struct CompletedRequest {
    pub sla_tier: SlaTier,
    pub latency_us: u64,
    pub violated: bool,
}

// ---------------------------------------------------------------------------
// Batch Scheduler
// ---------------------------------------------------------------------------

/// Deadline-aware batch scheduler with priority queue semantics
pub struct BatchScheduler {
    pub queue: Vec<InferenceRequest>,
    pub max_batch_size: usize,
}

impl BatchScheduler {
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            queue: Vec::new(),
            max_batch_size,
        }
    }

    pub fn enqueue(&mut self, request: InferenceRequest) {
        self.queue.push(request);
    }

    #[allow(dead_code)] // used in tests
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Sort queue by deadline (earliest deadline first = highest priority).
    pub fn sort_by_priority(&mut self) {
        self.queue.sort_by_key(|r| r.deadline_us);
    }

    // Select batch size based on deadline pressure of the head-of-queue.
    //
    // Tighter deadlines produce smaller batches; loose deadlines allow larger
    /// batches for better throughput.
    pub fn select_batch_size(&self, now_us: u64) -> usize {
        if self.queue.is_empty() {
            return 0;
        }

        // Head-of-queue has the tightest deadline after sorting.
        let head_slack = self.queue[0].slack_us(now_us);

        // Heuristic: each additional request in the batch adds ~200 us of
        // processing overhead (simulated).
        let per_request_cost_us: u64 = 200;
        let base_cost_us: u64 = 500;

        // How many requests can we afford before the head request misses its SLA?
        let affordable = if head_slack > base_cost_us {
            ((head_slack - base_cost_us) / per_request_cost_us) as usize
        } else {
            1
        };

        affordable.clamp(1, self.max_batch_size.min(self.queue.len()))
    }

    // Drain up to `batch_size` non-expired requests from the front of the
    /// priority queue, skipping any that have already missed their deadline.
    pub fn take_batch(&mut self, now_us: u64) -> BatchDecision {
        self.sort_by_priority();
        let batch_size = self.select_batch_size(now_us);

        let mut requests = Vec::with_capacity(batch_size);
        let mut remaining = Vec::with_capacity(self.queue.len());

        for req in self.queue.drain(..) {
            if requests.len() < batch_size && !req.is_expired(now_us) {
                requests.push(req);
            } else {
                remaining.push(req);
            }
        }
        self.queue = remaining;

        let n = requests.len();
        let estimated_latency_us = estimate_batch_latency(n);

        BatchDecision {
            batch_size: n,
            requests,
            estimated_latency_us,
        }
    }
}

// ---------------------------------------------------------------------------
// Latency Model
// ---------------------------------------------------------------------------

// Simulate batch processing latency.
//
/// Base cost plus sub-linear scaling (batching amortises overhead).
pub fn estimate_batch_latency(batch_size: usize) -> u64 {
    if batch_size == 0 {
        return 0;
    }
    let base: u64 = 500;
    let per_item: u64 = 200;
    base + per_item * batch_size as u64
}

// ---------------------------------------------------------------------------
// Request Generator
// ---------------------------------------------------------------------------

/// Generate a deterministic stream of requests with varying SLA tiers.
pub fn generate_requests(n: usize, seed: u64) -> Vec<InferenceRequest> {
    let mut requests = Vec::with_capacity(n);
    let mut time_us: u64 = 0;

    for id in 0..n {
        // Deterministic inter-arrival time (100-600 us)
        let mut hasher = DefaultHasher::new();
        (seed, "arrival", id).hash(&mut hasher);
        let gap = 100 + (hasher.finish() % 500);
        time_us += gap;

        // Tier selection weighted: 20% RealTime, 50% Interactive, 30% Batch
        let mut h2 = DefaultHasher::new();
        (seed, "tier", id).hash(&mut h2);
        let tier_val = h2.finish() % 100;
        let tier = if tier_val < 20 {
            SlaTier::RealTime
        } else if tier_val < 70 {
            SlaTier::Interactive
        } else {
            SlaTier::Batch
        };

        // Payload size (32-256 elements)
        let mut h3 = DefaultHasher::new();
        (seed, "payload", id).hash(&mut h3);
        let payload_size = 32 + (h3.finish() % 225) as usize;

        requests.push(InferenceRequest::new(id, tier, time_us, payload_size));
    }
    requests
}

// ---------------------------------------------------------------------------
// SLA Report
// ---------------------------------------------------------------------------

/// Per-tier compliance report
pub struct SlaReport {
    pub tier: SlaTier,
    pub total_requests: usize,
    pub violations: usize,
    pub compliance_pct: f64,
    pub p50_latency_us: u64,
    pub p95_latency_us: u64,
    pub p99_latency_us: u64,
}

pub fn percentile(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 * pct / 100.0).ceil() as usize).saturating_sub(1);
    sorted[idx.min(sorted.len() - 1)]
}

pub fn build_sla_reports(completed: &[CompletedRequest]) -> Vec<SlaReport> {
    SlaTier::all()
        .iter()
        .filter_map(|&tier| {
            let tier_reqs: Vec<&CompletedRequest> =
                completed.iter().filter(|c| c.sla_tier == tier).collect();
            if tier_reqs.is_empty() {
                return None;
            }
            let total = tier_reqs.len();
            let violations = tier_reqs.iter().filter(|c| c.violated).count();
            let compliance_pct = (total - violations) as f64 / total as f64 * 100.0;

            let mut latencies: Vec<u64> = tier_reqs.iter().map(|c| c.latency_us).collect();
            latencies.sort_unstable();

            Some(SlaReport {
                tier,
                total_requests: total,
                violations,
                compliance_pct,
                p50_latency_us: percentile(&latencies, 50.0),
                p95_latency_us: percentile(&latencies, 95.0),
                p99_latency_us: percentile(&latencies, 99.0),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Throughput vs Latency Point
// ---------------------------------------------------------------------------

/// A single point on the throughput-latency Pareto curve
pub struct ThroughputLatencyPoint {
    pub batch_size: usize,
    pub throughput_rps: f64,
    pub avg_latency_us: f64,
}

/// Simulate fixed-batch-size serving for the throughput/latency curve.
pub fn measure_throughput_latency(
    requests: &[InferenceRequest],
    fixed_batch_size: usize,
) -> ThroughputLatencyPoint {
    let mut total_latency_us: u64 = 0;
    let mut processed: usize = 0;
    let mut time_us: u64 = 0;

    let mut idx = 0;
    while idx < requests.len() {
        let end = (idx + fixed_batch_size).min(requests.len());
        let batch = &requests[idx..end];
        let latency = estimate_batch_latency(batch.len());
        time_us += latency;

        for req in batch {
            let wait = time_us.saturating_sub(req.arrival_time_us);
            total_latency_us += wait;
            processed += 1;
        }
        idx = end;
    }

    let avg_latency_us = if processed > 0 {
        total_latency_us as f64 / processed as f64
    } else {
        0.0
    };
    let throughput_rps = if time_us > 0 {
        processed as f64 / (time_us as f64 / 1_000_000.0)
    } else {
        0.0
    };

    ThroughputLatencyPoint {
        batch_size: fixed_batch_size,
        throughput_rps,
        avg_latency_us,
    }
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

pub fn run_simulation(
    requests: Vec<InferenceRequest>,
    max_batch_size: usize,
) -> Vec<CompletedRequest> {
    let mut scheduler = BatchScheduler::new(max_batch_size);
    let mut completed: Vec<CompletedRequest> = Vec::new();
    let mut req_iter = requests.into_iter().peekable();
    let mut now_us: u64 = 0;

    loop {
        // Enqueue arriving requests
        while let Some(req) = req_iter.peek() {
            if req.arrival_time_us <= now_us {
                scheduler.enqueue(req_iter.next().unwrap());
            } else {
                break;
            }
        }

        if scheduler.is_empty() {
            if req_iter.peek().is_none() {
                break;
            }
            // Fast-forward to next arrival
            now_us = req_iter.peek().unwrap().arrival_time_us;
            continue;
        }

        let decision = scheduler.take_batch(now_us);
        if decision.batch_size == 0 {
            if req_iter.peek().is_none() {
                // Drain any remaining expired requests
                break;
            }
            now_us += 100; // advance time
            continue;
        }

        let finish_us = now_us + decision.estimated_latency_us;

        for req in decision.requests {
            let latency_us = finish_us.saturating_sub(req.arrival_time_us);
            let violated = finish_us > req.deadline_us;
            completed.push(CompletedRequest {
                sla_tier: req.sla_tier,
                latency_us,
                violated,
            });
        }

        now_us = finish_us;
    }

    completed
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
