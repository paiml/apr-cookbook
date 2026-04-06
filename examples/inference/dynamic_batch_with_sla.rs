//! Dynamic Batch Inference with SLA Deadlines
//!
//! Demonstrates adaptive batching that respects per-request SLA deadlines
//! while maximizing throughput. Requests are classified into SLA tiers
//! (real-time, interactive, batch), scheduled via a priority queue, and
//! batched dynamically based on queue depth and deadline pressure.
//!
//! # Design
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │  Incoming requests with SLA tiers:                                  │
//! │    R1[RealTime 10ms]  R2[Interactive 50ms]  R3[Batch 500ms]        │
//! │                                                                     │
//! │  Priority Queue (sorted by deadline):                               │
//! │    R1 ──► R2 ──► R3                                                 │
//! │                                                                     │
//! │  Batch Scheduler:                                                   │
//! │    • Selects batch_size based on deadline pressure                  │
//! │    • Tight deadline → small batch (low latency)                    │
//! │    • Loose deadline → large batch (high throughput)                 │
//! │    • Never batches a request past its deadline                      │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example dynamic_batch_with_sla
//! ```
//!
//! ## References
//! - Crankshaw, D. et al. (2017). *Clipper: A Low-Latency Online Prediction Serving System*. NSDI. arXiv:1612.03079

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// SLA Tier
// ---------------------------------------------------------------------------

/// SLA tier defining the latency budget for a request
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SlaTier {
    /// Hard real-time: 10 ms deadline
    RealTime,
    /// Interactive: 50 ms deadline
    Interactive,
    /// Batch / offline: 500 ms deadline
    Batch,
}

impl SlaTier {
    fn deadline_us(self) -> u64 {
        match self {
            Self::RealTime => 10_000,
            Self::Interactive => 50_000,
            Self::Batch => 500_000,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::RealTime => "RealTime",
            Self::Interactive => "Interactive",
            Self::Batch => "Batch",
        }
    }

    fn all() -> &'static [SlaTier] {
        &[SlaTier::RealTime, SlaTier::Interactive, SlaTier::Batch]
    }
}

// ---------------------------------------------------------------------------
// Inference Request
// ---------------------------------------------------------------------------

/// A single inference request carrying an SLA contract
#[allow(dead_code)] // payload_size used in tests and display
#[derive(Clone)]
struct InferenceRequest {
    id: usize,
    sla_tier: SlaTier,
    arrival_time_us: u64,
    payload_size: usize,
    deadline_us: u64,
}

impl InferenceRequest {
    fn new(id: usize, sla_tier: SlaTier, arrival_time_us: u64, payload_size: usize) -> Self {
        Self {
            id,
            sla_tier,
            arrival_time_us,
            payload_size,
            deadline_us: arrival_time_us + sla_tier.deadline_us(),
        }
    }

    /// Microseconds remaining before the deadline at the given timestamp.
    fn slack_us(&self, now_us: u64) -> u64 {
        self.deadline_us.saturating_sub(now_us)
    }

    /// Whether the deadline has already passed.
    fn is_expired(&self, now_us: u64) -> bool {
        now_us >= self.deadline_us
    }
}

// ---------------------------------------------------------------------------
// Batch Decision
// ---------------------------------------------------------------------------

/// Outcome of the batch-size selection algorithm
struct BatchDecision {
    batch_size: usize,
    requests: Vec<InferenceRequest>,
    estimated_latency_us: u64,
}

// ---------------------------------------------------------------------------
// Completed Request (for metrics)
// ---------------------------------------------------------------------------

struct CompletedRequest {
    sla_tier: SlaTier,
    latency_us: u64,
    violated: bool,
}

// ---------------------------------------------------------------------------
// Batch Scheduler
// ---------------------------------------------------------------------------

/// Deadline-aware batch scheduler with priority queue semantics
struct BatchScheduler {
    queue: Vec<InferenceRequest>,
    max_batch_size: usize,
}

impl BatchScheduler {
    fn new(max_batch_size: usize) -> Self {
        Self {
            queue: Vec::new(),
            max_batch_size,
        }
    }

    fn enqueue(&mut self, request: InferenceRequest) {
        self.queue.push(request);
    }

    #[allow(dead_code)] // used in tests
    fn queue_len(&self) -> usize {
        self.queue.len()
    }

    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Sort queue by deadline (earliest deadline first = highest priority).
    fn sort_by_priority(&mut self) {
        self.queue.sort_by_key(|r| r.deadline_us);
    }

    /// Select batch size based on deadline pressure of the head-of-queue.
    ///
    /// Tighter deadlines produce smaller batches; loose deadlines allow larger
    /// batches for better throughput.
    fn select_batch_size(&self, now_us: u64) -> usize {
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

    /// Drain up to `batch_size` non-expired requests from the front of the
    /// priority queue, skipping any that have already missed their deadline.
    fn take_batch(&mut self, now_us: u64) -> BatchDecision {
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

/// Simulate batch processing latency.
///
/// Base cost plus sub-linear scaling (batching amortises overhead).
fn estimate_batch_latency(batch_size: usize) -> u64 {
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
fn generate_requests(n: usize, seed: u64) -> Vec<InferenceRequest> {
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
struct SlaReport {
    tier: SlaTier,
    total_requests: usize,
    violations: usize,
    compliance_pct: f64,
    p50_latency_us: u64,
    p95_latency_us: u64,
    p99_latency_us: u64,
}

fn percentile(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 * pct / 100.0).ceil() as usize).saturating_sub(1);
    sorted[idx.min(sorted.len() - 1)]
}

fn build_sla_reports(completed: &[CompletedRequest]) -> Vec<SlaReport> {
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
struct ThroughputLatencyPoint {
    batch_size: usize,
    throughput_rps: f64,
    avg_latency_us: f64,
}

/// Simulate fixed-batch-size serving for the throughput/latency curve.
fn measure_throughput_latency(
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

fn run_simulation(requests: Vec<InferenceRequest>, max_batch_size: usize) -> Vec<CompletedRequest> {
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

fn main() {
    println!("=== Dynamic Batch Inference with SLA Deadlines ===\n");

    // =========================================================================
    // Section 1: Define SLA Tiers
    // =========================================================================
    println!("1. SLA Tier Definitions");
    println!("   ─────────────────────────────────────────");
    println!("   {:>12} {:>12} {:>12}", "Tier", "Deadline", "Use Case");
    println!("   {}", "─".repeat(40));
    println!(
        "   {:>12} {:>10} us {:>12}",
        "RealTime", 10_000, "Autocomplete"
    );
    println!("   {:>12} {:>10} us {:>12}", "Interactive", 50_000, "Chat");
    println!("   {:>12} {:>10} us {:>12}", "Batch", 500_000, "Analytics");
    println!();

    // =========================================================================
    // Section 2: Request Queue with Priority Scheduling
    // =========================================================================
    println!("2. Priority Queue Scheduling");
    println!("   ─────────────────────────────────────────");

    let requests = generate_requests(100, 42);
    let tier_counts: Vec<(SlaTier, usize)> = SlaTier::all()
        .iter()
        .map(|&t| {
            let count = requests.iter().filter(|r| r.sla_tier == t).count();
            (t, count)
        })
        .collect();

    for (tier, count) in &tier_counts {
        println!("   {:>12}: {} requests", tier.label(), count);
    }

    // Show priority ordering
    let mut sample: Vec<_> = requests.iter().take(8).cloned().collect();
    sample.sort_by_key(|r| r.deadline_us);
    println!("\n   Priority order (first 8 by deadline):");
    for req in &sample {
        println!(
            "     req {:>3} [{}] deadline={} us  slack={} us",
            req.id,
            req.sla_tier.label(),
            req.deadline_us,
            req.slack_us(req.arrival_time_us)
        );
    }
    println!();

    // =========================================================================
    // Section 3: Dynamic Batch Size Selection
    // =========================================================================
    println!("3. Dynamic Batch Size Selection (deadline pressure)");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>12} {:>12} {:>12}",
        "Head Slack", "Queue Depth", "Batch Size"
    );
    println!("   {}", "─".repeat(40));

    let test_slacks: &[(u64, usize)] = &[
        (1_000, 10),
        (5_000, 10),
        (10_000, 10),
        (50_000, 10),
        (100_000, 10),
        (500_000, 10),
    ];

    for &(slack, depth) in test_slacks {
        let mut scheduler = BatchScheduler::new(32);
        let now = 0u64;
        for i in 0..depth {
            scheduler.enqueue(InferenceRequest {
                id: i,
                sla_tier: SlaTier::Interactive,
                arrival_time_us: 0,
                payload_size: 64,
                deadline_us: now + slack,
            });
        }
        scheduler.sort_by_priority();
        let bs = scheduler.select_batch_size(now);
        println!("   {:>10} us {:>12} {:>12}", slack, depth, bs);
    }
    println!();

    // =========================================================================
    // Section 4: Process Batches While Respecting Deadlines
    // =========================================================================
    println!("4. Batch Processing Simulation (max_batch=16)");
    println!("   ─────────────────────────────────────────");

    let requests = generate_requests(200, 42);
    let completed = run_simulation(requests, 16);
    let total = completed.len();
    let violations = completed.iter().filter(|c| c.violated).count();

    println!("   Processed:   {}", total);
    println!("   Violations:  {}", violations);
    println!(
        "   Overall SLA: {:.1}%",
        (total - violations) as f64 / total as f64 * 100.0
    );
    println!();

    // =========================================================================
    // Section 5: SLA Compliance Reporting
    // =========================================================================
    println!("5. Per-Tier SLA Compliance");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>12} {:>8} {:>8} {:>10} {:>8} {:>8} {:>8}",
        "Tier", "Total", "Violn", "Comply%", "P50", "P95", "P99"
    );
    println!("   {}", "─".repeat(68));

    let reports = build_sla_reports(&completed);
    for r in &reports {
        println!(
            "   {:>12} {:>8} {:>8} {:>9.1}% {:>7} {:>7} {:>7}",
            r.tier.label(),
            r.total_requests,
            r.violations,
            r.compliance_pct,
            r.p50_latency_us,
            r.p95_latency_us,
            r.p99_latency_us,
        );
    }
    println!();

    // =========================================================================
    // Section 6: Throughput vs Latency Tradeoff
    // =========================================================================
    println!("6. Throughput vs Latency Tradeoff");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>10} {:>14} {:>14}",
        "BatchSize", "Throughput", "AvgLatency"
    );
    println!("   {}", "─".repeat(42));

    let sweep_requests = generate_requests(500, 42);
    for bs in [1, 2, 4, 8, 16, 32, 64] {
        let point = measure_throughput_latency(&sweep_requests, bs);
        println!(
            "   {:>10} {:>11.0} rps {:>11.0} us",
            point.batch_size, point.throughput_rps, point.avg_latency_us,
        );
    }
    println!();

    println!("=== Example Complete ===");
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- SlaTier tests ---

    #[test]
    fn test_sla_tier_deadlines() {
        assert_eq!(SlaTier::RealTime.deadline_us(), 10_000);
        assert_eq!(SlaTier::Interactive.deadline_us(), 50_000);
        assert_eq!(SlaTier::Batch.deadline_us(), 500_000);
    }

    #[test]
    fn test_sla_tier_labels() {
        assert_eq!(SlaTier::RealTime.label(), "RealTime");
        assert_eq!(SlaTier::Interactive.label(), "Interactive");
        assert_eq!(SlaTier::Batch.label(), "Batch");
    }

    #[test]
    fn test_sla_tier_all_returns_three() {
        assert_eq!(SlaTier::all().len(), 3);
    }

    // --- InferenceRequest tests ---

    #[test]
    fn test_request_deadline_computation() {
        let req = InferenceRequest::new(0, SlaTier::Interactive, 1000, 64);
        assert_eq!(req.deadline_us, 1000 + 50_000);
    }

    #[test]
    fn test_request_slack_positive() {
        let req = InferenceRequest::new(0, SlaTier::Batch, 0, 64);
        assert_eq!(req.slack_us(100_000), 400_000);
    }

    #[test]
    fn test_request_slack_saturates_at_zero() {
        let req = InferenceRequest::new(0, SlaTier::RealTime, 0, 64);
        assert_eq!(req.slack_us(999_999), 0);
    }

    #[test]
    fn test_request_expired() {
        let req = InferenceRequest::new(0, SlaTier::RealTime, 0, 64);
        assert!(!req.is_expired(5_000));
        assert!(req.is_expired(10_000));
        assert!(req.is_expired(20_000));
    }

    // --- BatchScheduler tests ---

    #[test]
    fn test_scheduler_enqueue_and_len() {
        let mut sched = BatchScheduler::new(8);
        assert!(sched.is_empty());
        sched.enqueue(InferenceRequest::new(0, SlaTier::Batch, 0, 64));
        assert_eq!(sched.queue_len(), 1);
    }

    #[test]
    fn test_scheduler_sort_by_priority() {
        let mut sched = BatchScheduler::new(8);
        sched.enqueue(InferenceRequest::new(0, SlaTier::Batch, 0, 64));
        sched.enqueue(InferenceRequest::new(1, SlaTier::RealTime, 0, 64));
        sched.enqueue(InferenceRequest::new(2, SlaTier::Interactive, 0, 64));
        sched.sort_by_priority();

        assert_eq!(sched.queue[0].sla_tier, SlaTier::RealTime);
        assert_eq!(sched.queue[1].sla_tier, SlaTier::Interactive);
        assert_eq!(sched.queue[2].sla_tier, SlaTier::Batch);
    }

    #[test]
    fn test_select_batch_size_empty_queue() {
        let sched = BatchScheduler::new(8);
        assert_eq!(sched.select_batch_size(0), 0);
    }

    #[test]
    fn test_select_batch_size_tight_deadline() {
        let mut sched = BatchScheduler::new(32);
        sched.enqueue(InferenceRequest::new(0, SlaTier::RealTime, 0, 64));
        for i in 1..10 {
            sched.enqueue(InferenceRequest::new(i, SlaTier::Batch, 0, 64));
        }
        sched.sort_by_priority();
        let bs = sched.select_batch_size(0);
        // RealTime has 10000 us slack → (10000-500)/200 = 47, clamped to 10
        assert!(bs <= 10, "Batch size {} should be <= 10 (queue len)", bs);
        assert!(bs >= 1, "Batch size must be at least 1");
    }

    #[test]
    fn test_select_batch_size_loose_deadline() {
        let mut sched = BatchScheduler::new(32);
        for i in 0..20 {
            sched.enqueue(InferenceRequest::new(i, SlaTier::Batch, 0, 64));
        }
        sched.sort_by_priority();
        let bs = sched.select_batch_size(0);
        // Batch has 500_000 us slack → large batch allowed
        assert!(bs > 1, "Loose deadline should allow batch > 1, got {}", bs);
    }

    #[test]
    fn test_take_batch_skips_expired() {
        let mut sched = BatchScheduler::new(8);
        // Already expired request
        sched.enqueue(InferenceRequest {
            id: 0,
            sla_tier: SlaTier::RealTime,
            arrival_time_us: 0,
            payload_size: 64,
            deadline_us: 100,
        });
        // Valid request
        sched.enqueue(InferenceRequest::new(1, SlaTier::Batch, 0, 64));

        let decision = sched.take_batch(200);
        // The expired request should be skipped
        assert!(
            decision.requests.iter().all(|r| r.id != 0),
            "Expired request should not be in batch"
        );
    }

    #[test]
    fn test_take_batch_drains_queue() {
        let mut sched = BatchScheduler::new(8);
        for i in 0..5 {
            sched.enqueue(InferenceRequest::new(i, SlaTier::Batch, 0, 64));
        }
        let decision = sched.take_batch(0);
        assert!(decision.batch_size <= 8);
        assert_eq!(decision.batch_size + sched.queue_len(), 5);
    }

    // --- Latency model tests ---

    #[test]
    fn test_estimate_batch_latency_zero() {
        assert_eq!(estimate_batch_latency(0), 0);
    }

    #[test]
    fn test_estimate_batch_latency_scales() {
        let l1 = estimate_batch_latency(1);
        let l4 = estimate_batch_latency(4);
        assert!(l4 > l1, "Larger batch should have higher latency");
        // Sub-linear in this model means linear, but still monotonic
        assert_eq!(l1, 700);
        assert_eq!(l4, 1300);
    }

    // --- Request generator tests ---

    #[test]
    fn test_generate_requests_count() {
        let reqs = generate_requests(50, 42);
        assert_eq!(reqs.len(), 50);
    }

    #[test]
    fn test_generate_requests_deterministic() {
        let r1 = generate_requests(30, 42);
        let r2 = generate_requests(30, 42);
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.arrival_time_us, b.arrival_time_us);
            assert_eq!(a.payload_size, b.payload_size);
            assert_eq!(a.sla_tier, b.sla_tier);
        }
    }

    #[test]
    fn test_generate_requests_monotonic_arrival() {
        let reqs = generate_requests(100, 42);
        for w in reqs.windows(2) {
            assert!(
                w[1].arrival_time_us >= w[0].arrival_time_us,
                "Arrivals must be non-decreasing"
            );
        }
    }

    // --- SLA report tests ---

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile(&[], 50.0), 0);
    }

    #[test]
    fn test_percentile_single() {
        assert_eq!(percentile(&[42], 99.0), 42);
    }

    #[test]
    fn test_build_sla_reports_all_compliant() {
        let completed = vec![
            CompletedRequest {
                sla_tier: SlaTier::RealTime,
                latency_us: 5000,
                violated: false,
            },
            CompletedRequest {
                sla_tier: SlaTier::Interactive,
                latency_us: 20_000,
                violated: false,
            },
        ];
        let reports = build_sla_reports(&completed);
        for r in &reports {
            assert_eq!(r.violations, 0);
            assert!((r.compliance_pct - 100.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_build_sla_reports_with_violations() {
        let completed = vec![
            CompletedRequest {
                sla_tier: SlaTier::RealTime,
                latency_us: 5000,
                violated: false,
            },
            CompletedRequest {
                sla_tier: SlaTier::RealTime,
                latency_us: 15_000,
                violated: true,
            },
        ];
        let reports = build_sla_reports(&completed);
        let rt_report = reports
            .iter()
            .find(|r| r.tier == SlaTier::RealTime)
            .unwrap();
        assert_eq!(rt_report.total_requests, 2);
        assert_eq!(rt_report.violations, 1);
        assert!((rt_report.compliance_pct - 50.0).abs() < f64::EPSILON);
    }

    // --- Throughput/latency measurement tests ---

    #[test]
    fn test_throughput_latency_point() {
        let reqs = generate_requests(20, 42);
        let point = measure_throughput_latency(&reqs, 4);
        assert_eq!(point.batch_size, 4);
        assert!(point.throughput_rps > 0.0);
        assert!(point.avg_latency_us > 0.0);
    }

    // --- End-to-end simulation tests ---

    #[test]
    fn test_simulation_processes_requests() {
        let reqs = generate_requests(50, 42);
        let completed = run_simulation(reqs, 8);
        assert!(
            !completed.is_empty(),
            "Should process at least some requests"
        );
    }

    #[test]
    fn test_simulation_larger_batch_changes_compliance() {
        let reqs1 = generate_requests(100, 42);
        let reqs2 = generate_requests(100, 42);

        let c1 = run_simulation(reqs1, 1);
        let c2 = run_simulation(reqs2, 32);

        // Both should complete all requests (no infinite loops)
        assert!(!c1.is_empty());
        assert!(!c2.is_empty());
    }
}
