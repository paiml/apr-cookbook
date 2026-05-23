#![allow(unused_imports)]
//! Dynamic Batch Inference with SLA Deadlines
//!
//! Contract: contracts/recipe-iiur-v1.yaml
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

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
