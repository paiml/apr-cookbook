#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use super::types::*;

#[allow(unused_imports)]
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Format a percentile value for display (e.g. 99.9 -> "99.9", 50.0 -> "50").
pub fn format_percentile(p: f64) -> String {
    if (p - p.round()).abs() < 1e-9 {
        format!("{}", p as u64)
    } else {
        format!("{p:.1}")
    }
}

/// Compute the bucket index for a given latency value.
pub fn bucket_index(latency_us: u64) -> usize {
    let idx = latency_us / BUCKET_WIDTH_US;
    (idx as usize).min(NUM_BUCKETS - 1)
}

/// - `batch`:   highly variable (mean ~50ms)
pub fn simulate_latency(variant: usize, rng: &mut DeterministicRng, degradation: f64) -> u64 {
    let base_lambda = match variant {
        0 => 0.2,  // model_a: 1/0.2 = 5ms mean
        1 => 0.05, // model_b: 1/0.05 = 20ms mean
        _ => 0.02, // batch:   1/0.02 = 50ms mean
    };

    // Apply degradation factor (lower lambda = higher latency)
    let lambda = base_lambda / (1.0 + degradation);
    let latency_ms = rng.next_exponential(lambda);

    // Convert to microseconds and clamp
    let latency_us = (latency_ms * 1000.0) as u64;
    latency_us.min(MAX_LATENCY_US)
}

/// Define the standard SLO thresholds used across all sections.
pub fn define_slos() -> [Slo; 5] {
    [
        Slo {
            percentile: 50.0,
            threshold_us: 15_000,
        },
        Slo {
            percentile: 90.0,
            threshold_us: 50_000,
        },
        Slo {
            percentile: 95.0,
            threshold_us: 80_000,
        },
        Slo {
            percentile: 99.0,
            threshold_us: 100_000,
        },
        Slo {
            percentile: 99.9,
            threshold_us: 150_000,
        },
    ]
}

/// and percentiles.
pub fn run_histogram_tracking(rng: &mut DeterministicRng) -> LatencyHistogram {
    println!("1. Histogram-Based Latency Tracking");
    println!("   -----------------------------------------------");

    let mut global_histogram = LatencyHistogram::new();

    let total_requests = 1000;
    for i in 0..total_requests {
        let variant = i % NUM_VARIANTS;
        let latency = simulate_latency(variant, rng, 0.0);
        global_histogram.record(latency);
    }

    println!("   Total requests: {}", global_histogram.count());
    println!("   Mean latency:   {}ms", global_histogram.mean_us() / 1000);
    println!("   Min latency:    {}ms", global_histogram.min_us / 1000);
    println!("   Max latency:    {}ms", global_histogram.max_us / 1000);
    println!();

    println!("   Percentiles:");
    for &p in &PERCENTILES {
        let val = global_histogram.percentile(p);
        println!("     p{:>5}: {}ms", format_percentile(p), val / 1000);
    }
    println!();

    global_histogram
}

/// Checks the global histogram against defined SLOs and reports compliance.
pub fn run_slo_compliance(histogram: &LatencyHistogram, slos: &[Slo]) {
    println!("2. SLO Compliance Monitoring");
    println!("   -----------------------------------------------");

    println!("   Defined SLOs:");
    for slo in slos {
        println!("     {slo}");
    }
    println!();

    let results = histogram.check_slos(slos);
    let pass_count = results.iter().filter(|r| r.passed).count();
    let total_slos = results.len();

    println!("   SLO Check Results:");
    for result in &results {
        println!("     {result}");
    }
    println!();
    println!(
        "   Compliance: {}/{} SLOs passing ({:.1}%)",
        pass_count,
        total_slos,
        pass_count as f64 / total_slos as f64 * 100.0
    );
    println!();
}

/// percentiles, SLO violations, and the overall p99 trend.
pub fn run_rolling_windows(rng: &mut DeterministicRng, slos: &[Slo]) {
    println!("3. Rolling Window Percentiles Over Time");
    println!("   -----------------------------------------------");

    let mut window_tracker = RollingWindowTracker::new();

    for window_id in 0..NUM_WINDOWS {
        let degradation = window_id as f64 * 0.3;
        let mut window_hist = LatencyHistogram::new();

        for i in 0..REQUESTS_PER_WINDOW {
            let variant = i % NUM_VARIANTS;
            let latency = simulate_latency(variant, rng, degradation);
            window_hist.record(latency);
        }

        window_tracker.record_window(&window_hist);

        let p50 = window_hist.percentile(50.0) / 1000;
        let p99 = window_hist.percentile(99.0) / 1000;
        let slo_results = window_hist.check_slos(slos);
        let violations = slo_results.iter().filter(|r| !r.passed).count();

        println!(
            "   Window {}: p50={}ms, p99={}ms, SLO violations={}",
            window_id, p50, p99, violations
        );
    }

    let trend = window_tracker.p99_trend();
    let trend_label = match trend.cmp(&0) {
        std::cmp::Ordering::Greater => "DEGRADING",
        std::cmp::Ordering::Less => "IMPROVING",
        std::cmp::Ordering::Equal => "STABLE",
    };
    println!();
    println!(
        "   p99 trend over {} windows: {:+}ms ({})",
        window_tracker.window_count(),
        trend / 1000,
        trend_label
    );
    println!();
}

/// and returns the merged aggregate histogram.
pub fn run_variant_breakdown(rng: &mut DeterministicRng) -> LatencyHistogram {
    println!("4. Latency Breakdown by Model Variant");
    println!("   -----------------------------------------------");

    let mut variant_histograms: [LatencyHistogram; NUM_VARIANTS] = [
        LatencyHistogram::new(),
        LatencyHistogram::new(),
        LatencyHistogram::new(),
    ];

    for _ in 0..500 {
        for (v, hist) in variant_histograms.iter_mut().enumerate() {
            let latency = simulate_latency(v, rng, 0.0);
            hist.record(latency);
        }
    }

    for (i, name) in VARIANT_NAMES.iter().enumerate() {
        let hist = &variant_histograms[i];
        println!(
            "   {:>8}: n={:<5} mean={:>4}ms  p50={:>4}ms  p95={:>5}ms  p99={:>5}ms",
            name,
            hist.count(),
            hist.mean_us() / 1000,
            hist.percentile(50.0) / 1000,
            hist.percentile(95.0) / 1000,
            hist.percentile(99.0) / 1000,
        );
    }

    let mut aggregate = LatencyHistogram::new();
    for hist in &variant_histograms {
        aggregate.merge(hist);
    }
    println!(
        "   {:>8}: n={:<5} mean={:>4}ms  p50={:>4}ms  p95={:>5}ms  p99={:>5}ms",
        "TOTAL",
        aggregate.count(),
        aggregate.mean_us() / 1000,
        aggregate.percentile(50.0) / 1000,
        aggregate.percentile(95.0) / 1000,
        aggregate.percentile(99.0) / 1000,
    );
    println!();

    aggregate
}

/// Renders the aggregate histogram as an ASCII bar chart.
pub fn run_ascii_visualization(aggregate: &LatencyHistogram) {
    println!("5. ASCII Histogram Visualization");
    println!("   -----------------------------------------------");
    println!(
        "   Aggregate latency distribution ({} requests):",
        aggregate.count()
    );
    println!();
    print!("{}", aggregate.ascii_histogram());
    println!();
}

/// individual SLO violations and total violation count.
pub fn run_slo_alerts(slos: &[Slo]) {
    println!("6. SLO Violation Alerts");
    println!("   -----------------------------------------------");

    let mut total_violations = 0_usize;
    let mut rng_alerts = DeterministicRng::new(99);

    for window_id in 0..NUM_WINDOWS {
        let degradation = window_id as f64 * 0.3;
        let mut window_hist = LatencyHistogram::new();

        for i in 0..REQUESTS_PER_WINDOW {
            let variant = i % NUM_VARIANTS;
            let latency = simulate_latency(variant, &mut rng_alerts, degradation);
            window_hist.record(latency);
        }

        let slo_results = window_hist.check_slos(slos);
        let violations: Vec<&SloCheckResult> = slo_results.iter().filter(|r| !r.passed).collect();

        if violations.is_empty() {
            println!("   Window {}: All SLOs passing", window_id);
        } else {
            for violation in &violations {
                println!("   Window {}: ALERT - {}", window_id, violation);
                total_violations += 1;
            }
        }
    }

    println!();
    println!("   Total SLO violations across all windows: {total_violations}");

    if total_violations > 0 {
        println!("   Recommendation: Investigate latency regression in recent deployments");
    } else {
        println!("   All windows within SLO bounds");
    }
}
