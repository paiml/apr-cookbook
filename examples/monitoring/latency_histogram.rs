//! Latency Percentile Tracking and SLO Monitoring for Production ML Inference
//!
//! Demonstrates histogram-based latency tracking with percentile computation,
//! SLO compliance monitoring, rolling window analysis, per-variant breakdown,
//! and ASCII histogram visualization -- all with zero external dependencies.
//!
//! # Techniques
//!
//! - **Histogram-based percentile tracking**: Fixed-bucket histograms for O(1)
//!   insertion and fast percentile queries (p50, p90, p95, p99, p99.9)
//! - **SLO compliance monitoring**: Configurable thresholds with violation
//!   detection and compliance percentage reporting
//! - **Rolling window percentiles**: Time-window analysis showing latency
//!   evolution across successive request batches
//! - **Per-variant breakdown**: Latency segmentation by model variant or
//!   request type for targeted optimization
//! - **ASCII histogram visualization**: Terminal-friendly bar charts of the
//!   latency distribution
//! - **Alert on SLO violations**: Automatic alerting when percentile latencies
//!   exceed configured SLO targets
//!
//! # Architecture
//!
//! ```text
//! +-------------------------------------------------------------------+
//! |               Latency Histogram Monitor                           |
//! +-------------------------------------------------------------------+
//! |                                                                   |
//! |  Requests -------> LatencyHistogram -------> Percentiles          |
//! |    (model_a)            |                    (p50/p90/p95/p99)    |
//! |    (model_b)            |                                         |
//! |    (batch)              v                                         |
//! |                   RollingWindow -------> Trend Detection          |
//! |                         |                                         |
//! |                         v                                         |
//! |                   SloMonitor ----------> Alerts                   |
//! |                         |                [PASS / VIOLATION]       |
//! |                         v                                         |
//! |                   ASCII Histogram                                 |
//! |                   [  ###       ]                                  |
//! |                   [  ######    ]                                  |
//! |                   [  #         ]                                  |
//! +-------------------------------------------------------------------+
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example latency_histogram
//! ```
//!
//! # Recipe Metadata
//!
//! - **Category**: Monitoring
//! - **Complexity**: Intermediate
//! - **Dependencies**: None (std only)
//! - **IIUR**: Isolated, Idempotent, Useful, Reproducible
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

// ============================================================================
// Constants
// ============================================================================

/// Number of fixed-width histogram buckets
const NUM_BUCKETS: usize = 20;

/// Maximum latency tracked (microseconds). Values above are clamped.
const MAX_LATENCY_US: u64 = 200_000;

/// Bucket width in microseconds
const BUCKET_WIDTH_US: u64 = MAX_LATENCY_US / NUM_BUCKETS as u64;

/// Number of model variants to simulate
const NUM_VARIANTS: usize = 3;

/// Variant labels
const VARIANT_NAMES: [&str; NUM_VARIANTS] = ["model_a", "model_b", "batch"];

/// Number of rolling windows to simulate
const NUM_WINDOWS: usize = 6;

/// Requests per rolling window
const REQUESTS_PER_WINDOW: usize = 200;

/// Standard percentiles to report
const PERCENTILES: [f64; 5] = [50.0, 90.0, 95.0, 99.0, 99.9];

/// Width of the ASCII bar chart (characters)
const BAR_WIDTH: usize = 50;

// ============================================================================
// Deterministic RNG
// ============================================================================

/// Deterministic pseudo-random number generator using `DefaultHasher`.
///
/// Produces repeatable sequences given the same seed, suitable for
/// simulation without pulling in external crate dependencies.
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance state and return a pseudo-random `u64`.
    fn next_u64(&mut self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.state.hash(&mut hasher);
        self.state = hasher.finish();
        self.state
    }

    /// Return a float in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Approximate an exponential distribution with rate parameter `lambda`.
    /// Returns -ln(U) / lambda where U is uniform in (0, 1).
    fn next_exponential(&mut self, lambda: f64) -> f64 {
        let u = self.next_f64().max(1e-15); // avoid ln(0)
        -(u.ln()) / lambda
    }
}

// ============================================================================
// SLO Definition
// ============================================================================

/// A Service Level Objective binding a percentile to a latency threshold.
#[derive(Debug, Clone, Copy)]
struct Slo {
    /// Percentile (e.g. 99.0 for p99)
    percentile: f64,
    /// Maximum acceptable latency in microseconds
    threshold_us: u64,
}

impl fmt::Display for Slo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "p{} < {}ms",
            format_percentile(self.percentile),
            self.threshold_us / 1000
        )
    }
}

/// Format a percentile value for display (e.g. 99.9 -> "99.9", 50.0 -> "50").
fn format_percentile(p: f64) -> String {
    if (p - p.round()).abs() < 1e-9 {
        format!("{}", p as u64)
    } else {
        format!("{p:.1}")
    }
}

// ============================================================================
// SLO Check Result
// ============================================================================

/// Result of checking a single SLO against measured latency.
#[derive(Debug, Clone, Copy)]
struct SloCheckResult {
    slo: Slo,
    actual_us: u64,
    passed: bool,
}

impl fmt::Display for SloCheckResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.passed { "PASS" } else { "VIOLATION" };
        write!(
            f,
            "[{}] {}: actual={}ms",
            status,
            self.slo,
            self.actual_us / 1000
        )
    }
}

// ============================================================================
// Latency Histogram
// ============================================================================

/// Fixed-bucket histogram for latency tracking.
///
/// Buckets cover `[0, MAX_LATENCY_US)` in equal-width intervals.
/// Values exceeding `MAX_LATENCY_US` are placed in the last bucket.
struct LatencyHistogram {
    /// Bucket counts; index `i` covers `[i * BUCKET_WIDTH_US, (i+1) * BUCKET_WIDTH_US)`
    buckets: [u64; NUM_BUCKETS],
    /// Total number of recorded observations
    count: u64,
    /// Running sum for mean calculation (microseconds)
    sum_us: u64,
    /// Minimum observed latency
    min_us: u64,
    /// Maximum observed latency
    max_us: u64,
}

impl LatencyHistogram {
    /// Create an empty histogram.
    fn new() -> Self {
        Self {
            buckets: [0; NUM_BUCKETS],
            count: 0,
            sum_us: 0,
            min_us: u64::MAX,
            max_us: 0,
        }
    }

    /// Record a latency observation in microseconds.
    fn record(&mut self, latency_us: u64) {
        let idx = bucket_index(latency_us);
        self.buckets[idx] += 1;
        self.count += 1;
        self.sum_us += latency_us;
        if latency_us < self.min_us {
            self.min_us = latency_us;
        }
        if latency_us > self.max_us {
            self.max_us = latency_us;
        }
    }

    /// Compute the approximate latency at a given percentile (0-100).
    ///
    /// Uses linear interpolation within the bucket that contains the target
    /// cumulative count.
    fn percentile(&self, p: f64) -> u64 {
        if self.count == 0 {
            return 0;
        }
        let target = (p / 100.0 * self.count as f64).ceil() as u64;
        let target = target.max(1).min(self.count);

        let mut cumulative: u64 = 0;
        for (i, &bucket_count) in self.buckets.iter().enumerate() {
            cumulative += bucket_count;
            if cumulative >= target {
                // Linear interpolation within this bucket
                let prev_cumulative = cumulative - bucket_count;
                let fraction = if bucket_count > 0 {
                    (target - prev_cumulative) as f64 / bucket_count as f64
                } else {
                    0.0
                };
                let low = i as u64 * BUCKET_WIDTH_US;
                let high = low + BUCKET_WIDTH_US;
                return low + (fraction * (high - low) as f64) as u64;
            }
        }
        MAX_LATENCY_US
    }

    /// Mean latency in microseconds.
    fn mean_us(&self) -> u64 {
        if self.count == 0 {
            return 0;
        }
        self.sum_us / self.count
    }

    /// Total number of recorded observations.
    fn count(&self) -> u64 {
        self.count
    }

    /// Merge another histogram into this one.
    fn merge(&mut self, other: &Self) {
        for (i, &c) in other.buckets.iter().enumerate() {
            self.buckets[i] += c;
        }
        self.count += other.count;
        self.sum_us += other.sum_us;
        if other.count > 0 {
            if other.min_us < self.min_us {
                self.min_us = other.min_us;
            }
            if other.max_us > self.max_us {
                self.max_us = other.max_us;
            }
        }
    }

    /// Check a list of SLOs against the current histogram.
    fn check_slos(&self, slos: &[Slo]) -> Vec<SloCheckResult> {
        slos.iter()
            .map(|&slo| {
                let actual = self.percentile(slo.percentile);
                SloCheckResult {
                    slo,
                    actual_us: actual,
                    passed: actual <= slo.threshold_us,
                }
            })
            .collect()
    }

    /// Render an ASCII histogram to a string.
    fn ascii_histogram(&self) -> String {
        let max_count = self.buckets.iter().copied().max().unwrap_or(1).max(1);
        let mut out = String::new();

        for (i, &count) in self.buckets.iter().enumerate() {
            let low_ms = (i as u64 * BUCKET_WIDTH_US) / 1000;
            let high_ms = ((i as u64 + 1) * BUCKET_WIDTH_US) / 1000;
            let bar_len = (count as usize * BAR_WIDTH) / max_count as usize;
            let bar: String = "#".repeat(bar_len);
            out.push_str(&format!(
                "   {:>4}-{:<4}ms |{:<width$}| {}\n",
                low_ms,
                high_ms,
                bar,
                count,
                width = BAR_WIDTH
            ));
        }
        out
    }
}

/// Compute the bucket index for a given latency value.
fn bucket_index(latency_us: u64) -> usize {
    let idx = latency_us / BUCKET_WIDTH_US;
    (idx as usize).min(NUM_BUCKETS - 1)
}

// ============================================================================
// Rolling Window Tracker
// ============================================================================

/// Tracks percentile statistics across successive time windows.
struct RollingWindowTracker {
    /// Per-window percentile snapshots: `[window_idx][percentile_idx]`
    snapshots: Vec<[u64; 5]>,
    /// Per-window request counts
    counts: Vec<u64>,
}

impl RollingWindowTracker {
    fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            counts: Vec::new(),
        }
    }

    /// Record a snapshot from a completed window histogram.
    fn record_window(&mut self, histogram: &LatencyHistogram) {
        let snapshot = [
            histogram.percentile(PERCENTILES[0]),
            histogram.percentile(PERCENTILES[1]),
            histogram.percentile(PERCENTILES[2]),
            histogram.percentile(PERCENTILES[3]),
            histogram.percentile(PERCENTILES[4]),
        ];
        self.snapshots.push(snapshot);
        self.counts.push(histogram.count());
    }

    /// Number of recorded windows.
    fn window_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Detect trend: returns the change in p99 from first to last window.
    /// Positive means latency increased (degradation).
    fn p99_trend(&self) -> i64 {
        if self.snapshots.len() < 2 {
            return 0;
        }
        let first = self.snapshots[0][3]; // p99 index
        let last = self.snapshots[self.snapshots.len() - 1][3];
        last as i64 - first as i64
    }
}

// ============================================================================
// Latency Simulation
// ============================================================================

/// Simulate a latency sample for a given variant.
///
/// Each variant has a different base latency profile:
/// - `model_a`: fast (mean ~5ms)
/// - `model_b`: slower (mean ~20ms)
/// - `batch`:   highly variable (mean ~50ms)
fn simulate_latency(variant: usize, rng: &mut DeterministicRng, degradation: f64) -> u64 {
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

// ============================================================================
// Main
// ============================================================================

// ============================================================================
// Helper Functions (extracted from main for reduced cyclomatic complexity)
// ============================================================================

/// Define the standard SLO thresholds used across all sections.
fn define_slos() -> [Slo; 5] {
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

/// Section 1: Histogram-Based Latency Tracking.
///
/// Generates 1000 requests across all variants and reports summary statistics
/// and percentiles.
fn run_histogram_tracking(rng: &mut DeterministicRng) -> LatencyHistogram {
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

/// Section 2: SLO Compliance Monitoring.
///
/// Checks the global histogram against defined SLOs and reports compliance.
fn run_slo_compliance(histogram: &LatencyHistogram, slos: &[Slo]) {
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

/// Section 3: Rolling Window Percentiles Over Time.
///
/// Simulates gradual degradation across windows and reports per-window
/// percentiles, SLO violations, and the overall p99 trend.
fn run_rolling_windows(rng: &mut DeterministicRng, slos: &[Slo]) {
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

/// Section 4: Latency Breakdown by Model Variant.
///
/// Generates per-variant data, prints individual and aggregate statistics,
/// and returns the merged aggregate histogram.
fn run_variant_breakdown(rng: &mut DeterministicRng) -> LatencyHistogram {
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

/// Section 5: ASCII Histogram Visualization.
///
/// Renders the aggregate histogram as an ASCII bar chart.
fn run_ascii_visualization(aggregate: &LatencyHistogram) {
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

/// Section 6: SLO Violation Alerts.
///
/// Re-simulates per-window data with a separate RNG seed and reports
/// individual SLO violations and total violation count.
fn run_slo_alerts(slos: &[Slo]) {
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

fn main() {
    println!("=== Latency Histogram & SLO Monitoring Example ===\n");

    let mut rng = DeterministicRng::new(42);
    let slos = define_slos();

    let global_histogram = run_histogram_tracking(&mut rng);
    run_slo_compliance(&global_histogram, &slos);
    run_rolling_windows(&mut rng, &slos);
    let aggregate = run_variant_breakdown(&mut rng);
    run_ascii_visualization(&aggregate);
    run_slo_alerts(&slos);

    println!("\n=== Example Complete ===");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_histogram_percentiles() {
        let hist = LatencyHistogram::new();
        assert_eq!(hist.percentile(50.0), 0);
        assert_eq!(hist.percentile(99.0), 0);
        assert_eq!(hist.count(), 0);
        assert_eq!(hist.mean_us(), 0);
    }

    #[test]
    fn test_single_observation() {
        let mut hist = LatencyHistogram::new();
        hist.record(5000); // 5ms
        assert_eq!(hist.count(), 1);
        assert_eq!(hist.mean_us(), 5000);
        assert_eq!(hist.min_us, 5000);
        assert_eq!(hist.max_us, 5000);
        // Percentile should land in the bucket containing 5000us
        let p50 = hist.percentile(50.0);
        assert!(
            p50 <= BUCKET_WIDTH_US,
            "p50={p50} should be within first bucket"
        );
    }

    #[test]
    fn test_percentile_monotonicity() {
        let mut rng = DeterministicRng::new(42);
        let mut hist = LatencyHistogram::new();
        for _ in 0..1000 {
            let latency = (rng.next_f64() * 100_000.0) as u64;
            hist.record(latency);
        }
        let p50 = hist.percentile(50.0);
        let p90 = hist.percentile(90.0);
        let p95 = hist.percentile(95.0);
        let p99 = hist.percentile(99.0);
        assert!(p50 <= p90, "p50={p50} should be <= p90={p90}");
        assert!(p90 <= p95, "p90={p90} should be <= p95={p95}");
        assert!(p95 <= p99, "p95={p95} should be <= p99={p99}");
    }

    #[test]
    fn test_bucket_index_boundaries() {
        assert_eq!(bucket_index(0), 0);
        assert_eq!(bucket_index(BUCKET_WIDTH_US - 1), 0);
        assert_eq!(bucket_index(BUCKET_WIDTH_US), 1);
        assert_eq!(bucket_index(MAX_LATENCY_US), NUM_BUCKETS - 1);
        // Values above max should clamp to last bucket
        assert_eq!(bucket_index(MAX_LATENCY_US + 1000), NUM_BUCKETS - 1);
    }

    #[test]
    fn test_histogram_merge() {
        let mut a = LatencyHistogram::new();
        let mut b = LatencyHistogram::new();
        a.record(1000);
        a.record(2000);
        b.record(3000);
        b.record(4000);

        a.merge(&b);
        assert_eq!(a.count(), 4);
        assert_eq!(a.sum_us, 10_000);
        assert_eq!(a.min_us, 1000);
        assert_eq!(a.max_us, 4000);
    }

    #[test]
    fn test_slo_pass_and_violation() {
        let mut hist = LatencyHistogram::new();
        // Record all low latencies
        for _ in 0..100 {
            hist.record(5000); // 5ms
        }

        let slos = [
            Slo {
                percentile: 99.0,
                threshold_us: 10_000,
            },
            Slo {
                percentile: 99.0,
                threshold_us: 1_000,
            },
        ];

        let results = hist.check_slos(&slos);
        assert!(results[0].passed, "5ms should pass 10ms SLO");
        assert!(!results[1].passed, "5ms should violate 1ms SLO");
    }

    #[test]
    fn test_slo_display_format() {
        let slo = Slo {
            percentile: 99.0,
            threshold_us: 100_000,
        };
        let display = format!("{slo}");
        assert!(
            display.contains("p99"),
            "Should contain percentile: {display}"
        );
        assert!(
            display.contains("100ms"),
            "Should contain threshold: {display}"
        );
    }

    #[test]
    fn test_slo_check_result_display() {
        let result = SloCheckResult {
            slo: Slo {
                percentile: 95.0,
                threshold_us: 80_000,
            },
            actual_us: 50_000,
            passed: true,
        };
        let display = format!("{result}");
        assert!(display.contains("PASS"), "Should show PASS: {display}");

        let violation = SloCheckResult {
            slo: Slo {
                percentile: 99.0,
                threshold_us: 100_000,
            },
            actual_us: 120_000,
            passed: false,
        };
        let vdisplay = format!("{violation}");
        assert!(
            vdisplay.contains("VIOLATION"),
            "Should show VIOLATION: {vdisplay}"
        );
    }

    #[test]
    fn test_rolling_window_tracker() {
        let mut tracker = RollingWindowTracker::new();
        assert_eq!(tracker.window_count(), 0);
        assert_eq!(tracker.p99_trend(), 0);

        let mut hist1 = LatencyHistogram::new();
        for _ in 0..100 {
            hist1.record(5000);
        }
        tracker.record_window(&hist1);

        let mut hist2 = LatencyHistogram::new();
        for _ in 0..100 {
            hist2.record(20_000);
        }
        tracker.record_window(&hist2);

        assert_eq!(tracker.window_count(), 2);
        // p99 should increase from window 1 to window 2
        assert!(
            tracker.p99_trend() > 0,
            "Trend should be positive (degrading), got {}",
            tracker.p99_trend()
        );
    }

    #[test]
    fn test_deterministic_rng_reproducibility() {
        let mut rng1 = DeterministicRng::new(123);
        let mut rng2 = DeterministicRng::new(123);

        let seq1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
        let seq2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();
        assert_eq!(seq1, seq2, "Same seed must produce identical sequences");
    }

    #[test]
    fn test_exponential_distribution_positive() {
        let mut rng = DeterministicRng::new(42);
        for _ in 0..100 {
            let val = rng.next_exponential(0.1);
            assert!(val >= 0.0, "Exponential should be non-negative, got {val}");
        }
    }

    #[test]
    fn test_ascii_histogram_not_empty() {
        let mut hist = LatencyHistogram::new();
        hist.record(10_000);
        hist.record(50_000);
        hist.record(100_000);

        let output = hist.ascii_histogram();
        assert!(!output.is_empty(), "ASCII histogram should produce output");
        assert!(output.contains('#'), "Should contain bar characters");
        assert!(output.contains("ms"), "Should contain ms labels");
    }

    #[test]
    fn test_format_percentile_integer() {
        assert_eq!(format_percentile(50.0), "50");
        assert_eq!(format_percentile(99.0), "99");
    }

    #[test]
    fn test_format_percentile_fractional() {
        assert_eq!(format_percentile(99.9), "99.9");
    }

    #[test]
    fn test_simulate_latency_variants_differ() {
        let mut rng_a = DeterministicRng::new(42);
        let mut rng_b = DeterministicRng::new(42);

        let mut sum_a: u64 = 0;
        let mut sum_b: u64 = 0;
        let n = 500;

        for _ in 0..n {
            sum_a += simulate_latency(0, &mut rng_a, 0.0);
            sum_b += simulate_latency(1, &mut rng_b, 0.0);
        }

        let mean_a = sum_a / n as u64;
        let mean_b = sum_b / n as u64;

        // model_b should be significantly slower than model_a
        assert!(
            mean_b > mean_a,
            "model_b mean ({mean_b}) should be > model_a mean ({mean_a})"
        );
    }

    #[test]
    fn test_histogram_clamps_high_values() {
        let mut hist = LatencyHistogram::new();
        hist.record(MAX_LATENCY_US + 100_000);
        assert_eq!(hist.count(), 1);
        // Should be in the last bucket
        assert_eq!(hist.buckets[NUM_BUCKETS - 1], 1);
    }

    #[test]
    fn test_merge_empty_histogram() {
        let mut a = LatencyHistogram::new();
        a.record(5000);

        let empty = LatencyHistogram::new();
        a.merge(&empty);

        assert_eq!(a.count(), 1);
        assert_eq!(a.min_us, 5000);
        assert_eq!(a.max_us, 5000);
    }
}
