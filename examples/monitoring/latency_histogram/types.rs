#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports, clippy::wildcard_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

// ============================================================================
// Constants
// ============================================================================

/// Number of fixed-width histogram buckets
pub const NUM_BUCKETS: usize = 20;

/// Maximum latency tracked (microseconds). Values above are clamped.
pub const MAX_LATENCY_US: u64 = 200_000;

/// Bucket width in microseconds
pub const BUCKET_WIDTH_US: u64 = MAX_LATENCY_US / NUM_BUCKETS as u64;

/// Number of model variants to simulate
pub const NUM_VARIANTS: usize = 3;

/// Variant labels
pub const VARIANT_NAMES: [&str; NUM_VARIANTS] = ["model_a", "model_b", "batch"];

/// Number of rolling windows to simulate
pub const NUM_WINDOWS: usize = 6;

/// Requests per rolling window
pub const REQUESTS_PER_WINDOW: usize = 200;

/// Standard percentiles to report
pub const PERCENTILES: [f64; 5] = [50.0, 90.0, 95.0, 99.0, 99.9];

/// Width of the ASCII bar chart (characters)
pub const BAR_WIDTH: usize = 50;

// ============================================================================
// Deterministic RNG
// ============================================================================

// Deterministic pseudo-random number generator using `DefaultHasher`.
//
// Produces repeatable sequences given the same seed, suitable for
/// simulation without pulling in external crate dependencies.
pub struct DeterministicRng {
    pub state: u64,
}

impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance state and return a pseudo-random `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.state.hash(&mut hasher);
        self.state = hasher.finish();
        self.state
    }

    /// Return a float in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    // Approximate an exponential distribution with rate parameter `lambda`.
    /// Returns -ln(U) / lambda where U is uniform in (0, 1).
    pub fn next_exponential(&mut self, lambda: f64) -> f64 {
        let u = self.next_f64().max(1e-15); // avoid ln(0)
        -(u.ln()) / lambda
    }
}

// ============================================================================
// SLO Definition
// ============================================================================

/// A Service Level Objective binding a percentile to a latency threshold.
#[derive(Debug, Clone, Copy)]
pub struct Slo {
    // Percentile (e.g. 99.0 for p99)
    pub percentile: f64,
    // Maximum acceptable latency in microseconds
    pub threshold_us: u64,
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

// ============================================================================
// SLO Check Result
// ============================================================================

/// Result of checking a single SLO against measured latency.
#[derive(Debug, Clone, Copy)]
pub struct SloCheckResult {
    pub slo: Slo,
    pub actual_us: u64,
    pub passed: bool,
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

// Fixed-bucket histogram for latency tracking.
//
// Buckets cover `[0, MAX_LATENCY_US)` in equal-width intervals.
/// Values exceeding `MAX_LATENCY_US` are placed in the last bucket.
pub struct LatencyHistogram {
    // Bucket counts; index `i` covers `[i * BUCKET_WIDTH_US, (i+1) * BUCKET_WIDTH_US)`
    pub buckets: [u64; NUM_BUCKETS],
    // Total number of recorded observations
    pub count: u64,
    // Running sum for mean calculation (microseconds)
    pub sum_us: u64,
    // Minimum observed latency
    pub min_us: u64,
    // Maximum observed latency
    pub max_us: u64,
}

impl LatencyHistogram {
    /// Create an empty histogram.
    pub fn new() -> Self {
        Self {
            buckets: [0; NUM_BUCKETS],
            count: 0,
            sum_us: 0,
            min_us: u64::MAX,
            max_us: 0,
        }
    }

    /// Record a latency observation in microseconds.
    pub fn record(&mut self, latency_us: u64) {
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

    // Compute the approximate latency at a given percentile (0-100).
    //
    // Uses linear interpolation within the bucket that contains the target
    /// cumulative count.
    pub fn percentile(&self, p: f64) -> u64 {
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
    pub fn mean_us(&self) -> u64 {
        if self.count == 0 {
            return 0;
        }
        self.sum_us / self.count
    }

    /// Total number of recorded observations.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Merge another histogram into this one.
    pub fn merge(&mut self, other: &Self) {
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
    pub fn check_slos(&self, slos: &[Slo]) -> Vec<SloCheckResult> {
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
    pub fn ascii_histogram(&self) -> String {
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

// ============================================================================
// Rolling Window Tracker
// ============================================================================

/// Tracks percentile statistics across successive time windows.
pub struct RollingWindowTracker {
    // Per-window percentile snapshots: `[window_idx][percentile_idx]`
    pub snapshots: Vec<[u64; 5]>,
    // Per-window request counts
    pub counts: Vec<u64>,
}

impl RollingWindowTracker {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            counts: Vec::new(),
        }
    }

    /// Record a snapshot from a completed window histogram.
    pub fn record_window(&mut self, histogram: &LatencyHistogram) {
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
    pub fn window_count(&self) -> usize {
        self.snapshots.len()
    }

    // Detect trend: returns the change in p99 from first to last window.
    /// Positive means latency increased (degradation).
    pub fn p99_trend(&self) -> i64 {
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

// Simulate a latency sample for a given variant.
//
// Each variant has a different base latency profile:
// - `model_a`: fast (mean ~5ms)
// - `model_b`: slower (mean ~20ms)

// ============================================================================
// Main
// ============================================================================

// ============================================================================
// Helper Functions (extracted from main for reduced cyclomatic complexity)
// ============================================================================

// Section 1: Histogram-Based Latency Tracking.
//
// Generates 1000 requests across all variants and reports summary statistics

// Section 2: SLO Compliance Monitoring.
//

// Section 3: Rolling Window Percentiles Over Time.
//
// Simulates gradual degradation across windows and reports per-window

// Section 4: Latency Breakdown by Model Variant.
//
// Generates per-variant data, prints individual and aggregate statistics,

// Section 5: ASCII Histogram Visualization.
//

// Section 6: SLO Violation Alerts.
//
// Re-simulates per-window data with a separate RNG seed and reports
