#![allow(unused_imports)]
//! Latency Percentile Tracking and SLO Monitoring for Production ML Inference
//!
//! Contract: contracts/recipe-iiur-v1.yaml
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

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
    #[allow(unused_imports, clippy::wildcard_imports)]
    use super::helpers::*;
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
