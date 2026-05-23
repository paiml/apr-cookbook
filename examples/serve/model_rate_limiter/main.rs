#![allow(unused_imports)]
//! Model Rate Limiter Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Demonstrates rate limiting and request throttling strategies for ML model
//! serving. Compares token bucket, sliding window, and per-client fairness
//! approaches with request prioritization and throughput metrics.
//!
//! ```text
//! Rate Limiting Pipeline:
//!
//!   [Client A] ─┐                    ┌─ [High Priority] ──→ immediate
//!   [Client B] ─┤──→ [Rate Limiter] ─┤─ [Med  Priority] ──→ if tokens available
//!   [Client C] ─┘        │           └─ [Low  Priority] ──→ best effort
//!                         │
//!                    ┌────┴────┐
//!                    │Strategy │
//!                    ├─────────┤
//!                    │ Token   │  Fixed refill rate, burst capacity
//!                    │ Bucket  │
//!                    ├─────────┤
//!                    │ Sliding │  Window-based count over time
//!                    │ Window  │
//!                    ├─────────┤
//!                    │ Per-    │  Fair share per client identity
//!                    │ Client  │
//!                    └─────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example model_rate_limiter
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr serve model.apr          # APR native format
//! apr serve model.gguf         # GGUF (llama.cpp compatible)
//! apr serve model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Crankshaw, D. et al. (2017). *Clipper: A Low-Latency Online Prediction Serving System*. NSDI. arXiv:1612.03079

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
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
    println!("=== Model Rate Limiter Example ===\n");

    let seed = 42_u64;

    demo_token_bucket();
    demo_sliding_window();
    demo_per_client_fairness();
    demo_request_prioritization(seed);
    demo_throughput_under_load(seed);
    demo_strategy_comparison(seed);

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket_initial_capacity() {
        let bucket = TokenBucket::new(50, 100);
        assert_eq!(bucket.available(), 50);
    }

    #[test]
    fn test_token_bucket_acquire_decrements() {
        let mut bucket = TokenBucket::new(10, 100);
        assert!(bucket.try_acquire(0, 3));
        assert_eq!(bucket.available(), 7);
    }

    #[test]
    fn test_token_bucket_rejects_when_empty() {
        let mut bucket = TokenBucket::new(5, 100);
        for _ in 0..5 {
            assert!(bucket.try_acquire(0, 1));
        }
        assert!(!bucket.try_acquire(0, 1));
        assert_eq!(bucket.available(), 0);
    }

    #[test]
    fn test_token_bucket_refill() {
        let mut bucket = TokenBucket::new(10, 100);
        // Drain all tokens
        for _ in 0..10 {
            bucket.try_acquire(0, 1);
        }
        assert_eq!(bucket.available(), 0);
        // Refill after 500ms at 100/sec = 50 tokens, capped at capacity 10
        bucket.refill(500);
        assert_eq!(bucket.available(), 10);
    }

    #[test]
    fn test_token_bucket_refill_caps_at_capacity() {
        let mut bucket = TokenBucket::new(20, 1000);
        // Already full, refilling shouldn't exceed capacity
        bucket.refill(5000);
        assert_eq!(bucket.available(), 20);
    }

    #[test]
    fn test_sliding_window_allows_within_limit() {
        let mut sw = SlidingWindow::new(10, 1000);
        for _ in 0..10 {
            assert!(sw.try_acquire(0));
        }
        assert!(!sw.try_acquire(0));
    }

    #[test]
    fn test_sliding_window_expires_old_slots() {
        let mut sw = SlidingWindow::new(10, 1000);
        // Fill the window at t=0
        for _ in 0..10 {
            sw.try_acquire(0);
        }
        assert!(!sw.try_acquire(0));
        // After the full window duration, slots should expire
        assert!(sw.try_acquire(1100));
    }

    #[test]
    fn test_sliding_window_current_count() {
        let mut sw = SlidingWindow::new(100, 1000);
        for _ in 0..25 {
            sw.try_acquire(0);
        }
        assert_eq!(sw.current_count(0), 25);
        // After expiry
        assert_eq!(sw.current_count(1100), 0);
    }

    #[test]
    fn test_per_client_limiter_isolates_clients() {
        let mut limiter = PerClientLimiter::new(5, 10, 100, 200);
        // Client A uses all its tokens
        for _ in 0..5 {
            assert!(limiter.try_acquire("client-a", 0));
        }
        assert!(!limiter.try_acquire("client-a", 0));
        // Client B should still have tokens
        assert!(limiter.try_acquire("client-b", 0));
    }

    #[test]
    fn test_per_client_global_limit() {
        let mut limiter = PerClientLimiter::new(100, 200, 3, 10);
        // Global limit is 3, even though per-client is 100
        assert!(limiter.try_acquire("a", 0));
        assert!(limiter.try_acquire("b", 0));
        assert!(limiter.try_acquire("c", 0));
        assert!(!limiter.try_acquire("d", 0));
    }

    #[test]
    fn test_per_client_tracks_clients() {
        let mut limiter = PerClientLimiter::new(10, 20, 100, 200);
        limiter.try_acquire("alpha", 0);
        limiter.try_acquire("beta", 0);
        limiter.try_acquire("gamma", 0);
        assert_eq!(limiter.client_count(), 3);
    }

    #[test]
    fn test_priority_high_gets_more_access() {
        let mut limiter = PrioritizedLimiter::new(10, 0);
        // No refill, just 10 tokens
        // High costs 1 token each, Low costs 3 each
        let mut high_ok = 0_u32;
        let mut low_ok = 0_u32;

        // Try high priority requests
        for _ in 0..10 {
            if limiter.try_acquire(0, Priority::High) {
                high_ok += 1;
            }
        }

        // Reset with a new limiter for fair comparison
        let mut limiter2 = PrioritizedLimiter::new(10, 0);
        for _ in 0..10 {
            if limiter2.try_acquire(0, Priority::Low) {
                low_ok += 1;
            }
        }

        assert!(
            high_ok > low_ok,
            "High priority ({high_ok}) should get more throughput than Low ({low_ok})"
        );
    }

    #[test]
    fn test_priority_acceptance_rate() {
        let mut limiter = PrioritizedLimiter::new(5, 0);
        limiter.try_acquire(0, Priority::High);
        limiter.try_acquire(0, Priority::High);
        // 2 accepted out of 2
        assert!((limiter.acceptance_rate(Priority::High) - 1.0).abs() < 1e-6);
        assert!((limiter.acceptance_rate(Priority::Low) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_priority_totals() {
        let mut limiter = PrioritizedLimiter::new(100, 1000);
        for _ in 0..10 {
            limiter.try_acquire(0, Priority::High);
            limiter.try_acquire(0, Priority::Medium);
            limiter.try_acquire(0, Priority::Low);
        }
        let total = limiter.total_accepted() + limiter.total_rejected();
        assert_eq!(total, 30);
    }

    #[test]
    fn test_load_metrics_empty() {
        let m = LoadTestMetrics::new();
        assert!(m.acceptance_rate().abs() < 1e-6);
        assert!(m.rejection_rate().abs() < 1e-6);
        assert!(m.avg_latency_us().abs() < 1e-6);
    }

    #[test]
    fn test_load_metrics_recording() {
        let mut m = LoadTestMetrics::new();
        m.record(true, 100);
        m.record(true, 200);
        m.record(false, 150);
        assert_eq!(m.total_requests, 3);
        assert_eq!(m.accepted, 2);
        assert_eq!(m.rejected, 1);
        assert!((m.acceptance_rate() - 2.0 / 3.0).abs() < 1e-6);
        assert!((m.rejection_rate() - 1.0 / 3.0).abs() < 1e-6);
        assert!((m.avg_latency_us() - 150.0).abs() < 1e-6);
    }

    #[test]
    fn test_sim_clock_monotonic() {
        let mut clock = SimClock::new();
        assert_eq!(clock.now_ms(), 0);
        clock.advance(100);
        assert_eq!(clock.now_ms(), 100);
        clock.advance(50);
        assert_eq!(clock.now_ms(), 150);
    }

    #[test]
    fn test_priority_from_index_cycles() {
        assert_eq!(Priority::from_index(0), Priority::High);
        assert_eq!(Priority::from_index(1), Priority::Medium);
        assert_eq!(Priority::from_index(2), Priority::Low);
        assert_eq!(Priority::from_index(3), Priority::High);
    }

    #[test]
    fn test_hash_deterministic() {
        assert_eq!(hash_u64(42, 0), hash_u64(42, 0));
        assert_ne!(hash_u64(42, 0), hash_u64(42, 1));
    }

    #[test]
    fn test_simulated_latency_range() {
        for i in 0..100 {
            let lat = simulated_latency_us(42, i);
            assert!(lat >= 50, "Latency {lat} below minimum");
            assert!(lat < 550, "Latency {lat} above maximum");
        }
    }
}
