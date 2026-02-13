//! Model Rate Limiter Example
//!
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

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_RATE: u64 = 100;
const DEFAULT_BURST: u64 = 20;
const SLIDING_WINDOW_SIZE: usize = 10;
const NUM_PRIORITIES: usize = 3;
const PRIORITY_NAMES: [&str; NUM_PRIORITIES] = ["High", "Medium", "Low"];

// Priority multipliers: High gets 3x tokens, Medium 2x, Low 1x
const PRIORITY_MULTIPLIERS: [u64; NUM_PRIORITIES] = [3, 2, 1];

// ---------------------------------------------------------------------------
// Deterministic time source
// ---------------------------------------------------------------------------

/// Simulated monotonic clock for deterministic testing.
#[derive(Clone, Copy, Debug)]
struct SimClock {
    now_ms: u64,
}

impl SimClock {
    const fn new() -> Self {
        Self { now_ms: 0 }
    }

    fn advance(&mut self, ms: u64) {
        self.now_ms += ms;
    }

    const fn now_ms(self) -> u64 {
        self.now_ms
    }
}

// ---------------------------------------------------------------------------
// Priority
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Priority {
    High = 0,
    Medium = 1,
    Low = 2,
}

impl Priority {
    fn from_index(idx: usize) -> Self {
        match idx % NUM_PRIORITIES {
            0 => Self::High,
            1 => Self::Medium,
            _ => Self::Low,
        }
    }

    const fn name(self) -> &'static str {
        PRIORITY_NAMES[self as usize]
    }

    const fn multiplier(self) -> u64 {
        PRIORITY_MULTIPLIERS[self as usize]
    }
}

// ---------------------------------------------------------------------------
// Token Bucket Rate Limiter
// ---------------------------------------------------------------------------

struct TokenBucket {
    tokens: u64,
    capacity: u64,
    refill_rate: u64, // tokens per second
    last_refill_ms: u64,
}

impl TokenBucket {
    fn new(capacity: u64, refill_rate: u64) -> Self {
        Self {
            tokens: capacity,
            capacity,
            refill_rate,
            last_refill_ms: 0,
        }
    }

    fn refill(&mut self, now_ms: u64) {
        let elapsed_ms = now_ms.saturating_sub(self.last_refill_ms);
        let new_tokens = elapsed_ms * self.refill_rate / 1000;
        if new_tokens > 0 {
            self.tokens = (self.tokens + new_tokens).min(self.capacity);
            self.last_refill_ms = now_ms;
        }
    }

    fn try_acquire(&mut self, now_ms: u64, cost: u64) -> bool {
        self.refill(now_ms);
        if self.tokens >= cost {
            self.tokens -= cost;
            true
        } else {
            false
        }
    }

    const fn available(&self) -> u64 {
        self.tokens
    }
}

// ---------------------------------------------------------------------------
// Sliding Window Rate Limiter
// ---------------------------------------------------------------------------

struct SlidingWindow {
    window_slots: [u64; SLIDING_WINDOW_SIZE],
    counts: [u64; SLIDING_WINDOW_SIZE],
    max_per_window: u64,
    window_duration_ms: u64,
}

impl SlidingWindow {
    fn new(max_per_window: u64, window_duration_ms: u64) -> Self {
        Self {
            window_slots: [0; SLIDING_WINDOW_SIZE],
            counts: [0; SLIDING_WINDOW_SIZE],
            max_per_window,
            window_duration_ms,
        }
    }

    fn slot_index(&self, now_ms: u64) -> usize {
        ((now_ms / self.slot_duration_ms()) as usize) % SLIDING_WINDOW_SIZE
    }

    const fn slot_duration_ms(&self) -> u64 {
        self.window_duration_ms / SLIDING_WINDOW_SIZE as u64
    }

    fn clean_expired(&mut self, now_ms: u64) {
        let current_slot_time = now_ms / self.slot_duration_ms();
        for (i, slot) in self.window_slots.iter_mut().enumerate() {
            let slot_time = *slot;
            if current_slot_time.saturating_sub(slot_time) >= SLIDING_WINDOW_SIZE as u64 {
                *slot = 0;
                self.counts[i] = 0;
            }
        }
    }

    fn current_count(&mut self, now_ms: u64) -> u64 {
        self.clean_expired(now_ms);
        self.counts.iter().sum()
    }

    fn try_acquire(&mut self, now_ms: u64) -> bool {
        self.clean_expired(now_ms);
        let total: u64 = self.counts.iter().sum();
        if total >= self.max_per_window {
            return false;
        }
        let idx = self.slot_index(now_ms);
        let slot_time = now_ms / self.slot_duration_ms();
        self.window_slots[idx] = slot_time;
        self.counts[idx] += 1;
        true
    }
}

// ---------------------------------------------------------------------------
// Per-Client Fair Rate Limiter
// ---------------------------------------------------------------------------

struct PerClientLimiter {
    buckets: HashMap<String, TokenBucket>,
    per_client_capacity: u64,
    per_client_rate: u64,
    global_bucket: TokenBucket,
}

impl PerClientLimiter {
    fn new(
        per_client_capacity: u64,
        per_client_rate: u64,
        global_capacity: u64,
        global_rate: u64,
    ) -> Self {
        Self {
            buckets: HashMap::new(),
            per_client_capacity,
            per_client_rate,
            global_bucket: TokenBucket::new(global_capacity, global_rate),
        }
    }

    fn try_acquire(&mut self, client_id: &str, now_ms: u64) -> bool {
        // Check global limit first
        if !self.global_bucket.try_acquire(now_ms, 1) {
            return false;
        }

        let cap = self.per_client_capacity;
        let rate = self.per_client_rate;
        let bucket = self
            .buckets
            .entry(client_id.to_string())
            .or_insert_with(|| TokenBucket::new(cap, rate));

        if bucket.try_acquire(now_ms, 1) {
            true
        } else {
            // Refund the global token since per-client limit was exceeded
            self.global_bucket.tokens =
                (self.global_bucket.tokens + 1).min(self.global_bucket.capacity);
            false
        }
    }

    fn client_count(&self) -> usize {
        self.buckets.len()
    }
}

// ---------------------------------------------------------------------------
// Prioritized Rate Limiter
// ---------------------------------------------------------------------------

struct PrioritizedLimiter {
    bucket: TokenBucket,
    accepted: [u64; NUM_PRIORITIES],
    rejected: [u64; NUM_PRIORITIES],
}

impl PrioritizedLimiter {
    fn new(capacity: u64, refill_rate: u64) -> Self {
        Self {
            bucket: TokenBucket::new(capacity, refill_rate),
            accepted: [0; NUM_PRIORITIES],
            rejected: [0; NUM_PRIORITIES],
        }
    }

    fn try_acquire(&mut self, now_ms: u64, priority: Priority) -> bool {
        let cost = match priority {
            Priority::High => 1,
            Priority::Medium => 2,
            Priority::Low => 3,
        };

        // High priority gets bonus tokens via multiplier check
        let effective_available = self.bucket.available() * priority.multiplier();
        let effective_cost = cost * priority.multiplier();

        if effective_available >= effective_cost && self.bucket.try_acquire(now_ms, cost) {
            self.accepted[priority as usize] += 1;
            true
        } else {
            self.rejected[priority as usize] += 1;
            false
        }
    }

    fn refill(&mut self, now_ms: u64) {
        self.bucket.refill(now_ms);
    }

    fn acceptance_rate(&self, priority: Priority) -> f64 {
        let idx = priority as usize;
        let total = self.accepted[idx] + self.rejected[idx];
        if total == 0 {
            0.0
        } else {
            self.accepted[idx] as f64 / total as f64
        }
    }

    fn total_accepted(&self) -> u64 {
        self.accepted.iter().sum()
    }

    fn total_rejected(&self) -> u64 {
        self.rejected.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

struct LoadTestMetrics {
    total_requests: u64,
    accepted: u64,
    rejected: u64,
    latency_sum_us: u64,
}

impl LoadTestMetrics {
    const fn new() -> Self {
        Self {
            total_requests: 0,
            accepted: 0,
            rejected: 0,
            latency_sum_us: 0,
        }
    }

    fn record(&mut self, was_accepted: bool, latency_us: u64) {
        self.total_requests += 1;
        self.latency_sum_us += latency_us;
        if was_accepted {
            self.accepted += 1;
        } else {
            self.rejected += 1;
        }
    }

    fn acceptance_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.accepted as f64 / self.total_requests as f64
        }
    }

    fn rejection_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.rejected as f64 / self.total_requests as f64
        }
    }

    fn avg_latency_us(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.latency_sum_us as f64 / self.total_requests as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic helpers
// ---------------------------------------------------------------------------

fn hash_u64(seed: u64, idx: usize) -> u64 {
    let mut h = DefaultHasher::new();
    (seed, idx).hash(&mut h);
    h.finish()
}

fn simulated_latency_us(seed: u64, request_id: usize) -> u64 {
    hash_u64(seed, request_id) % 500 + 50
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Model Rate Limiter Example ===\n");

    let seed = 42_u64;

    // =========================================================================
    println!("1. Token Bucket Rate Limiter");
    println!("   ─────────────────────────────────────────");

    let mut bucket = TokenBucket::new(DEFAULT_BURST, DEFAULT_RATE);
    let mut clock = SimClock::new();

    println!(
        "   Capacity: {} tokens | Refill: {} tokens/sec",
        DEFAULT_BURST, DEFAULT_RATE
    );
    println!("   Initial tokens: {}", bucket.available());

    // Drain the bucket with a burst
    let mut accepted = 0_u64;
    let mut rejected = 0_u64;
    for _ in 0..30 {
        if bucket.try_acquire(clock.now_ms(), 1) {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }
    println!(
        "   Burst of 30 requests: {} accepted, {} rejected",
        accepted, rejected
    );
    println!("   Tokens remaining: {}", bucket.available());

    // Advance time to refill
    clock.advance(500); // 500ms = 50 tokens at 100/sec
    bucket.refill(clock.now_ms());
    println!(
        "   After 500ms refill: {} tokens available",
        bucket.available()
    );
    println!();

    // =========================================================================
    println!("2. Sliding Window Rate Limiter");
    println!("   ─────────────────────────────────────────");

    let window_max = 50_u64;
    let window_ms = 1000_u64;
    let mut window = SlidingWindow::new(window_max, window_ms);
    let mut clock = SimClock::new();

    println!(
        "   Max {} requests per {}ms window ({} slots)",
        window_max, window_ms, SLIDING_WINDOW_SIZE
    );

    // Fill the window
    accepted = 0;
    rejected = 0;
    for _ in 0..70 {
        if window.try_acquire(clock.now_ms()) {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }
    println!(
        "   70 requests at t=0ms: {} accepted, {} rejected",
        accepted, rejected
    );
    println!(
        "   Current window count: {}",
        window.current_count(clock.now_ms())
    );

    // Advance past window
    clock.advance(1100);
    println!(
        "   After 1100ms: window count = {} (slots expired)",
        window.current_count(clock.now_ms())
    );

    accepted = 0;
    for _ in 0..30 {
        if window.try_acquire(clock.now_ms()) {
            accepted += 1;
        }
    }
    println!("   30 more requests: {} accepted", accepted);
    println!();

    // =========================================================================
    println!("3. Per-Client Rate Limiting with Fairness");
    println!("   ─────────────────────────────────────────");

    let mut per_client = PerClientLimiter::new(10, 20, 100, 200);
    let mut clock = SimClock::new();

    let clients = [
        "client-alpha",
        "client-beta",
        "client-gamma",
        "client-delta",
    ];
    println!("   Per-client: 10 burst / 20 per sec | Global: 100 burst / 200 per sec");

    // Each client sends requests
    let mut client_accepted = [0_u64; 4];
    let mut client_rejected = [0_u64; 4];

    for round in 0..5_u64 {
        clock.advance(100);
        for (i, &client) in clients.iter().enumerate() {
            // Some clients are greedier than others
            let num_requests = (i + 1) * 3;
            for _ in 0..num_requests {
                if per_client.try_acquire(client, clock.now_ms()) {
                    client_accepted[i] += 1;
                } else {
                    client_rejected[i] += 1;
                }
            }
        }

        if round == 0 {
            println!(
                "   Round 1: {} unique clients tracked",
                per_client.client_count()
            );
        }
    }

    println!(
        "   {:>14} {:>10} {:>10} {:>10}",
        "Client", "Accepted", "Rejected", "Rate"
    );
    println!("   {}", "\u{2500}".repeat(48));
    for (i, &client) in clients.iter().enumerate() {
        let total = client_accepted[i] + client_rejected[i];
        let rate = if total == 0 {
            0.0
        } else {
            client_accepted[i] as f64 / total as f64
        };
        println!(
            "   {:>14} {:>10} {:>10} {:>9.1}%",
            client,
            client_accepted[i],
            client_rejected[i],
            rate * 100.0
        );
    }
    println!();

    // =========================================================================
    println!("4. Request Prioritization");
    println!("   ─────────────────────────────────────────");

    let mut prio_limiter = PrioritizedLimiter::new(30, 50);
    let mut clock = SimClock::new();

    println!("   Bucket: 30 capacity, 50/sec refill");
    println!("   Cost: High=1 token, Medium=2 tokens, Low=3 tokens");

    // Send mixed priority requests
    for round in 0..10_u64 {
        clock.advance(100);
        prio_limiter.refill(clock.now_ms());

        for req_idx in 0..15_usize {
            let priority = Priority::from_index(hash_u64(seed + round, req_idx) as usize);
            prio_limiter.try_acquire(clock.now_ms(), priority);
        }
    }

    println!(
        "   {:>8} {:>10} {:>10} {:>12}",
        "Priority", "Accepted", "Rejected", "Accept Rate"
    );
    println!("   {}", "\u{2500}".repeat(44));
    for i in 0..NUM_PRIORITIES {
        let priority = Priority::from_index(i);
        println!(
            "   {:>8} {:>10} {:>10} {:>11.1}%",
            priority.name(),
            prio_limiter.accepted[i],
            prio_limiter.rejected[i],
            prio_limiter.acceptance_rate(priority) * 100.0
        );
    }
    println!(
        "   Total: {} accepted, {} rejected",
        prio_limiter.total_accepted(),
        prio_limiter.total_rejected()
    );
    println!();

    // =========================================================================
    println!("5. Throughput Under Load");
    println!("   ─────────────────────────────────────────");

    let load_levels = [50, 100, 200, 500, 1000];
    let mut bucket = TokenBucket::new(DEFAULT_BURST, DEFAULT_RATE);
    let mut clock = SimClock::new();

    println!(
        "   {:>8} {:>10} {:>10} {:>12} {:>12}",
        "Load", "Accepted", "Rejected", "Accept Rate", "Avg Lat(us)"
    );
    println!("   {}", "\u{2500}".repeat(56));

    for &load in &load_levels {
        let mut metrics = LoadTestMetrics::new();
        clock.advance(1000); // 1 second between tests

        for i in 0..load {
            let ok = bucket.try_acquire(clock.now_ms(), 1);
            let lat = simulated_latency_us(seed, i);
            metrics.record(ok, lat);

            // Advance a small amount per request to simulate real time
            if load > 0 {
                clock.advance(1000 / load as u64);
            }
        }

        println!(
            "   {:>8} {:>10} {:>10} {:>11.1}% {:>10.0}us",
            load,
            metrics.accepted,
            metrics.rejected,
            metrics.acceptance_rate() * 100.0,
            metrics.avg_latency_us()
        );
    }
    println!();

    // =========================================================================
    println!("6. Strategy Comparison");
    println!("   ─────────────────────────────────────────");

    let test_load = 200_usize;
    let strategies = ["TokenBucket", "SlidingWindow", "PerClient"];

    // Token bucket test
    let mut tb = TokenBucket::new(DEFAULT_BURST, DEFAULT_RATE);
    let mut tb_clock = SimClock::new();
    let mut tb_metrics = LoadTestMetrics::new();
    for i in 0..test_load {
        let ok = tb.try_acquire(tb_clock.now_ms(), 1);
        tb_metrics.record(ok, simulated_latency_us(seed, i));
        tb_clock.advance(5);
    }

    // Sliding window test
    let mut sw = SlidingWindow::new(100, 1000);
    let mut sw_clock = SimClock::new();
    let mut sw_metrics = LoadTestMetrics::new();
    for i in 0..test_load {
        let ok = sw.try_acquire(sw_clock.now_ms());
        sw_metrics.record(ok, simulated_latency_us(seed, i));
        sw_clock.advance(5);
    }

    // Per-client test (4 clients, round-robin)
    let mut pc = PerClientLimiter::new(10, 20, 100, DEFAULT_RATE);
    let mut pc_clock = SimClock::new();
    let mut pc_metrics = LoadTestMetrics::new();
    let test_clients = ["svc-a", "svc-b", "svc-c", "svc-d"];
    for i in 0..test_load {
        let client = test_clients[i % test_clients.len()];
        let ok = pc.try_acquire(client, pc_clock.now_ms());
        pc_metrics.record(ok, simulated_latency_us(seed, i));
        pc_clock.advance(5);
    }

    let all_metrics = [&tb_metrics, &sw_metrics, &pc_metrics];

    println!(
        "   {:>14} {:>10} {:>10} {:>12} {:>12}",
        "Strategy", "Accepted", "Rejected", "Accept Rate", "Reject Rate"
    );
    println!("   {}", "\u{2500}".repeat(58));
    for (i, &strategy) in strategies.iter().enumerate() {
        let m = all_metrics[i];
        println!(
            "   {:>14} {:>10} {:>10} {:>11.1}% {:>11.1}%",
            strategy,
            m.accepted,
            m.rejected,
            m.acceptance_rate() * 100.0,
            m.rejection_rate() * 100.0
        );
    }

    let total_accepted: u64 = all_metrics.iter().map(|m| m.accepted).sum();
    let total_rejected: u64 = all_metrics.iter().map(|m| m.rejected).sum();
    println!(
        "   Combined: {} accepted / {} rejected across {} strategies\n",
        total_accepted,
        total_rejected,
        strategies.len()
    );

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
