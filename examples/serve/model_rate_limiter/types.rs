#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const DEFAULT_RATE: u64 = 100;
pub const DEFAULT_BURST: u64 = 20;
pub const SLIDING_WINDOW_SIZE: usize = 10;
pub const NUM_PRIORITIES: usize = 3;
pub const PRIORITY_NAMES: [&str; NUM_PRIORITIES] = ["High", "Medium", "Low"];

// Priority multipliers: High gets 3x tokens, Medium 2x, Low 1x
pub const PRIORITY_MULTIPLIERS: [u64; NUM_PRIORITIES] = [3, 2, 1];

// ---------------------------------------------------------------------------
// Deterministic time source
// ---------------------------------------------------------------------------

/// Simulated monotonic clock for deterministic testing.
#[derive(Clone, Copy, Debug)]
pub struct SimClock {
    pub now_ms: u64,
}

impl SimClock {
    pub const fn new() -> Self {
        Self { now_ms: 0 }
    }

    pub fn advance(&mut self, ms: u64) {
        self.now_ms += ms;
    }

    pub const fn now_ms(self) -> u64 {
        self.now_ms
    }
}

// ---------------------------------------------------------------------------
// Priority
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Priority {
    High = 0,
    Medium = 1,
    Low = 2,
}

impl Priority {
    pub fn from_index(idx: usize) -> Self {
        match idx % NUM_PRIORITIES {
            0 => Self::High,
            1 => Self::Medium,
            _ => Self::Low,
        }
    }

    pub const fn name(self) -> &'static str {
        PRIORITY_NAMES[self as usize]
    }

    pub const fn multiplier(self) -> u64 {
        PRIORITY_MULTIPLIERS[self as usize]
    }
}

// ---------------------------------------------------------------------------
// Token Bucket Rate Limiter
// ---------------------------------------------------------------------------

pub struct TokenBucket {
    pub tokens: u64,
    pub capacity: u64,
    pub refill_rate: u64, // tokens per second
    pub last_refill_ms: u64,
}

impl TokenBucket {
    pub fn new(capacity: u64, refill_rate: u64) -> Self {
        Self {
            tokens: capacity,
            capacity,
            refill_rate,
            last_refill_ms: 0,
        }
    }

    pub fn refill(&mut self, now_ms: u64) {
        let elapsed_ms = now_ms.saturating_sub(self.last_refill_ms);
        let new_tokens = elapsed_ms * self.refill_rate / 1000;
        if new_tokens > 0 {
            self.tokens = (self.tokens + new_tokens).min(self.capacity);
            self.last_refill_ms = now_ms;
        }
    }

    pub fn try_acquire(&mut self, now_ms: u64, cost: u64) -> bool {
        self.refill(now_ms);
        if self.tokens >= cost {
            self.tokens -= cost;
            true
        } else {
            false
        }
    }

    pub const fn available(&self) -> u64 {
        self.tokens
    }
}

// ---------------------------------------------------------------------------
// Sliding Window Rate Limiter
// ---------------------------------------------------------------------------

pub struct SlidingWindow {
    window_slots: [u64; SLIDING_WINDOW_SIZE],
    counts: [u64; SLIDING_WINDOW_SIZE],
    pub max_per_window: u64,
    pub window_duration_ms: u64,
}

impl SlidingWindow {
    pub fn new(max_per_window: u64, window_duration_ms: u64) -> Self {
        Self {
            window_slots: [0; SLIDING_WINDOW_SIZE],
            counts: [0; SLIDING_WINDOW_SIZE],
            max_per_window,
            window_duration_ms,
        }
    }

    pub fn slot_index(&self, now_ms: u64) -> usize {
        ((now_ms / self.slot_duration_ms()) as usize) % SLIDING_WINDOW_SIZE
    }

    pub const fn slot_duration_ms(&self) -> u64 {
        self.window_duration_ms / SLIDING_WINDOW_SIZE as u64
    }

    pub fn clean_expired(&mut self, now_ms: u64) {
        let current_slot_time = now_ms / self.slot_duration_ms();
        for (i, slot) in self.window_slots.iter_mut().enumerate() {
            let slot_time = *slot;
            if current_slot_time.saturating_sub(slot_time) >= SLIDING_WINDOW_SIZE as u64 {
                *slot = 0;
                self.counts[i] = 0;
            }
        }
    }

    pub fn current_count(&mut self, now_ms: u64) -> u64 {
        self.clean_expired(now_ms);
        self.counts.iter().sum()
    }

    pub fn try_acquire(&mut self, now_ms: u64) -> bool {
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

pub struct PerClientLimiter {
    pub buckets: HashMap<String, TokenBucket>,
    pub per_client_capacity: u64,
    pub per_client_rate: u64,
    pub global_bucket: TokenBucket,
}

impl PerClientLimiter {
    pub fn new(
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

    pub fn try_acquire(&mut self, client_id: &str, now_ms: u64) -> bool {
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

    pub fn client_count(&self) -> usize {
        self.buckets.len()
    }
}

// ---------------------------------------------------------------------------
// Prioritized Rate Limiter
// ---------------------------------------------------------------------------

pub struct PrioritizedLimiter {
    pub bucket: TokenBucket,
    pub accepted: [u64; NUM_PRIORITIES],
    pub rejected: [u64; NUM_PRIORITIES],
}

impl PrioritizedLimiter {
    pub fn new(capacity: u64, refill_rate: u64) -> Self {
        Self {
            bucket: TokenBucket::new(capacity, refill_rate),
            accepted: [0; NUM_PRIORITIES],
            rejected: [0; NUM_PRIORITIES],
        }
    }

    pub fn try_acquire(&mut self, now_ms: u64, priority: Priority) -> bool {
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

    pub fn refill(&mut self, now_ms: u64) {
        self.bucket.refill(now_ms);
    }

    pub fn acceptance_rate(&self, priority: Priority) -> f64 {
        let idx = priority as usize;
        let total = self.accepted[idx] + self.rejected[idx];
        if total == 0 {
            0.0
        } else {
            self.accepted[idx] as f64 / total as f64
        }
    }

    pub fn total_accepted(&self) -> u64 {
        self.accepted.iter().sum()
    }

    pub fn total_rejected(&self) -> u64 {
        self.rejected.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

pub struct LoadTestMetrics {
    pub total_requests: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub latency_sum_us: u64,
}

impl LoadTestMetrics {
    pub const fn new() -> Self {
        Self {
            total_requests: 0,
            accepted: 0,
            rejected: 0,
            latency_sum_us: 0,
        }
    }

    pub fn record(&mut self, was_accepted: bool, latency_us: u64) {
        self.total_requests += 1;
        self.latency_sum_us += latency_us;
        if was_accepted {
            self.accepted += 1;
        } else {
            self.rejected += 1;
        }
    }

    pub fn acceptance_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.accepted as f64 / self.total_requests as f64
        }
    }

    pub fn rejection_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.rejected as f64 / self.total_requests as f64
        }
    }

    pub fn avg_latency_us(&self) -> f64 {
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

// ---------------------------------------------------------------------------
// Demo helper functions (extracted from main to reduce cyclomatic complexity)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
