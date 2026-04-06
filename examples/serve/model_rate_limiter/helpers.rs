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
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

pub fn hash_u64(seed: u64, idx: usize) -> u64 {
    let mut h = DefaultHasher::new();
    (seed, idx).hash(&mut h);
    h.finish()
}

pub fn simulated_latency_us(seed: u64, request_id: usize) -> u64 {
    hash_u64(seed, request_id) % 500 + 50
}

pub fn demo_token_bucket() {
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
}

pub fn demo_sliding_window() {
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
    let mut accepted = 0_u64;
    let mut rejected = 0_u64;
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
}

pub fn demo_per_client_fairness() {
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
}

pub fn demo_request_prioritization(seed: u64) {
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
}

pub fn demo_throughput_under_load(seed: u64) {
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
}

pub fn demo_strategy_comparison(seed: u64) {
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
}
