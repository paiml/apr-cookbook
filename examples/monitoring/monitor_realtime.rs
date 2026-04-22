//! # Recipe: Monitor — Real-time Inference with Rolling Metrics
//!
//! **Category**: monitoring
//! **CLI Equivalent**: `apr monitor model.apr --window 60s --metrics p50,p99,throughput`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example monitor_realtime` exits 0
//! 2. [x] `cargo test --example monitor_realtime` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr monitor` in-process (no shell-out)
//! 10. [x] Unit tests cover window roll-off, percentile math, throughput
//!
//! ## Learning Objective
//! Implements a rolling-window real-time inference monitor. Each simulated
//! inference emits a latency sample; old samples fall out of the window. The
//! monitor reports p50, p99, mean, and throughput per tick — the same metrics
//! `apr monitor` exposes live.
//!
//! ## Run Command
//! ```bash
//! cargo run --example monitor_realtime
//! ```
//!
//! ## References
//! - Dean, J. & Barroso, L.A. (2013). *The Tail at Scale*. CACM. DOI: 10.1145/2408776.2408794

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use rand::Rng;
use serde_json::json;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
struct Sample {
    ts_ms: u64,
    latency_us: u64,
}

#[derive(Debug, Clone)]
struct WindowStats {
    #[allow(dead_code)]
    window_ms: u64,
    n_samples: usize,
    p50_us: u64,
    p99_us: u64,
    mean_us: f64,
    throughput_qps: f64,
}

struct RollingMonitor {
    window_ms: u64,
    samples: VecDeque<Sample>,
}

impl RollingMonitor {
    fn new(window_ms: u64) -> Self {
        Self {
            window_ms,
            samples: VecDeque::new(),
        }
    }
    fn record(&mut self, s: Sample) {
        self.samples.push_back(s);
        self.evict_older_than(s.ts_ms);
    }
    fn evict_older_than(&mut self, now_ms: u64) {
        let cutoff = now_ms.saturating_sub(self.window_ms);
        while let Some(front) = self.samples.front() {
            if front.ts_ms < cutoff {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }
    fn stats(&self) -> WindowStats {
        let n = self.samples.len();
        if n == 0 {
            return WindowStats {
                window_ms: self.window_ms,
                n_samples: 0,
                p50_us: 0,
                p99_us: 0,
                mean_us: 0.0,
                throughput_qps: 0.0,
            };
        }
        let mut lats: Vec<u64> = self.samples.iter().map(|s| s.latency_us).collect();
        lats.sort_unstable();
        let p = |q: f64| {
            let idx = ((n as f64) * q).floor() as usize;
            let idx = idx.min(n - 1);
            lats[idx]
        };
        let mean = lats.iter().copied().sum::<u64>() as f64 / n as f64;
        let throughput = if self.window_ms == 0 {
            0.0
        } else {
            n as f64 * 1000.0 / self.window_ms as f64
        };
        WindowStats {
            window_ms: self.window_ms,
            n_samples: n,
            p50_us: p(0.50),
            p99_us: p(0.99),
            mean_us: mean,
            throughput_qps: throughput,
        }
    }
}

fn simulate_tick<R: Rng>(rng: &mut R, ts_ms: u64, n_per_tick: usize) -> Vec<Sample> {
    (0..n_per_tick)
        .map(|_| Sample {
            ts_ms,
            // Lognormal-ish: mostly 200-500us, occasional spike 2000-10000us.
            latency_us: if rng.gen_bool(0.02) {
                rng.gen_range(2_000..10_000)
            } else {
                rng.gen_range(200..500)
            },
        })
        .collect()
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("monitor_realtime")?;
    println!("=== Recipe: {} ===", ctx.name());

    let mut mon = RollingMonitor::new(5_000); // 5s window
    let n_ticks = 20;
    let tick_interval_ms = 500;
    let n_per_tick = 10;

    println!(
        "\nwindow={}ms  interval={}ms  per_tick={}",
        mon.window_ms, tick_interval_ms, n_per_tick
    );
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Tick", "N", "p50 us", "p99 us", "mean us", "qps"
    );

    let mut all_stats = Vec::new();
    for tick in 0..n_ticks {
        let ts_ms = tick as u64 * tick_interval_ms;
        for s in simulate_tick(ctx.rng(), ts_ms, n_per_tick) {
            mon.record(s);
        }
        let st = mon.stats();
        println!(
            "{:>6} {:>10} {:>10} {:>10} {:>10.1} {:>10.1}",
            tick, st.n_samples, st.p50_us, st.p99_us, st.mean_us, st.throughput_qps
        );
        all_stats.push((tick, st));
    }

    let report = json!({
        "recipe": ctx.name(),
        "window_ms": mon.window_ms,
        "ticks": all_stats.iter().map(|(t, s)| json!({
            "tick": t,
            "n_samples": s.n_samples,
            "p50_us": s.p50_us,
            "p99_us": s.p99_us,
            "mean_us": s.mean_us,
            "throughput_qps": s.throughput_qps,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("monitor.json");
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out, bytes)?;

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(ts: u64, lat: u64) -> Sample {
        Sample {
            ts_ms: ts,
            latency_us: lat,
        }
    }

    #[test]
    fn empty_stats_are_zero() {
        let m = RollingMonitor::new(1000);
        let st = m.stats();
        assert_eq!(st.n_samples, 0);
        assert_eq!(st.p50_us, 0);
        assert_eq!(st.p99_us, 0);
    }

    #[test]
    fn window_evicts_old_samples() {
        let mut m = RollingMonitor::new(1000);
        m.record(s(0, 100));
        m.record(s(500, 200));
        m.record(s(1500, 300));
        // Only samples >= 500 remain after evicting <500.
        assert_eq!(m.stats().n_samples, 2);
    }

    #[test]
    fn p50_is_middle_value() {
        let mut m = RollingMonitor::new(10_000);
        for (i, v) in [100, 200, 300, 400, 500].iter().enumerate() {
            m.record(s(i as u64 * 100, *v));
        }
        let st = m.stats();
        assert_eq!(st.p50_us, 300);
    }

    #[test]
    fn p99_is_near_max() {
        let mut m = RollingMonitor::new(100_000);
        for v in 1..=100 {
            m.record(s(0, v));
        }
        let st = m.stats();
        assert!(st.p99_us >= 99);
    }

    #[test]
    fn throughput_scales_with_samples() {
        let mut m = RollingMonitor::new(1000); // 1s window
        for i in 0..10 {
            m.record(s(i * 50, 100));
        }
        let st = m.stats();
        // 10 samples in 1s => 10 qps.
        assert!((st.throughput_qps - 10.0).abs() < 1e-9);
    }
}
