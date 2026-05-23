//! # Monte-Carlo Event Correlation Window
//!
//! Sim event arrivals across N sources; detect concurrent bursts
//! (>= `min_concurrent` sources fire within a `window_ms` of each
//! other). Reports detected burst count.
//!
//! Demonstrates the **MC.107** recipe for PMAT-194 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SIEM correlation rule design (NIST SP 800-92);
//!  sliding-window event detection.
//!
//! Run with: cargo run --example mc_event_correlation_window
//!
//! Added by PMAT-194 (catalog 1369→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::VecDeque;

#[derive(Debug, PartialEq)]
pub enum CorrVerdict {
    Ok { burst_count: u32, events_total: u32 },
    InvalidConfig,
}

pub fn simulate(
    duration_ms: u32,
    sources: u32,
    event_prob_per_ms: f64,
    window_ms: u32,
    min_concurrent: u32,
    seed: u64,
) -> CorrVerdict {
    if duration_ms == 0
        || sources == 0
        || !(0.0..=1.0).contains(&event_prob_per_ms)
        || window_ms == 0
        || min_concurrent == 0
    {
        return CorrVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    // (timestamp, source_id) events.
    let mut events: VecDeque<(u32, u32)> = VecDeque::new();
    let mut burst_count = 0u32;
    let mut events_total = 0u32;
    for t in 0..duration_ms {
        // Each source has independent fire prob.
        for s in 0..sources {
            let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
            if r < event_prob_per_ms {
                events.push_back((t, s));
                events_total += 1;
            }
        }
        // Evict events outside window.
        while let Some(&(et, _)) = events.front() {
            if t.saturating_sub(et) > window_ms {
                events.pop_front();
            } else {
                break;
            }
        }
        // Count distinct sources in current window.
        let mut distinct: u32 = 0;
        let mut seen: Vec<u32> = Vec::new();
        for &(_, s) in &events {
            if !seen.contains(&s) {
                seen.push(s);
                distinct += 1;
            }
        }
        if distinct >= min_concurrent {
            burst_count += 1;
            // Clear window after detect to avoid double-count.
            events.clear();
        }
    }
    CorrVerdict::Ok {
        burst_count,
        events_total,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_event_correlation_window")?;

    println!("rare bursts: {:?}", simulate(10_000, 5, 0.001, 100, 3, 42));
    println!(
        "frequent bursts: {:?}",
        simulate(10_000, 5, 0.05, 100, 3, 42)
    );
    println!("invalid: {:?}", simulate(0, 5, 0.001, 100, 3, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn high_prob_more_bursts() {
        let lo = simulate(10_000, 5, 0.001, 100, 3, 42);
        let hi = simulate(10_000, 5, 0.05, 100, 3, 42);
        if let (CorrVerdict::Ok { burst_count: l, .. }, CorrVerdict::Ok { burst_count: h, .. }) =
            (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn invalid_zero_duration() {
        assert_eq!(
            simulate(0, 5, 0.001, 100, 3, 42),
            CorrVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_sources() {
        assert_eq!(
            simulate(1000, 0, 0.001, 100, 3, 42),
            CorrVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_prob_out_of_range() {
        assert_eq!(
            simulate(1000, 5, 1.5, 100, 3, 42),
            CorrVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_window() {
        assert_eq!(
            simulate(1000, 5, 0.001, 0, 3, 42),
            CorrVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_min_concurrent() {
        assert_eq!(
            simulate(1000, 5, 0.001, 100, 0, 42),
            CorrVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 5, 0.01, 100, 3, 42);
        let b = simulate(500, 5, 0.01, 100, 3, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn min_concurrent_gt_sources_no_bursts() {
        let v = simulate(1000, 3, 0.5, 100, 10, 42);
        if let CorrVerdict::Ok { burst_count, .. } = v {
            assert_eq!(burst_count, 0);
        }
    }

    #[test]
    fn zero_prob_no_events() {
        let v = simulate(1000, 5, 0.0, 100, 3, 42);
        if let CorrVerdict::Ok {
            events_total,
            burst_count,
        } = v
        {
            assert_eq!(events_total, 0);
            assert_eq!(burst_count, 0);
        }
    }

    #[test]
    fn full_prob_lots_of_bursts() {
        let v = simulate(1000, 5, 1.0, 10, 5, 42);
        if let CorrVerdict::Ok { burst_count, .. } = v {
            assert!(burst_count > 0);
        }
    }

    #[test]
    fn burst_le_duration() {
        let v = simulate(1000, 5, 0.5, 10, 3, 42);
        if let CorrVerdict::Ok { burst_count, .. } = v {
            assert!(burst_count <= 1000);
        }
    }
}
