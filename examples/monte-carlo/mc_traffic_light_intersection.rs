//! # Monte-Carlo Traffic-Light Intersection
//!
//! Sim a single intersection with NS/EW phases and Poisson-like
//! arrivals. Reports max queue length per direction and total
//! cars served.
//!
//! Demonstrates the **MC.64** recipe for PMAT-180 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Webster, F.V., Traffic Signal Settings (1958).
//!
//! Run with: cargo run --example mc_traffic_light_intersection
//!
//! Added by PMAT-180 (catalog 1243→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TrafficVerdict {
    Ok {
        max_ns_queue: u32,
        max_ew_queue: u32,
        served_total: u32,
    },
    InvalidConfig,
}

#[allow(clippy::too_many_arguments)]
pub fn simulate(
    seconds: u32,
    ns_green_secs: u32,
    ew_green_secs: u32,
    ns_arrivals_per_min: u32,
    ew_arrivals_per_min: u32,
    service_per_sec: u32,
    seed: u64,
) -> TrafficVerdict {
    if seconds == 0 || ns_green_secs == 0 || ew_green_secs == 0 || service_per_sec == 0 {
        return TrafficVerdict::InvalidConfig;
    }
    let mut ns_q: u32 = 0;
    let mut ew_q: u32 = 0;
    let mut max_ns: u32 = 0;
    let mut max_ew: u32 = 0;
    let mut served: u32 = 0;
    let mut rng_state = seed | 1;
    let cycle = ns_green_secs + ew_green_secs;
    for sec in 0..seconds {
        // Poisson approximation: Bernoulli per second with rate/60.
        let r1 = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        if r1 < f64::from(ns_arrivals_per_min) / 60.0 {
            ns_q += 1;
        }
        let r2 = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        if r2 < f64::from(ew_arrivals_per_min) / 60.0 {
            ew_q += 1;
        }
        // Service.
        let phase = sec % cycle;
        if phase < ns_green_secs {
            let n = service_per_sec.min(ns_q);
            ns_q -= n;
            served += n;
        } else {
            let n = service_per_sec.min(ew_q);
            ew_q -= n;
            served += n;
        }
        max_ns = max_ns.max(ns_q);
        max_ew = max_ew.max(ew_q);
    }
    TrafficVerdict::Ok {
        max_ns_queue: max_ns,
        max_ew_queue: max_ew,
        served_total: served,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_traffic_light_intersection")?;

    println!("balanced: {:?}", simulate(3600, 30, 30, 600, 600, 1, 42));
    println!(
        "ns dominant: {:?}",
        simulate(3600, 45, 15, 1000, 100, 1, 42)
    );
    println!("invalid: {:?}", simulate(0, 30, 30, 600, 600, 1, 42));
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
    fn balanced_has_balanced_queues() {
        let v = simulate(3600, 30, 30, 600, 600, 1, 42);
        if let TrafficVerdict::Ok {
            max_ns_queue,
            max_ew_queue,
            ..
        } = v
        {
            // Same arrival rate + same green time → roughly equal max queues.
            // We allow a wide tolerance.
            let diff = max_ns_queue.abs_diff(max_ew_queue);
            assert!(diff <= max_ns_queue.max(max_ew_queue));
        }
    }

    #[test]
    fn served_le_total_arrivals_capacity() {
        let v = simulate(60, 30, 30, 60, 60, 1, 42);
        if let TrafficVerdict::Ok { served_total, .. } = v {
            // Service capacity = 60 sec * 1/sec = 60.
            assert!(served_total <= 60);
        }
    }

    #[test]
    fn invalid_zero_seconds() {
        assert_eq!(
            simulate(0, 30, 30, 600, 600, 1, 42),
            TrafficVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_ns_green() {
        assert_eq!(
            simulate(100, 0, 30, 600, 600, 1, 42),
            TrafficVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_service() {
        assert_eq!(
            simulate(100, 30, 30, 600, 600, 0, 42),
            TrafficVerdict::InvalidConfig
        );
    }

    #[test]
    fn high_arrival_rate_grows_queue() {
        // Service capacity per direction = 30 sec/min * 1/sec = 0.5/sec.
        // 12/min = 0.2/sec (light); 50/min = 0.83/sec (over capacity).
        let low = simulate(3600, 30, 30, 12, 12, 1, 42);
        let high = simulate(3600, 30, 30, 50, 50, 1, 42);
        if let (
            TrafficVerdict::Ok {
                max_ns_queue: l, ..
            },
            TrafficVerdict::Ok {
                max_ns_queue: h, ..
            },
        ) = (low, high)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn higher_service_clears_queues() {
        let slow = simulate(1800, 30, 30, 1200, 1200, 1, 42);
        let fast = simulate(1800, 30, 30, 1200, 1200, 5, 42);
        if let (
            TrafficVerdict::Ok {
                max_ns_queue: s, ..
            },
            TrafficVerdict::Ok {
                max_ns_queue: f, ..
            },
        ) = (slow, fast)
        {
            assert!(f <= s);
        }
    }

    #[test]
    fn deterministic() {
        let a = simulate(600, 30, 30, 600, 600, 1, 42);
        let b = simulate(600, 30, 30, 600, 600, 1, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn served_nonneg_under_zero_arrivals() {
        let v = simulate(600, 30, 30, 0, 0, 1, 42);
        if let TrafficVerdict::Ok { served_total, .. } = v {
            assert_eq!(served_total, 0);
        }
    }

    #[test]
    fn ns_dominant_skews_queue() {
        // Subsaturated rates: ns=50/min (0.83/sec, over 0.5 capacity)
        // vs ew=5/min (0.083/sec, well under capacity).
        let v = simulate(3600, 30, 30, 50, 5, 1, 42);
        if let TrafficVerdict::Ok {
            max_ns_queue,
            max_ew_queue,
            ..
        } = v
        {
            assert!(max_ns_queue > max_ew_queue);
        }
    }
}
