//! # Monte-Carlo Disk I/O Queue Depth
//!
//! Sim a NCQ-style disk with bounded queue depth. Concurrent I/O
//! requests beyond depth wait. Reports avg latency by queue-depth
//! limit.
//!
//! Demonstrates the **MC.69** recipe for PMAT-182 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SATA NCQ spec (T13/D1532); fio iodepth parameter docs.
//!
//! Run with: cargo run --example mc_disk_io_queue_depth
//!
//! Added by PMAT-182 (catalog 1261→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::VecDeque;

#[derive(Debug, PartialEq)]
pub enum DiskVerdict {
    Ok {
        avg_latency: f64,
        max_latency: u32,
        completed: u32,
    },
    InvalidConfig,
}

pub fn simulate(requests: u32, queue_depth: u32, avg_service_us: u32, seed: u64) -> DiskVerdict {
    if requests == 0 || queue_depth == 0 || avg_service_us == 0 {
        return DiskVerdict::InvalidConfig;
    }
    let mut in_flight: VecDeque<u32> = VecDeque::with_capacity(queue_depth as usize);
    let mut now: u32 = 0;
    let mut total_latency: u64 = 0;
    let mut max_latency: u32 = 0;
    let mut completed: u32 = 0;
    let mut rng_state = seed | 1;
    for _ in 0..requests {
        // Block until queue has room.
        if in_flight.len() >= queue_depth as usize {
            if let Some(oldest_done) = in_flight.pop_front() {
                if oldest_done > now {
                    now = oldest_done;
                }
                completed += 1;
            }
        }
        let arrived_at = now;
        let service = 1 + ((lcg(&mut rng_state) >> 32) as u32) % (2 * avg_service_us);
        let done_at = arrived_at + service;
        in_flight.push_back(done_at);
        let latency = done_at - arrived_at;
        total_latency += u64::from(latency);
        if latency > max_latency {
            max_latency = latency;
        }
    }
    while let Some(d) = in_flight.pop_front() {
        completed += 1;
        if d > now {
            now = d;
        }
    }
    let avg_latency = total_latency as f64 / f64::from(requests);
    DiskVerdict::Ok {
        avg_latency,
        max_latency,
        completed,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_disk_io_queue_depth")?;

    println!("shallow: {:?}", simulate(1000, 1, 100, 42));
    println!("deep: {:?}", simulate(1000, 32, 100, 42));
    println!("invalid: {:?}", simulate(0, 1, 100, 42));
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
    fn deeper_queue_completes_all() {
        let v = simulate(1000, 32, 100, 42);
        if let DiskVerdict::Ok { completed, .. } = v {
            assert_eq!(completed, 1000);
        }
    }

    #[test]
    fn shallow_queue_completes_all() {
        let v = simulate(100, 1, 100, 42);
        if let DiskVerdict::Ok { completed, .. } = v {
            assert_eq!(completed, 100);
        }
    }

    #[test]
    fn invalid_zero_requests() {
        assert_eq!(simulate(0, 1, 100, 42), DiskVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_depth() {
        assert_eq!(simulate(100, 0, 100, 42), DiskVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_service() {
        assert_eq!(simulate(100, 1, 0, 42), DiskVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 4, 100, 42);
        let b = simulate(500, 4, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn avg_le_max_latency() {
        let v = simulate(1000, 4, 100, 42);
        if let DiskVerdict::Ok {
            avg_latency,
            max_latency,
            ..
        } = v
        {
            assert!(avg_latency <= f64::from(max_latency));
        }
    }

    #[test]
    fn avg_at_least_one() {
        // Service min is 1us → avg latency ≥ 1.
        let v = simulate(1000, 4, 100, 42);
        if let DiskVerdict::Ok { avg_latency, .. } = v {
            assert!(avg_latency >= 1.0);
        }
    }

    #[test]
    fn max_latency_in_range() {
        // Service is in [1, 2*avg] → max ≤ 2*avg_service_us.
        let v = simulate(1000, 4, 100, 42);
        if let DiskVerdict::Ok { max_latency, .. } = v {
            assert!(max_latency <= 200);
        }
    }

    #[test]
    fn single_request_works() {
        let v = simulate(1, 1, 100, 42);
        if let DiskVerdict::Ok { completed, .. } = v {
            assert_eq!(completed, 1);
        }
    }

    #[test]
    fn higher_service_higher_avg() {
        let lo = simulate(500, 4, 10, 42);
        let hi = simulate(500, 4, 1000, 42);
        if let (DiskVerdict::Ok { avg_latency: l, .. }, DiskVerdict::Ok { avg_latency: h, .. }) =
            (lo, hi)
        {
            assert!(h > l);
        }
    }
}
