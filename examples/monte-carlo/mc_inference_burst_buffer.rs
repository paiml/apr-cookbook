//! # Monte-Carlo Inference Burst-Buffer Occupancy
//!
//! Sim a fixed-capacity burst buffer during traffic spikes. Traffic
//! arrives in bursts; buffer drains at steady_rate. Returns max
//! buffer occupancy and overflow count.
//!
//! Demonstrates the **MC.33** recipe for PMAT-168 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Token-bucket / leaky-bucket buffer modelling.
//!
//! Run with: cargo run --example mc_inference_burst_buffer
//!
//! Added by PMAT-168 (catalog 1135→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BufferVerdict {
    Ok {
        max_occupancy: u32,
        overflow_count: u32,
        utilization_pct: f64,
    },
    InvalidConfig,
}

pub fn simulate(capacity: u32, drain_per_step: u32, arrivals_per_step: &[u32]) -> BufferVerdict {
    if capacity == 0 || arrivals_per_step.is_empty() {
        return BufferVerdict::InvalidConfig;
    }
    let mut occupancy: u32 = 0;
    let mut max_occupancy: u32 = 0;
    let mut overflow: u32 = 0;
    let mut total_used: u64 = 0;
    let steps = arrivals_per_step.len() as u64;
    for arrivals in arrivals_per_step {
        // First drain.
        occupancy = occupancy.saturating_sub(drain_per_step);
        // Then add arrivals; clamp at capacity, count overflow.
        let new_total = occupancy + arrivals;
        if new_total > capacity {
            overflow += new_total - capacity;
            occupancy = capacity;
        } else {
            occupancy = new_total;
        }
        if occupancy > max_occupancy {
            max_occupancy = occupancy;
        }
        total_used += u64::from(occupancy);
    }
    let utilization_pct = (total_used as f64 / (steps * u64::from(capacity)) as f64) * 100.0;
    BufferVerdict::Ok {
        max_occupancy,
        overflow_count: overflow,
        utilization_pct,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_inference_burst_buffer")?;

    let steady = vec![5; 10];
    println!("steady: {:?}", simulate(100, 10, &steady));

    let bursty = vec![5, 5, 200, 5, 5, 5];
    println!("bursty: {:?}", simulate(100, 10, &bursty));

    let overflow = vec![1000; 5];
    println!("overflow: {:?}", simulate(100, 10, &overflow));
    println!("invalid: {:?}", simulate(0, 10, &steady));
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
    fn steady_no_overflow() {
        let arrivals = vec![5; 10];
        let v = simulate(100, 10, &arrivals);
        if let BufferVerdict::Ok { overflow_count, .. } = v {
            assert_eq!(overflow_count, 0);
        }
    }

    #[test]
    fn bursty_high_max() {
        let arrivals = vec![5, 5, 200, 5];
        let v = simulate(100, 10, &arrivals);
        if let BufferVerdict::Ok { max_occupancy, .. } = v {
            assert_eq!(max_occupancy, 100);
        }
    }

    #[test]
    fn overflow_counted() {
        let arrivals = vec![1000; 5];
        let v = simulate(100, 10, &arrivals);
        if let BufferVerdict::Ok { overflow_count, .. } = v {
            assert!(overflow_count > 0);
        }
    }

    #[test]
    fn invalid_zero_capacity() {
        assert_eq!(simulate(0, 10, &[5]), BufferVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_empty_arrivals() {
        assert_eq!(simulate(100, 10, &[]), BufferVerdict::InvalidConfig);
    }

    #[test]
    fn drain_zero_fills_up() {
        let arrivals = vec![10; 20];
        let v = simulate(100, 0, &arrivals);
        if let BufferVerdict::Ok { max_occupancy, .. } = v {
            assert_eq!(max_occupancy, 100);
        }
    }

    #[test]
    fn drain_above_arrival_keeps_low() {
        let arrivals = vec![5; 10];
        let v = simulate(100, 100, &arrivals);
        if let BufferVerdict::Ok { max_occupancy, .. } = v {
            assert!(max_occupancy <= 5);
        }
    }

    #[test]
    fn utilization_in_unit_range() {
        let arrivals = vec![5; 100];
        let v = simulate(100, 10, &arrivals);
        if let BufferVerdict::Ok {
            utilization_pct, ..
        } = v
        {
            assert!((0.0..=100.0).contains(&utilization_pct));
        }
    }

    #[test]
    fn deterministic() {
        let arrivals = vec![5, 5, 200, 5];
        let a = simulate(100, 10, &arrivals);
        let b = simulate(100, 10, &arrivals);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_buffer_no_drain() {
        let arrivals = vec![0; 5];
        let v = simulate(100, 10, &arrivals);
        if let BufferVerdict::Ok {
            max_occupancy,
            overflow_count,
            ..
        } = v
        {
            assert_eq!(max_occupancy, 0);
            assert_eq!(overflow_count, 0);
        }
    }
}
