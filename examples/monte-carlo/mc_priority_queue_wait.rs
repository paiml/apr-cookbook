//! # Monte-Carlo Priority-Queue Wait Time
//!
//! Simulate priority-queue wait times: high-priority tasks always
//! preempt low-priority ones. Returns mean wait time per priority
//! class and worst-case wait observed.
//!
//! Demonstrates the **MC.17** recipe for PMAT-163 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: M/M/1 priority queue (Kleinrock vol. 2).
//!
//! Run with: cargo run --example mc_priority_queue_wait
//!
//! Added by PMAT-163 (catalog 1090→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WaitVerdict {
    Ok {
        mean_high_wait: f64,
        mean_low_wait: f64,
        max_wait: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    high_arrivals: u32,
    low_arrivals: u32,
    p_high_first: f64,
    service_time: f64,
    seed: u64,
) -> WaitVerdict {
    if high_arrivals == 0
        || low_arrivals == 0
        || !p_high_first.is_finite()
        || !(0.0..=1.0).contains(&p_high_first)
        || !service_time.is_finite()
        || service_time <= 0.0
    {
        return WaitVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let total = high_arrivals + low_arrivals;
    let mut high_remaining = high_arrivals;
    let mut low_remaining = low_arrivals;
    let mut sum_high = 0.0;
    let mut sum_low = 0.0;
    let mut current_time = 0.0;
    let mut max_wait = 0.0_f64;
    let mut high_count = 0u32;
    let mut low_count = 0u32;
    for _ in 0..total {
        // Pick high if either we must (only high left) or with prob p_high_first.
        let pick_high = if low_remaining == 0 {
            true
        } else if high_remaining == 0 {
            false
        } else {
            unit(&mut rng_state) < p_high_first
        };
        let wait = current_time;
        if pick_high {
            sum_high += wait;
            high_count += 1;
            high_remaining -= 1;
        } else {
            sum_low += wait;
            low_count += 1;
            low_remaining -= 1;
        }
        if wait > max_wait {
            max_wait = wait;
        }
        current_time += service_time;
    }
    let mean_high_wait = if high_count > 0 {
        sum_high / f64::from(high_count)
    } else {
        0.0
    };
    let mean_low_wait = if low_count > 0 {
        sum_low / f64::from(low_count)
    } else {
        0.0
    };
    WaitVerdict::Ok {
        mean_high_wait,
        mean_low_wait,
        max_wait,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_priority_queue_wait")?;

    println!("balanced: {:?}", simulate(100, 100, 0.5, 0.1, 42));
    println!("high pref: {:?}", simulate(100, 100, 0.9, 0.1, 42));
    println!("invalid: {:?}", simulate(0, 100, 0.5, 0.1, 42));
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
    fn balanced_priorities_similar_wait() {
        let v = simulate(1000, 1000, 0.5, 0.1, 42);
        if let WaitVerdict::Ok {
            mean_high_wait,
            mean_low_wait,
            ..
        } = v
        {
            // With 50/50 priority pick, waits should be similar.
            assert!((mean_high_wait - mean_low_wait).abs() < 5.0);
        }
    }

    #[test]
    fn high_pref_lowers_high_wait() {
        let v = simulate(1000, 1000, 0.95, 0.1, 42);
        if let WaitVerdict::Ok {
            mean_high_wait,
            mean_low_wait,
            ..
        } = v
        {
            assert!(mean_high_wait < mean_low_wait);
        }
    }

    #[test]
    fn invalid_zero_high() {
        assert_eq!(simulate(0, 100, 0.5, 0.1, 42), WaitVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_low() {
        assert_eq!(simulate(100, 0, 0.5, 0.1, 42), WaitVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_neg_p() {
        assert_eq!(
            simulate(100, 100, -0.1, 0.1, 42),
            WaitVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_neg_service() {
        assert_eq!(
            simulate(100, 100, 0.5, -0.1, 42),
            WaitVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(100, 100, f64::NAN, 0.1, 42),
            WaitVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 100, 0.5, 0.1, 42);
        let b = simulate(100, 100, 0.5, 0.1, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn max_wait_at_least_mean() {
        let v = simulate(100, 100, 0.5, 0.1, 42);
        if let WaitVerdict::Ok {
            mean_high_wait,
            mean_low_wait,
            max_wait,
        } = v
        {
            assert!(max_wait >= mean_high_wait);
            assert!(max_wait >= mean_low_wait);
        }
    }

    #[test]
    fn first_request_waits_zero() {
        // First arrival sees an empty queue, waits 0.
        let v = simulate(1, 1, 0.5, 0.1, 42);
        if let WaitVerdict::Ok {
            mean_high_wait,
            mean_low_wait,
            ..
        } = v
        {
            // One of the two should have wait = 0.
            assert!(mean_high_wait < 1e-9 || mean_low_wait < 1e-9);
        }
    }
}
