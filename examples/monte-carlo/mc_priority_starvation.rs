//! # Monte-Carlo Priority Starvation
//!
//! Sim a priority queue with strict-priority scheduling. Returns the
//! starvation rate of the lowest-priority class (requests that never
//! get serviced within `simulation_steps`).
//!
//! Demonstrates the **MC.53** recipe for PMAT-175 (catalog crosses 1200).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: priority-queue starvation analysis (Kleinrock 1976).
//!
//! Run with: cargo run --example mc_priority_starvation
//!
//! Added by PMAT-175 (catalog 1198→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum StarvationVerdict {
    Ok {
        low_starvation_rate: f64,
        high_serviced: u32,
        low_serviced: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    high_arrival_prob: f64,
    low_arrival_prob: f64,
    service_per_step: u32,
    steps: u32,
    seed: u64,
) -> StarvationVerdict {
    if !high_arrival_prob.is_finite()
        || !low_arrival_prob.is_finite()
        || !(0.0..=1.0).contains(&high_arrival_prob)
        || !(0.0..=1.0).contains(&low_arrival_prob)
        || service_per_step == 0
        || steps == 0
    {
        return StarvationVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut high_q: u32 = 0;
    let mut low_q: u32 = 0;
    let mut high_serviced: u32 = 0;
    let mut low_serviced: u32 = 0;
    let mut low_total_arrivals: u32 = 0;
    for _ in 0..steps {
        if unit(&mut rng_state) < high_arrival_prob {
            high_q += 1;
        }
        if unit(&mut rng_state) < low_arrival_prob {
            low_q += 1;
            low_total_arrivals += 1;
        }
        let mut budget = service_per_step;
        let serve_high = high_q.min(budget);
        high_q -= serve_high;
        high_serviced += serve_high;
        budget -= serve_high;
        let serve_low = low_q.min(budget);
        low_q -= serve_low;
        low_serviced += serve_low;
    }
    let low_starvation_rate = if low_total_arrivals > 0 {
        f64::from(low_q) / f64::from(low_total_arrivals)
    } else {
        0.0
    };
    StarvationVerdict::Ok {
        low_starvation_rate,
        high_serviced,
        low_serviced,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_priority_starvation")?;

    println!("no starvation: {:?}", simulate(0.3, 0.3, 2, 1000, 42));
    println!("heavy high: {:?}", simulate(0.95, 0.3, 1, 1000, 42));
    println!("invalid: {:?}", simulate(-0.1, 0.3, 1, 1000, 42));
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
    fn balanced_load_no_starvation() {
        let v = simulate(0.3, 0.3, 2, 10_000, 42);
        if let StarvationVerdict::Ok {
            low_starvation_rate,
            ..
        } = v
        {
            assert!(low_starvation_rate < 0.1);
        }
    }

    #[test]
    fn heavy_high_starves_low() {
        let v = simulate(0.95, 0.3, 1, 10_000, 42);
        if let StarvationVerdict::Ok {
            low_starvation_rate,
            ..
        } = v
        {
            assert!(low_starvation_rate > 0.5);
        }
    }

    #[test]
    fn invalid_neg_high() {
        assert_eq!(
            simulate(-0.1, 0.3, 1, 1000, 42),
            StarvationVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_service() {
        assert_eq!(
            simulate(0.3, 0.3, 0, 1000, 42),
            StarvationVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(
            simulate(0.3, 0.3, 1, 0, 42),
            StarvationVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(f64::NAN, 0.3, 1, 1000, 42),
            StarvationVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(0.5, 0.3, 1, 1000, 42);
        let b = simulate(0.5, 0.3, 1, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn high_serviced_at_least_low() {
        // High has priority; should usually be serviced more.
        let v = simulate(0.5, 0.5, 1, 10_000, 42);
        if let StarvationVerdict::Ok {
            high_serviced,
            low_serviced,
            ..
        } = v
        {
            assert!(high_serviced >= low_serviced);
        }
    }

    #[test]
    fn ample_capacity_no_starvation() {
        let v = simulate(0.3, 0.3, 100, 1000, 42);
        if let StarvationVerdict::Ok {
            low_starvation_rate,
            ..
        } = v
        {
            assert!(low_starvation_rate < 0.05);
        }
    }

    #[test]
    fn rate_in_unit_range() {
        let v = simulate(0.7, 0.3, 1, 1000, 42);
        if let StarvationVerdict::Ok {
            low_starvation_rate,
            ..
        } = v
        {
            assert!((0.0..=1.0).contains(&low_starvation_rate));
        }
    }
}
