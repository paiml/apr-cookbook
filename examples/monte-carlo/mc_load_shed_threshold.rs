//! # Monte-Carlo Load-Shed Threshold
//!
//! Simulate accept/reject decisions under a load-shed policy that
//! kicks in when concurrent_inflight ≥ threshold. Returns reject_pct
//! and observed mean concurrent inflight.
//!
//! Demonstrates the **MC.13** recipe for PMAT-162 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Adaptive load shedding (Netflix concurrency-limits).
//!
//! Run with: cargo run --example mc_load_shed_threshold
//!
//! Added by PMAT-162 (catalog 1081→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ShedVerdict {
    Ok {
        reject_pct: f64,
        mean_inflight: f64,
        rejects: u32,
    },
    InvalidConfig,
}

pub fn simulate(arrivals: u32, p_complete_per_step: f64, threshold: u32, seed: u64) -> ShedVerdict {
    if arrivals == 0
        || threshold == 0
        || !p_complete_per_step.is_finite()
        || !(0.0..=1.0).contains(&p_complete_per_step)
    {
        return ShedVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut inflight: u32 = 0;
    let mut rejects: u32 = 0;
    let mut sum_inflight: u64 = 0;
    for _ in 0..arrivals {
        // Each step, complete with probability p (independent per worker).
        let mut completed = 0u32;
        for _ in 0..inflight {
            if unit(&mut rng_state) < p_complete_per_step {
                completed += 1;
            }
        }
        inflight = inflight.saturating_sub(completed);
        if inflight >= threshold {
            rejects += 1;
        } else {
            inflight += 1;
        }
        sum_inflight += u64::from(inflight);
    }
    let reject_pct = (f64::from(rejects) / f64::from(arrivals)) * 100.0;
    let mean_inflight = sum_inflight as f64 / f64::from(arrivals);
    ShedVerdict::Ok {
        reject_pct,
        mean_inflight,
        rejects,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_load_shed_threshold")?;

    println!("light: {:?}", simulate(1000, 0.5, 100, 42));
    println!("heavy: {:?}", simulate(1000, 0.05, 20, 42));
    println!("invalid: {:?}", simulate(0, 0.5, 100, 42));
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
    fn light_load_no_rejects() {
        let v = simulate(100, 0.99, 10, 42);
        if let ShedVerdict::Ok { rejects, .. } = v {
            assert_eq!(rejects, 0);
        }
    }

    #[test]
    fn heavy_load_rejects() {
        let v = simulate(1000, 0.05, 5, 42);
        if let ShedVerdict::Ok { rejects, .. } = v {
            assert!(rejects > 0);
        }
    }

    #[test]
    fn invalid_zero_arrivals() {
        assert_eq!(simulate(0, 0.5, 10, 42), ShedVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_threshold() {
        assert_eq!(simulate(100, 0.5, 0, 42), ShedVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_negative_p() {
        assert_eq!(simulate(100, -0.1, 10, 42), ShedVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_over_one_p() {
        assert_eq!(simulate(100, 1.5, 10, 42), ShedVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(simulate(100, f64::NAN, 10, 42), ShedVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 0.3, 20, 42);
        let b = simulate(500, 0.3, 20, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn reject_pct_in_range() {
        let v = simulate(500, 0.3, 20, 42);
        if let ShedVerdict::Ok { reject_pct, .. } = v {
            assert!((0.0..=100.0).contains(&reject_pct));
        }
    }

    #[test]
    fn higher_threshold_fewer_rejects() {
        let lo = simulate(500, 0.1, 5, 42);
        let hi = simulate(500, 0.1, 50, 42);
        if let (ShedVerdict::Ok { rejects: r_lo, .. }, ShedVerdict::Ok { rejects: r_hi, .. }) =
            (lo, hi)
        {
            assert!(r_lo >= r_hi);
        }
    }
}
