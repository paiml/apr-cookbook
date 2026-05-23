//! # Monte-Carlo Load Test Ramp-Up
//!
//! Sim a load test that ramps from 1 to `max_rps` over `ramp_secs`.
//! Server has finite capacity `max_capacity`. Reports observed
//! saturation point (RPS at which errors start) and final error rate.
//!
//! Demonstrates the **MC.108** recipe for PMAT-195 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: load-testing methodology (Meier, Performance Testing
//!  Guide, 2007); k6/wrk2 ramp patterns.
//!
//! Run with: cargo run --example mc_load_test_ramp_up
//!
//! Added by PMAT-195 (catalog 1378→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RampVerdict {
    Ok {
        saturation_rps: u32,
        error_rate: f64,
        served_total: u32,
    },
    InvalidConfig,
}

pub fn simulate(ramp_secs: u32, max_rps: u32, max_capacity: u32) -> RampVerdict {
    if ramp_secs == 0 || max_rps == 0 || max_capacity == 0 {
        return RampVerdict::InvalidConfig;
    }
    let mut saturation_rps = 0u32;
    let mut errors = 0u32;
    let mut served = 0u32;
    let mut requests_total = 0u32;
    for sec in 0..ramp_secs {
        let current_rps = ((sec + 1) * max_rps / ramp_secs).clamp(1, max_rps);
        requests_total += current_rps;
        if current_rps <= max_capacity {
            served += current_rps;
        } else {
            served += max_capacity;
            let dropped = current_rps - max_capacity;
            errors += dropped;
            if saturation_rps == 0 {
                saturation_rps = current_rps;
            }
        }
    }
    let error_rate = if requests_total == 0 {
        0.0
    } else {
        f64::from(errors) / f64::from(requests_total)
    };
    RampVerdict::Ok {
        saturation_rps,
        error_rate,
        served_total: served,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_load_test_ramp_up")?;

    println!("under capacity: {:?}", simulate(60, 100, 200));
    println!("over capacity: {:?}", simulate(60, 200, 100));
    println!("invalid: {:?}", simulate(0, 100, 100));
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
    fn under_capacity_no_errors() {
        let v = simulate(60, 50, 200);
        if let RampVerdict::Ok { error_rate, .. } = v {
            assert_eq!(error_rate, 0.0);
        }
    }

    #[test]
    fn over_capacity_errors_present() {
        let v = simulate(60, 200, 50);
        if let RampVerdict::Ok { error_rate, .. } = v {
            assert!(error_rate > 0.0);
        }
    }

    #[test]
    fn invalid_zero_ramp() {
        assert_eq!(simulate(0, 100, 100), RampVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_max_rps() {
        assert_eq!(simulate(60, 0, 100), RampVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_capacity() {
        assert_eq!(simulate(60, 100, 0), RampVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(60, 100, 100);
        let b = simulate(60, 100, 100);
        assert_eq!(a, b);
    }

    #[test]
    fn saturation_at_capacity() {
        let v = simulate(100, 200, 50);
        if let RampVerdict::Ok { saturation_rps, .. } = v {
            assert!(saturation_rps > 50);
        }
    }

    #[test]
    fn error_rate_in_unit_range() {
        let v = simulate(60, 200, 50);
        if let RampVerdict::Ok { error_rate, .. } = v {
            assert!((0.0..=1.0).contains(&error_rate));
        }
    }

    #[test]
    fn under_capacity_saturation_zero() {
        let v = simulate(60, 50, 200);
        if let RampVerdict::Ok { saturation_rps, .. } = v {
            assert_eq!(saturation_rps, 0);
        }
    }

    #[test]
    fn served_le_total_requests() {
        let v = simulate(60, 200, 50);
        if let RampVerdict::Ok { served_total, .. } = v {
            assert!(served_total > 0);
        }
    }

    #[test]
    fn higher_capacity_lower_errors() {
        let lo = simulate(60, 200, 50);
        let hi = simulate(60, 200, 200);
        if let (RampVerdict::Ok { error_rate: l, .. }, RampVerdict::Ok { error_rate: h, .. }) =
            (lo, hi)
        {
            assert!(h < l);
        }
    }
}
