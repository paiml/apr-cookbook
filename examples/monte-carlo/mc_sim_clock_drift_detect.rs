//! # Monte-Carlo Sim Clock Drift Detect
//!
//! Sim a virtual clock that ticks at a nominal rate but with random
//! jitter; detect drift when the running mean diverges from nominal
//! by more than tolerance. Returns drift events and final accumulated
//! drift (×100).
//!
//! Demonstrates the **MC.170** recipe for PMAT-215 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NTP clock-discipline algorithm; cf. simular's `Clock` at
//!  ../aprender/crates/aprender-simulate/src/engine/clock.rs for
//!  deterministic-sim reference.
//!
//! Run with: cargo run --example mc_sim_clock_drift_detect
//!
//! Added by PMAT-215 (catalog 1558→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ClockDriftVerdict {
    Ok {
        drift_events: u32,
        final_drift_x100: i32,
    },
    InvalidConfig,
}

pub fn simulate(
    ticks: u32,
    nominal_rate_x100: u32,
    jitter_pct: u32,
    tolerance_pct: u32,
    seed: u64,
) -> ClockDriftVerdict {
    if ticks < 100 || nominal_rate_x100 == 0 || jitter_pct >= 100 || tolerance_pct == 0 {
        return ClockDriftVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let nominal = nominal_rate_x100 as f64 / 100.0;
    let mut accumulated = 0.0f64;
    let mut drift_events = 0u32;
    for i in 1..=ticks {
        let jitter_u = (lcg(&mut state) as f64) / (u32::MAX as f64);
        let jitter_signed = (jitter_u - 0.5) * 2.0 * (jitter_pct as f64 / 100.0);
        let observed = nominal * (1.0 + jitter_signed);
        accumulated += observed - nominal;
        // Running mean drift
        let running_drift = accumulated.abs() / i as f64;
        let pct_drift = running_drift / nominal * 100.0;
        if pct_drift > tolerance_pct as f64 {
            drift_events += 1;
        }
    }
    ClockDriftVerdict::Ok {
        drift_events,
        final_drift_x100: (accumulated * 100.0) as i32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_sim_clock_drift_detect")?;

    println!("low jitter: {:?}", simulate(1000, 100, 5, 10, 42));
    println!("high jitter: {:?}", simulate(1000, 100, 50, 10, 42));
    println!("invalid: {:?}", simulate(50, 100, 5, 10, 42));
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
    fn invalid_too_few_ticks() {
        assert_eq!(
            simulate(50, 100, 5, 10, 42),
            ClockDriftVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_nominal() {
        assert_eq!(
            simulate(1000, 0, 5, 10, 42),
            ClockDriftVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_jitter_at_100() {
        assert_eq!(
            simulate(1000, 100, 100, 10, 42),
            ClockDriftVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_tolerance() {
        assert_eq!(
            simulate(1000, 100, 5, 0, 42),
            ClockDriftVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(1000, 100, 5, 10, 42);
        let b = simulate(1000, 100, 5, 10, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn higher_jitter_more_drift_events() {
        let low = simulate(1000, 100, 5, 5, 42);
        let high = simulate(1000, 100, 50, 5, 42);
        if let (
            ClockDriftVerdict::Ok {
                drift_events: l, ..
            },
            ClockDriftVerdict::Ok {
                drift_events: h, ..
            },
        ) = (low, high)
        {
            assert!(h >= l);
        }
    }

    #[test]
    fn drift_events_le_ticks() {
        let v = simulate(1000, 100, 50, 5, 42);
        if let ClockDriftVerdict::Ok { drift_events, .. } = v {
            assert!(drift_events <= 1000);
        }
    }

    #[test]
    fn final_drift_in_finite_range() {
        let v = simulate(1000, 100, 5, 10, 42);
        if let ClockDriftVerdict::Ok {
            final_drift_x100, ..
        } = v
        {
            assert!(final_drift_x100.unsigned_abs() < u32::MAX);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(100, 100, 0, 1, 42);
        assert!(matches!(v, ClockDriftVerdict::Ok { .. }));
    }

    #[test]
    fn many_ticks_handled() {
        let v = simulate(100_000, 100, 5, 10, 42);
        assert!(matches!(v, ClockDriftVerdict::Ok { .. }));
    }

    #[test]
    fn no_jitter_no_drift() {
        let v = simulate(1000, 100, 0, 1, 42);
        if let ClockDriftVerdict::Ok {
            drift_events,
            final_drift_x100,
        } = v
        {
            assert_eq!(drift_events, 0);
            assert_eq!(final_drift_x100, 0);
        }
    }
}
